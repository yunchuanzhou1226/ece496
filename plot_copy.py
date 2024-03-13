from utils import OutputParser, ActionNormalizer
from circuit_graph import GraphLDO
from ldo_env import LDOEnv
from pathlib import Path
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pickle

import numpy as np

from datetime import datetime
date = datetime.today().strftime('%Y-%m-%d')

CktGraph = GraphLDO
SCH_PATH = (
    Path(__file__)
    .parents[1]
    .joinpath(CktGraph.schematic_path)
)
num_steps = 20000
file_name = 'memory_GraphLDO_2024-02-06-22_noise=uniform_reward=-0.26_ActorCriticGCN_rew_eng=True'
#rews_buf_GraphLDO_2023-11-29-16_noise=uniform_reward=-0.95_ActorCriticGCN_rew_eng=True.npy
with open(SCH_PATH.joinpath(f'./saved_memories/{file_name}.pkl'), 'rb') as memory_file:
    memory = pickle.load(memory_file)




PM_list = []
reward_list = []
eposide_list = []
PSRR_10_list = []
PSRR_s1_list = []
PSRR_l1_list = []
score = 0
count=0
for i in range(num_steps):
    info = memory.info_buf[i]
    if(i%10 == 0):
        PM_list.append(info.get('Loop-gain PM at high load current (deg)'))
        PSRR_10_list.append(info.get('PSRR worst at high load current (dB) < 10kHz'))
        PSRR_s1_list.append(info.get('PSRR worst at high load current (dB) < 1MHz'))
        PSRR_l1_list.append(info.get('PSRR worst at high load current (dB) > 1MHz'))
    reward = info.get('reward')
    reward_list.append(reward)
    
    if(reward<0):
        score +=reward
    else:
        score +=reward
        #if(len(eposide_list)<=2000):
        eposide_list.append(score)
        score = 0
        count = 0
    count+=1
    if (count%10 == 0):
        eposide_list.append(score)
        score = 0
        count = 0
    


plt.figure(1)
plt.plot(PM_list,label='Loop-gain PM (deg) at max load')
plt.xlabel("x10 steps")
plt.ylabel("degree")
plt.title("Loop-gain PM in each steps of training")
plt.axhline(45,0,1,c='r',label='target')
plt.yticks([45,0,20,40,60,80],['$45$','$0$','$20$','$40$','$60$','$80$'])
plt.legend()
plt.figure(2)
plt.plot(PSRR_10_list,label='PSRR worst (dB) < 10kHz')
plt.xlabel("x10 steps")
plt.ylabel('dB') #-30
plt.title("PSRR worst (dB) < 10kHz in each steps of training")
plt.axhline(-30,0,1,c='r',label='target')
plt.yticks([-30,-60,-40,-20,0,20],['$-30$','$-60$','$-40$','$-20$','$0$','$20$'])
plt.legend()
plt.figure(3)
plt.plot(PSRR_s1_list,label = 'PSRR worst (dB) < 1MHz')
plt.xlabel("x10 steps")
plt.ylabel('dB') #-20
plt.title("PSRR worst (dB) < 1MHz in each steps of training")
plt.axhline(-20,0,1,c='r',label='target')
plt.yticks([-50,-40,-30,-20,-10,0,10,20],['$-50$','$-40$','$-30$','$-20$','$-10$','$0$','$10$','$20$'])
plt.legend()
plt.figure(4)
plt.plot(PSRR_l1_list,label = 'PSRR worst (dB) > 1MHz')
plt.xlabel("x10 steps")
plt.ylabel('dB') #-5
plt.title("PSRR worst (dB) > 1MHz in each steps of training")
plt.yticks([-5,-30,-20,-10,0,10],['$-5$','$-30$','$-20$','$-10$','$0$','$10$'])
plt.axhline(-5,0,1,c='r',label='target')
plt.legend()
#plt.savefig()
#plt.close()
###plt.plot(eposide_list)
plt.figure(5)
plt.xlabel("eposide")
plt.ylabel("reward scores")
plt.title("reward scores in each eposide of training")
plt.plot(eposide_list)
#plt.savefig()
#plt.close()

plt.show()

#print(info.get('Loop-gain PM (deg) at max load')) 