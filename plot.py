#from utils import OutputParser, ActionNormalizer
from circuit_graph import GraphLDO
#from ldo_env import LDOEnv
from pathlib import Path
from IPython.display import clear_output
import matplotlib.pyplot as plt
from pylab import *
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
num_steps = 10000
file_name = 'memory_GraphLDO_2024-02-06-22_noise=uniform_reward=-0.26_ActorCriticGCN_rew_eng=True'
#rews_buf_GraphLDO_2023-11-22_noise=uniform_reward=0.00_ActorCriticGCN_rew_eng=True.npy
with open(SCH_PATH.joinpath(f'./saved_memories/{file_name}.pkl'), 'rb') as memory_file:
    memory = pickle.load(memory_file)


def average_window(rews_buf, window=100):
    avg_rewards=[]
    for k in range(len(rews_buf)):
        if k >= window:
            avg_reward = np.mean(rews_buf[k-window:k])
        else:
            avg_reward = np.inf
        avg_rewards.append(avg_reward)
    
    return avg_rewards

# PM_list = []
# reward_list = []
# eposide_list = []
# PSRR_10_list = []
# PSRR_s1_list = []
# PSRR_l1_list = []
# score = 0
# count=0
info = memory.info_buf[:num_steps]

# for i in range(num_steps):
#     info = memory.info_buf[i]
#     if(i%10 == 0):
#         PM_list.append(info.get('Loop-gain PM (deg) at max load'))
#         PSRR_10_list.append(info.get('PSRR worst (dB) < 10kHz'))
#         PSRR_s1_list.append(info.get('PSRR worst (dB) < 1MHz'))
#         PSRR_l1_list.append(info.get('PSRR worst (dB) > 1MHz'))
#     reward = info.get('reward')
#     reward_list.append(reward)
    
#     if(reward<0):
#         score +=reward
#     else:
#         score +=reward
#         #if(len(eposide_list)<=2000):
#         eposide_list.append(score)
#         score = 0
#         count = 0
#     count+=1
#     if (count%10 == 0):
#         eposide_list.append(score)
#         score = 0
#         count = 0
PM_array = np.array([info[i]['Loop-gain PM at high load current (deg) at max load'] for i in range(num_steps)])
PSRR_10_array = np.array([info[i]['PSRR worst at high load current (dB) < 10kHz'] for i in range(num_steps)])
PSRR_s1_array = np.array([info[i]['PSRR worst at high load current (dB) < 1MHz'] for i in range(num_steps)])
PSRR_l1_array = np.array([info[i]['PSRR worst at high load current (dB) > 1MHz'] for i in range(num_steps)])
PM_larray = np.array([info[i]['Loop-gain PM at low load current (deg) at max load'] for i in range(num_steps)])
PSRR_10_larray = np.array([info[i]['PSRR worst at low load current (dB) < 10kHz'] for i in range(num_steps)])
PSRR_s1_larray = np.array([info[i]['PSRR worst at low load current (dB) < 1MHz'] for i in range(num_steps)])
PSRR_l1_larray = np.array([info[i]['PSRR worst at low load current (dB) > 1MHz'] for i in range(num_steps)])
reward_array = np.array([info[i]['reward'] for i in range(num_steps)])
Iq_array = np.array([info[i]['Iq (uA)'] for i in range(num_steps)])


LINE_WIDTH      = 7
TITLE_SIZE      = 34
Y_LABEL_SIZE    = 32
Y_TICK_SIZE     = 32
X_LABEL_SIZE    = 32
X_TICK_SIZE     = 32
GRIDLINE_WIDTH  = 3
GRIDLINE_TRANSPARENCY = 0.4
LEGEND_SIZE     = 18
rc('axes', linewidth=3) # set the value globally

    
#print(np.array(average_window(PM_array)).shape)
'''
plt.figure(1, figsize=(10,10))
# plt.subplot(2,2,1)
plt.plot(np.array(average_window(PM_array)),  label='Loop-gain PM with high IL', color='red',  linewidth = LINE_WIDTH)
plt.plot(np.array(average_window(PM_larray)), label='Loop-gain PM with low IL',  color='blue', linewidth = LINE_WIDTH)
plt.title("Loop-gain Phase Margin vs Step", fontsize=TITLE_SIZE)
plt.xlabel("Steps", fontsize=X_LABEL_SIZE)
plt.xticks(fontsize=X_TICK_SIZE)
plt.ylabel("Phase Margin [deg]", fontsize=Y_LABEL_SIZE)
plt.yticks([45,0,20,40,60,80],['$45$','$0$','$20$','$40$','$60$','$80$'], fontsize=Y_TICK_SIZE)
plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY)
plt.axhline(45,0,1,c='r',label='Target',linewidth = LINE_WIDTH)
plt.legend(fontsize=LEGEND_SIZE)

plt.savefig("./pictures/PM plot.png")
plt.close()


plt.figure(2, figsize=(13,13))
# plt.subplot(2,2,2)
plt.plot(np.array(average_window(PSRR_10_array)),  label='Worst PSRR with high IL', color='red',  linewidth = LINE_WIDTH)
plt.plot(np.array(average_window(PSRR_10_larray)), label='Worst PSRR with low IL',  color='blue', linewidth = LINE_WIDTH)
plt.title("Low frequency (< 10kHz) worst PSRR vs Step", fontsize=TITLE_SIZE)
plt.xlabel("Steps", fontsize=X_LABEL_SIZE)
plt.xticks(fontsize=X_TICK_SIZE)
plt.ylabel('PSRR [dB]', fontsize=Y_LABEL_SIZE) #-30
plt.yticks([-30,-60,-40,-20,0,20],['$-30$','$-60$','$-40$','$-20$','$0$','$20$'], fontsize=Y_TICK_SIZE)
plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY)
plt.axhline(-30,0,1,c='r',label='Target',linewidth = LINE_WIDTH)
plt.legend(fontsize=LEGEND_SIZE)

plt.savefig("./pictures/Low frequency PSRR plot.png")
plt.close()


plt.figure(3, figsize=(13,12))
# plt.subplot(2,2,3)
plt.plot(np.array(average_window(PSRR_s1_array)),  label = 'Worst PSRR with high IL', color='red',  linewidth = LINE_WIDTH)
plt.plot(np.array(average_window(PSRR_s1_larray)), label = 'Worst PSRR with low IL',  color='blue', linewidth = LINE_WIDTH)
plt.title("Mid frequency (< 1MHz) worst PSRR vs Step", fontsize=TITLE_SIZE)
plt.xlabel("Steps", fontsize=X_LABEL_SIZE)
plt.xticks(fontsize=X_TICK_SIZE)
plt.ylabel('PSRR [dB]', fontsize=Y_LABEL_SIZE) #-20
plt.yticks([-50,-40,-30,-20,-10,0,10,20],['$-50$','$-40$','$-30$','$-20$','$-10$','$0$','$10$','$20$'], fontsize=Y_TICK_SIZE)
plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY)
plt.axhline(-20,0,1,c='r',label='Target',linewidth = LINE_WIDTH)
plt.legend(fontsize=LEGEND_SIZE)

plt.savefig("./pictures/Mid frequency PSRR plot.png")
plt.close()


plt.figure(4, figsize=(13,12))
# plt.subplot(2,2,4)
plt.plot(np.array(average_window(PSRR_l1_array)),  label = 'Worst PSRR with high IL', color='red',  linewidth = LINE_WIDTH)
plt.plot(np.array(average_window(PSRR_l1_larray)), label = 'Worst PSRR with low IL',  color='blue', linewidth = LINE_WIDTH)
plt.title("High frequency (> 1MHz) worst PSRR vs Step", fontsize=TITLE_SIZE)
plt.xlabel("Steps", fontsize=X_LABEL_SIZE)
plt.xticks(fontsize=X_TICK_SIZE)
plt.ylabel('PSRR [dB]', fontsize=Y_LABEL_SIZE) #-5
plt.yticks([-5,-30,-20,-10,0,10],['$-5$','$-30$','$-20$','$-10$','$0$','$10$'], fontsize=Y_TICK_SIZE)
plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY)
plt.axhline(0,0,1,c='r',label='Target',linewidth = LINE_WIDTH)
plt.legend(fontsize=LEGEND_SIZE)

plt.savefig("./pictures/High frequency PSRR plot.png")
plt.close()
'''

plt.figure(5, figsize=(13,12))
plt.plot(np.array(average_window(reward_array)),label = 'Reward Scores', linewidth = LINE_WIDTH,color='k')
plt.title("Reward Scores vs Step", fontsize=TITLE_SIZE,color='k')
plt.xlabel("Steps", fontsize=X_LABEL_SIZE,color='k')
plt.xticks(fontsize=X_TICK_SIZE,color='k')
plt.ylabel("Reward Scores", fontsize=Y_LABEL_SIZE,color='k')
plt.yticks(fontsize=Y_TICK_SIZE,color='k')
plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY,color='k')

plt.savefig("./pictures/Rewards Score plot.png",transparent = True)
plt.close()


# plt.figure(6)
# plt.subplot(2,2,1)
# plt.plot(np.array(average_window(PM_larray)),label='Loop-gain PM', linewidth = LINE_WIDTH)
# plt.title("Loop-gain Phase Margin at low IL vs Step", fontsize=TITLE_SIZE)
# plt.xlabel("Steps", fontsize=X_LABEL_SIZE)
# plt.xticks(fontsize=X_TICK_SIZE)
# plt.ylabel("Phase Margin [deg]", fontsize=Y_LABEL_SIZE)
# plt.yticks([45,0,20,40,60,80],['$45$','$0$','$20$','$40$','$60$','$80$'], fontsize=Y_TICK_SIZE)
# plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY)
# plt.axhline(45,0,1,c='r',label='target')
# plt.legend(fontsize=LEGEND_SIZE)

# plt.subplot(2,2,2)
# plt.plot(np.array(average_window(PSRR_10_larray)),label='PSRR worst', linewidth = LINE_WIDTH)
# plt.title("Low frequency (< 10kHz) worst PSRR at low IL vs Step", fontsize=TITLE_SIZE)
# plt.xlabel("Steps", fontsize=X_LABEL_SIZE)
# plt.xticks(fontsize=X_TICK_SIZE)
# plt.ylabel('PSRR [dB]', fontsize=Y_LABEL_SIZE) #-30
# plt.yticks([-30,-60,-40,-20,0,20],['$-30$','$-60$','$-40$','$-20$','$0$','$20$'], fontsize=Y_TICK_SIZE)
# plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY)
# plt.axhline(-30,0,1,c='r',label='target')
# plt.legend(fontsize=LEGEND_SIZE)

# plt.subplot(2,2,3)
# plt.plot(np.array(average_window(PSRR_s1_larray)),label = 'PSRR worst', linewidth = LINE_WIDTH)
# plt.title("Mid frequency (< 1MHz) worst PSRR at low IL vs Step", fontsize=TITLE_SIZE)
# plt.xlabel("Steps", fontsize=X_LABEL_SIZE)
# plt.xticks(fontsize=X_TICK_SIZE)
# plt.ylabel('PSRR [dB]', fontsize=Y_LABEL_SIZE) #-20
# plt.yticks([-50,-40,-30,-20,-10,0,10,20],['$-50$','$-40$','$-30$','$-20$','$-10$','$0$','$10$','$20$'], fontsize=Y_TICK_SIZE)
# plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY)
# plt.axhline(-20,0,1,c='r',label='target')
# plt.legend(fontsize=LEGEND_SIZE)

# plt.subplot(2,2,4)
# plt.plot(np.array(average_window(PSRR_l1_larray)),label = 'PSRR worst', linewidth = LINE_WIDTH)
# plt.title("High frequency (> 1MHz) worst PSRR at low IL vs Step", fontsize=TITLE_SIZE)
# plt.xlabel("Steps", fontsize=X_LABEL_SIZE)
# plt.xticks(fontsize=X_TICK_SIZE)
# plt.ylabel('PSRR [dB]', fontsize=Y_LABEL_SIZE) #-5
# plt.yticks([-5,-30,-20,-10,0,10],['$-5$','$-30$','$-20$','$-10$','$0$','$10$'], fontsize=Y_TICK_SIZE)
# plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY)
# plt.axhline(-5,0,1,c='r',label='target')
# plt.legend(fontsize=LEGEND_SIZE)

#plt.savefig()
#plt.close()

'''
plt.figure(7, figsize=(12,12))
plt.plot(np.array(average_window(Iq_array)),label = 'Iq', linewidth = LINE_WIDTH)
plt.title("Iq vs Step", fontsize=TITLE_SIZE)
plt.xlabel("Steps", fontsize=X_LABEL_SIZE)
plt.xticks(fontsize=X_TICK_SIZE)
plt.ylabel("Iq [uA]", fontsize=Y_LABEL_SIZE)
plt.yticks(fontsize=Y_TICK_SIZE)
plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY)

plt.savefig("./pictures/Iq plot.png")
plt.close()


#plt.show()
'''