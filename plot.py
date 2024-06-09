from pathlib import Path
from IPython.display import clear_output
import matplotlib.pyplot as plt
from pylab import *
import pickle
import sys

sys.path.append('..')



import numpy as np

from datetime import datetime
date = datetime.today().strftime('%Y-%m-%d')


schematic_path = "step_models"
SCH_PATH = (
    Path(__file__)
    .parents[2]
    .joinpath(schematic_path)
)
num_steps = 6000
file_name = 'memory_2024-03-19-02_reward=-0.24_step=6000'
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
PM_array = np.array([info[i]['Loop-gain PM at high load current (deg)'] for i in range(num_steps)])
PSRR_10_array = np.array([info[i]['PSRR worst at high load current (dB) < 10kHz'] for i in range(num_steps)])
PSRR_s1_array = np.array([info[i]['PSRR worst at high load current (dB) < 1MHz'] for i in range(num_steps)])
PSRR_l1_array = np.array([info[i]['PSRR worst at high load current (dB) > 1MHz'] for i in range(num_steps)])
PM_larray = np.array([info[i]['Loop-gain PM at low load current (deg)'] for i in range(num_steps)])
PSRR_10_larray = np.array([info[i]['PSRR worst at low load current (dB) < 10kHz'] for i in range(num_steps)])
PSRR_s1_larray = np.array([info[i]['PSRR worst at low load current (dB) < 1MHz'] for i in range(num_steps)])
PSRR_l1_larray = np.array([info[i]['PSRR worst at low load current (dB) > 1MHz'] for i in range(num_steps)])
reward_array = np.array([info[i]['reward'] for i in range(num_steps)])
Iq_array = np.array([info[i]['Iq (uA)'] for i in range(num_steps)])


LINE_WIDTH      = 4
TITLE_SIZE      = 34
Y_LABEL_SIZE    = 32
Y_TICK_SIZE     = 32
X_LABEL_SIZE    = 32
X_TICK_SIZE     = 32
GRIDLINE_WIDTH  = 3
GRIDLINE_TRANSPARENCY = 0.4
LEGEND_BACKGROUND_TRANSPARENCY = 0.2
LEGEND_SIZE     = 18
rc('axes', linewidth=3) # set the value globally

    
#print(np.array(average_window(PM_array)).shape)

fig = plt.figure(1, figsize=(10,10))
ax = fig.add_subplot(111)
plt.axhline(90,0,1,c='cyan',label='Target',linewidth = LINE_WIDTH)
ax.plot(np.array(average_window(PM_array)),  label='Loop-gain PM with high IL', color='red',  linewidth = LINE_WIDTH)
ax.plot(np.array(average_window(PM_larray)), label='Loop-gain PM with low IL',  color='green', linewidth = LINE_WIDTH)
plt.title("Loop-gain Phase Margin vs Step", fontsize=TITLE_SIZE, color='k')
plt.xlabel("Steps", fontsize=X_LABEL_SIZE, color='k')
plt.xticks(fontsize=X_TICK_SIZE, color='k')
plt.ylabel("Phase Margin [deg]", fontsize=Y_LABEL_SIZE, color='k')
plt.yticks([0,20,40,60,80,90],['$0$','$20$','$40$','$60$','$80$','$90$'], fontsize=Y_TICK_SIZE, color='k')
plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY, color='k')
plt.legend(fontsize=LEGEND_SIZE, facecolor='white', framealpha = LEGEND_BACKGROUND_TRANSPARENCY, labelcolor='black')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

plt.savefig("./pictures/PM plot.png", transparent = True)
plt.close()


fig = plt.figure(2, figsize=(13,13))
ax = fig.add_subplot(111)
plt.axhline(-30,0,1,c='cyan',label='Target',linewidth = LINE_WIDTH)
ax.plot(np.array(average_window(PSRR_10_array)),  label='Worst PSRR with high IL', color='red',  linewidth = LINE_WIDTH)
ax.plot(np.array(average_window(PSRR_10_larray)), label='Worst PSRR with low IL',  color='green', linewidth = LINE_WIDTH)
plt.title("Low frequency (< 10kHz) worst PSRR vs Step", fontsize=TITLE_SIZE, color='k')
plt.xlabel("Steps", fontsize=X_LABEL_SIZE, color='k')
plt.xticks(fontsize=X_TICK_SIZE, color='k')
plt.ylabel('PSRR [dB]', fontsize=Y_LABEL_SIZE, color='k')
plt.yticks([-60,-50,-40,-30,-20,-10,0,10,20],['$-60$','$-50$','$-40$','$-30$','$-20$','$-10$','$0$','$10$','$20$'], fontsize=Y_TICK_SIZE, color='k')
plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY, color='k')
plt.legend(fontsize=LEGEND_SIZE, facecolor='white', framealpha = LEGEND_BACKGROUND_TRANSPARENCY, labelcolor='black')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

plt.savefig("./pictures/Low frequency PSRR plot.png", transparent = True)
plt.close()


fig = plt.figure(3, figsize=(13,12))
ax = fig.add_subplot(111)
plt.axhline(-30,0,1,c='cyan',label='Target',linewidth = LINE_WIDTH)
ax.plot(np.array(average_window(PSRR_s1_array)),  label = 'Worst PSRR with high IL', color='red',  linewidth = LINE_WIDTH)
ax.plot(np.array(average_window(PSRR_s1_larray)), label = 'Worst PSRR with low IL',  color='green', linewidth = LINE_WIDTH)
plt.title("Mid frequency (< 1MHz) worst PSRR vs Step", fontsize=TITLE_SIZE, color='k')
plt.xlabel("Steps", fontsize=X_LABEL_SIZE, color='k')
plt.xticks(fontsize=X_TICK_SIZE, color='k')
plt.ylabel('PSRR [dB]', fontsize=Y_LABEL_SIZE, color='k')
plt.yticks([-50,-40,-30,-20,-10,0,10,20],['$-50$','$-40$','$-30$','$-20$','$-10$','$0$','$10$','$20$'], fontsize=Y_TICK_SIZE, color='k')
plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY, color='k')
plt.legend(fontsize=LEGEND_SIZE, facecolor='white', framealpha = LEGEND_BACKGROUND_TRANSPARENCY, labelcolor='black')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

plt.savefig("./pictures/Mid frequency PSRR plot.png", transparent = True)
plt.close()


fig = plt.figure(4, figsize=(13,12))
ax = fig.add_subplot(111)
plt.axhline(-10,0,1,c='cyan',label='Target',linewidth = LINE_WIDTH)
ax.plot(np.array(average_window(PSRR_l1_array)),  label = 'Worst PSRR with high IL', color='red',  linewidth = LINE_WIDTH)
ax.plot(np.array(average_window(PSRR_l1_larray)), label = 'Worst PSRR with low IL',  color='green', linewidth = LINE_WIDTH)
plt.title("High frequency (> 1MHz) worst PSRR vs Step", fontsize=TITLE_SIZE, color='k')
plt.xlabel("Steps", fontsize=X_LABEL_SIZE, color='k')
plt.xticks(fontsize=X_TICK_SIZE, color='k')
plt.ylabel('PSRR [dB]', fontsize=Y_LABEL_SIZE, color='k')
plt.yticks([-40,-30,-20,-10,0,10],['$-40$','$-30$','$-20$','$-10$','$0$','$10$'], fontsize=Y_TICK_SIZE, color='k')
plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY, color='k')
plt.legend(fontsize=LEGEND_SIZE, facecolor='white', framealpha = LEGEND_BACKGROUND_TRANSPARENCY, labelcolor='black')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

plt.savefig("./pictures/High frequency PSRR plot.png", transparent = True)
plt.close()


fig = plt.figure(5, figsize=(13,12))
ax = fig.add_subplot(111)
ax.plot(np.array(average_window(reward_array)),label = 'Reward Scores', linewidth = LINE_WIDTH,color='r')
plt.title("Reward Scores vs Step", fontsize=TITLE_SIZE,color='k')
plt.xlabel("Steps", fontsize=X_LABEL_SIZE,color='k')
plt.xticks(fontsize=X_TICK_SIZE,color='k')
plt.ylabel("Reward Scores", fontsize=Y_LABEL_SIZE,color='k')
plt.yticks(fontsize=Y_TICK_SIZE,color='k')
plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY,color='k')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

plt.savefig("./pictures/Rewards Score plot.png", transparent = True)
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


fig = plt.figure(7, figsize=(12,12))
ax = fig.add_subplot(111)
ax.plot(np.array(average_window(Iq_array)),label = 'Iq', linewidth = LINE_WIDTH, color='r')
plt.title("Iq vs Step", fontsize=TITLE_SIZE, color='k')
plt.xlabel("Steps", fontsize=X_LABEL_SIZE, color='k')
plt.xticks(fontsize=X_TICK_SIZE, color='k')
plt.ylabel("Iq [uA]", fontsize=Y_LABEL_SIZE, color='k')
plt.yticks(fontsize=Y_TICK_SIZE, color='k')
plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY, color='k')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

plt.savefig("./pictures/Iq plot.png", transparent = True)
plt.close()



rews_buf = memory.rews_buf[:num_steps]
info  = memory.info_buf[:num_steps]
best_design = np.argmax(rews_buf)
best_action = memory.acts_buf[best_design]
best_reward = np.max(rews_buf)
best_info = memory.info_buf[best_design]

low_load_current_AC_freq_array = np.array(best_info['low_load_current_ac_results'][0])
high_load_current_AC_freq_array = np.array(best_info['high_load_current_ac_results'][0])
low_load_current_PSRR_freq_array = np.array(best_info['low_load_current_ac_results'][1])
high_load_current_PSRR_freq_array = np.array(best_info['high_load_current_ac_results'][1])


fig = plt.figure(8, figsize=(14,12))
ax = fig.add_subplot(111)
ax.set(xlabel='Frequency', ylabel='PSRR [dB]', title='PSRR vs Frequency', xlim=[1,np.max(low_load_current_AC_freq_array)], xscale='log', yscale='log')
ax.plot(low_load_current_AC_freq_array, low_load_current_PSRR_freq_array, label = 'Low Load PSRR', linewidth = LINE_WIDTH, color='g')
ax.plot(high_load_current_AC_freq_array, high_load_current_PSRR_freq_array, label = 'High Load PSRR', linewidth = LINE_WIDTH, color='r')

plt.title("PSRR vs Frequency", fontsize=TITLE_SIZE, color='k')
plt.xlabel("Frequency", fontsize=X_LABEL_SIZE, color='k')
plt.xticks(fontsize=X_TICK_SIZE, color='k')
plt.ylabel("PSRR [dB]", fontsize=Y_LABEL_SIZE, color='k')
plt.yticks([10**(x/20) for x in range(-70,20,10)],['${}$'.format(x) for x in range(-70,20,10)], fontsize=Y_TICK_SIZE, color='k')
plt.grid(linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY, color='k')
plt.legend(fontsize=LEGEND_SIZE, facecolor='white', framealpha = LEGEND_BACKGROUND_TRANSPARENCY, labelcolor='black')

ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

plt.savefig("./pictures/PSRR_Freq_plot.png", transparent = True)
plt.close()

low_load_current_STB_freq_array = np.array(best_info['low_load_current_stb_results'][0])
high_load_current_STB_freq_array = np.array(best_info['high_load_current_stb_results'][0])
low_load_current_PM_freq_array = np.array(best_info['low_load_current_stb_results'][2])
high_load_current_PM_freq_array = np.array(best_info['high_load_current_stb_results'][2])
low_load_current_Gain_freq_array = np.array(best_info['low_load_current_stb_results'][1])
high_load_current_Gain_freq_array = np.array(best_info['high_load_current_stb_results'][1])



fig = plt.figure(9, figsize=(18,12))
ax = fig.add_subplot(111)
ax2 = ax.twinx()

ax2.set(ylabel='Gain [dB]', yscale='log', ylim=[10**(-120/20),10**(60/20)])
ax.set(xlabel='Frequency', ylabel='Phase [deg]', xlim=[1,np.max(low_load_current_STB_freq_array)], ylim=[-180,225], xscale='log')

ax.tick_params(axis='x', colors='k', labelsize=X_TICK_SIZE)
ax.tick_params(axis='y', colors='k', labelsize=Y_TICK_SIZE)
ax.set_yticks([x for x in range(-180,226,45)],['${}$'.format(x) for x in range(-180,226,45)])

ax2.tick_params(axis='x', colors='k', labelsize=X_TICK_SIZE)
ax2.tick_params(axis='y', colors='k', labelsize=Y_TICK_SIZE)
ax2.set_yticks([10**(x/20) for x in range(-120,61,20)],['${}$'.format(x) for x in range(-120,61,20)])

ax.yaxis.label.set_color('k')
ax.xaxis.label.set_color('k')

ax2.yaxis.label.set_color('k')
ax2.xaxis.label.set_color('k')

ax.yaxis.get_label().set_fontsize(Y_LABEL_SIZE)
ax.xaxis.get_label().set_fontsize(X_LABEL_SIZE)
ax2.yaxis.get_label().set_fontsize(Y_LABEL_SIZE)
ax2.xaxis.get_label().set_fontsize(X_LABEL_SIZE)

ax.plot(high_load_current_STB_freq_array, high_load_current_PM_freq_array, label = 'Phase', linewidth = LINE_WIDTH, color='r')
ax.legend(loc=2,fontsize=LEGEND_SIZE, facecolor='white', framealpha = LEGEND_BACKGROUND_TRANSPARENCY, labelcolor='black')
ax2.plot(high_load_current_STB_freq_array, high_load_current_Gain_freq_array, label = 'Gain', linewidth = LINE_WIDTH, color='g')
ax2.legend(loc=1,fontsize=LEGEND_SIZE, facecolor='white', framealpha = LEGEND_BACKGROUND_TRANSPARENCY, labelcolor='black')


plt.title("Loop Gain and Phase (High load)", fontsize=TITLE_SIZE, color='k')
plt.xlabel("Frequency", fontsize=X_LABEL_SIZE, color='k')
plt.xticks(fontsize=X_TICK_SIZE, color='k')

ax.grid(color = 'k', linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY)

ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

ax2.spines['bottom'].set_color('black')
ax2.spines['top'].set_color('black') 
ax2.spines['right'].set_color('black')
ax2.spines['left'].set_color('black')

plt.savefig("./pictures/High_PM_LG_plot.png", transparent = True)
plt.close()





fig = plt.figure(10, figsize=(18,12))
ax = fig.add_subplot(111)
ax2 = ax.twinx()

ax2.set(ylabel='Gain [dB]', yscale='log', ylim=[10**(-120/20),10**(60/20)])
ax.set(xlabel='Frequency', ylabel='Phase [deg]', xlim=[1,np.max(low_load_current_STB_freq_array)], ylim=[-180,225], xscale='log')

ax.tick_params(axis='x', colors='k', labelsize=X_TICK_SIZE)
ax.tick_params(axis='y', colors='k', labelsize=Y_TICK_SIZE)
ax.set_yticks([x for x in range(-180,226,45)],['${}$'.format(x) for x in range(-180,226,45)])

ax2.tick_params(axis='x', colors='k', labelsize=X_TICK_SIZE)
ax2.tick_params(axis='y', colors='k', labelsize=Y_TICK_SIZE)
ax2.set_yticks([10**(x/20) for x in range(-120,61,20)],['${}$'.format(x) for x in range(-120,61,20)])

ax.yaxis.label.set_color('k')
ax.xaxis.label.set_color('k')

ax2.yaxis.label.set_color('k')
ax2.xaxis.label.set_color('k')

ax.yaxis.get_label().set_fontsize(Y_LABEL_SIZE)
ax.xaxis.get_label().set_fontsize(X_LABEL_SIZE)
ax2.yaxis.get_label().set_fontsize(Y_LABEL_SIZE)
ax2.xaxis.get_label().set_fontsize(X_LABEL_SIZE)

ax.plot(low_load_current_STB_freq_array, low_load_current_PM_freq_array, label = 'Phase', linewidth = LINE_WIDTH, color='r')
ax.legend(loc=2,fontsize=LEGEND_SIZE, facecolor='white', framealpha = LEGEND_BACKGROUND_TRANSPARENCY, labelcolor='black')
ax2.plot(low_load_current_STB_freq_array, low_load_current_Gain_freq_array, label = 'Gain', linewidth = LINE_WIDTH, color='g')
ax2.legend(loc=1,fontsize=LEGEND_SIZE, facecolor='white', framealpha = LEGEND_BACKGROUND_TRANSPARENCY, labelcolor='black')


plt.title("Loop Gain and Phase (Low load)", fontsize=TITLE_SIZE, color='k')
plt.xlabel("Frequency", fontsize=X_LABEL_SIZE, color='k')
plt.xticks(fontsize=X_TICK_SIZE, color='k')

ax.grid(color = 'k', linewidth=GRIDLINE_WIDTH, alpha=GRIDLINE_TRANSPARENCY)

ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

ax2.spines['bottom'].set_color('black')
ax2.spines['top'].set_color('black') 
ax2.spines['right'].set_color('black')
ax2.spines['left'].set_color('black')

plt.savefig("./pictures/Low_PM_LG_plot.png", transparent = True)
plt.close()


#plt.show()
