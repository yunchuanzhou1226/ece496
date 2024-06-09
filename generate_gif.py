import numpy as np
import matplotlib
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from pathlib import Path
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pickle
import numpy as np
from datetime import datetime
date = datetime.today().strftime('%Y-%m-%d')

sys.path.append('..')
schematic_path = "Cadence_lib/Simulation/Low_Dropout_With_Diff_Pair/spectre/schematic"

SCH_PATH = (
    Path(__file__)
    .parents[2]
    .joinpath(schematic_path)
)
num_steps = 10000
file_name = 'memory_GraphLDO_2024-02-06-22_noise=uniform_reward=-0.26_ActorCriticGCN_rew_eng=True'
with open(SCH_PATH.joinpath(f'./saved_memories/{file_name}.pkl'), 'rb') as memory_file:
    memory = pickle.load(memory_file)


def average_window(rews_buf, window=500):
    avg_rewards=[]
    for k in range(len(rews_buf)):
        if k >= window:
            avg_reward = np.mean(rews_buf[k-window:k])
        else:
            avg_reward = np.inf
        avg_rewards.append(avg_reward)
    
    return avg_rewards

info = memory.info_buf[:num_steps]

high_load_current_PM_step_array = np.array([info[i]['Loop-gain PM at high load current (deg) at max load'] for i in range(num_steps)])
high_load_current_PSRR_lt_10K_step_array = np.array([info[i]['PSRR worst at high load current (dB) < 10kHz'] for i in range(num_steps)])
high_load_current_PSRR_lt_1M_step_array = np.array([info[i]['PSRR worst at high load current (dB) < 1MHz'] for i in range(num_steps)])
high_load_current_PSRR_gt_1M_step_array = np.array([info[i]['PSRR worst at high load current (dB) > 1MHz'] for i in range(num_steps)])
high_load_current_AC_freq_array = np.array([info[i]['high_load_current_ac_results'][0] for i in range(num_steps)])
high_load_current_PSRR_freq_array = np.array([info[i]['high_load_current_ac_results'][1] for i in range(num_steps)])
high_load_current_STB_freq_array = np.array([info[i]['high_load_current_stb_results'][0] for i in range(num_steps)])
high_load_current_Gain_freq_array = np.array([info[i]['high_load_current_stb_results'][1] for i in range(num_steps)])
high_load_current_PM_freq_array = np.array([info[i]['high_load_current_stb_results'][2] for i in range(num_steps)])

low_load_current_PM_step_array = np.array([info[i]['Loop-gain PM at low load current (deg) at max load'] for i in range(num_steps)])
low_load_current_PSRR_lt_10K_step_array = np.array([info[i]['PSRR worst at low load current (dB) < 10kHz'] for i in range(num_steps)])
low_load_current_PSRR_lt_1M_step_array = np.array([info[i]['PSRR worst at low load current (dB) < 1MHz'] for i in range(num_steps)])
low_load_current_PSRR_gt_1M_step_array = np.array([info[i]['PSRR worst at low load current (dB) > 1MHz'] for i in range(num_steps)])
low_load_current_AC_freq_array = np.array([info[i]['low_load_current_ac_results'][0] for i in range(num_steps)])
low_load_current_PSRR_freq_array = np.array([info[i]['low_load_current_ac_results'][1] for i in range(num_steps)])
low_load_current_STB_freq_array = np.array([info[i]['low_load_current_stb_results'][0] for i in range(num_steps)])
low_load_current_Gain_freq_array = np.array([info[i]['low_load_current_stb_results'][1] for i in range(num_steps)])
low_load_current_PM_freq_array = np.array([info[i]['low_load_current_stb_results'][2] for i in range(num_steps)])


reward_step_array = np.array([info[i]['reward'] for i in range(num_steps)])

high_load_current_PM_step_array = np.array(average_window(high_load_current_PM_step_array))
high_load_current_PSRR_lt_10K_step_array = np.array(average_window(high_load_current_PSRR_lt_10K_step_array))
high_load_current_PSRR_lt_1M_step_array = np.array(average_window(high_load_current_PSRR_lt_1M_step_array))
high_load_current_PSRR_gt_1M_step_array = np.array(average_window(high_load_current_PSRR_gt_1M_step_array))

low_load_current_PM_step_array = np.array(average_window(low_load_current_PM_step_array))
low_load_current_PSRR_lt_10K_step_array = np.array(average_window(low_load_current_PSRR_lt_10K_step_array))
low_load_current_PSRR_lt_1M_step_array = np.array(average_window(low_load_current_PSRR_lt_1M_step_array))
low_load_current_PSRR_gt_1M_step_array = np.array(average_window(low_load_current_PSRR_gt_1M_step_array))

reward_step_array = np.array(average_window(reward_step_array))


plots = {}
fig,axs = plt.subplots(2,4, figsize=(32, 16))
plots['PM vs step at low load current'],                    = axs[0,0].plot([], [], 'k-', label='Low current')
plots['PM vs step at low load current cursor'],             = axs[0,0].plot([], [], 'ko')
plots['PM vs step at high load current'],                   = axs[0,0].plot([], [], 'm-', label='High current')
plots['PM vs step at high load current cursor'],            = axs[0,0].plot([], [], 'mo')
plots['PSRR < 10kHz vs step at low load current'],          = axs[0,1].plot([], [], 'k-', label='Low current')
plots['PSRR < 10kHz vs step at low load current cursor'],   = axs[0,1].plot([], [], 'ko')
plots['PSRR < 1MHz vs step at low load current'],           = axs[0,2].plot([], [], 'k-', label='Low current')
plots['PSRR < 1MHz vs step at low load current cursor'],    = axs[0,2].plot([], [], 'ko')
plots['PSRR > 1MHz vs step at low load current'],           = axs[0,3].plot([], [], 'k-', label='Low current')
plots['PSRR > 1MHz vs step at low load current cursor'],    = axs[0,3].plot([], [], 'ko')
plots['PSRR < 10kHz vs step at high load current'],         = axs[0,1].plot([], [], 'm-', label='High current')
plots['PSRR < 10kHz vs step at high load current cursor'],  = axs[0,1].plot([], [], 'mo')
plots['PSRR < 1MHz vs step at high load current'],          = axs[0,2].plot([], [], 'm-', label='High current')
plots['PSRR < 1MHz vs step at high load current cursor'],   = axs[0,2].plot([], [], 'mo')
plots['PSRR > 1MHz vs step at high load current'],          = axs[0,3].plot([], [], 'm-', label='High current')
plots['PSRR > 1MHz vs step at high load current cursor'],   = axs[0,3].plot([], [], 'mo')

gain_ax = axs[1,1]
PM_ax   = axs[1,0]
plots['Gain vs freq at low load current'],                  = gain_ax.plot( [], [], 'k-', label='Low current')
plots['Gain vs freq at high load current'],                 = gain_ax.plot( [], [], 'm-', label='High current')
plots['PM vs freq at low load current'],                    = PM_ax.plot(   [], [], 'k-', label='Low current')
plots['PM vs freq at high load current'],                   = PM_ax.plot(   [], [], 'm-', label='High current')
plots['PSRR vs freq at low load current'],                  = axs[1,2].plot([], [], 'k-', label='Low current')
plots['PSRR vs freq at high load current'],                 = axs[1,2].plot([], [], 'm-', label='High current')

plots['Reward'],        = axs[1,3].plot([], [], 'k-')
plots['Reward cursor'], = axs[1,3].plot([], [], 'ko')

axs[0,0].set(xlabel='Step', ylabel='Phase Margin [deg]', title='Phase Margin vs Steps', xlim=[0,num_steps], ylim=[0,120])
axs[0,1].set(xlabel='Step', ylabel='PSRR [dB]', title='PSRR at low frequency vs Steps', xlim=[0,num_steps], ylim=[-60,10])
axs[0,2].set(xlabel='Step', ylabel='PSRR [dB]', title='PSRR at mid frequency vs Steps', xlim=[0,num_steps], ylim=[-60,10])
axs[0,3].set(xlabel='Step', ylabel='PSRR [dB]', title='PSRR at high frequency vs Steps', xlim=[0,num_steps], ylim=[-60,10])


gain_ax.set( xlabel='Frequency', ylabel='Gain [dB]', title='Gain vs Frequency', xlim=[1,np.max(low_load_current_STB_freq_array)], ylim=[np.min(low_load_current_Gain_freq_array), np.max(low_load_current_Gain_freq_array)], xscale='log', yscale='log')
PM_ax.set(   xlabel='Frequency', ylabel='Phase Margin [deg]', title='PM vs Frequency', xlim=[1,np.max(low_load_current_STB_freq_array)], ylim=[np.min(low_load_current_PM_freq_array), np.max(low_load_current_PM_freq_array)+10], xscale='log')
axs[1,2].set(xlabel='Frequency', ylabel='PSRR [dB]', title='PSRR vs Frequency', xlim=[1,np.max(low_load_current_AC_freq_array)], ylim=[np.min(high_load_current_PSRR_freq_array), np.max(high_load_current_PSRR_freq_array)], xscale='log', yscale='log')
axs[1,3].set(xlabel='Step', ylabel='Reward', title='Reward vs Steps', xlim=[0,num_steps], ylim=[-50,5])

axs[0,0].legend(fontsize=14)
axs[0,1].legend(fontsize=14)
axs[0,2].legend(fontsize=14)
axs[0,3].legend(fontsize=14)
axs[1,0].legend(fontsize=14)
axs[1,1].legend(fontsize=14)
axs[1,2].legend(fontsize=14)

fig.patch.set_alpha(0.)

def func(x):
    return np.sin(x)*3

def func2(x):
    return np.cos(x)*3

metadata = dict(title='Movie', artist='Team 748')
writer = PillowWriter(fps=15, metadata=metadata)

xlist = list(range(len(reward_step_array)))

with writer.saving(fig, "Results.gif", 100):

    # Plot lines and cursors
    for xval in range(0,len(reward_step_array), 20):
        # plots['PM vs step at low load current'].set_data(xlist[:xval],low_load_current_PM_step_array[:xval])
        # plots['PM vs step at low load current cursor'].set_data([xval],[low_load_current_PM_step_array[xval]])
        # plots['PM vs step at high load current'].set_data(xlist[:xval],high_load_current_PM_step_array[:xval])
        # plots['PM vs step at high load current cursor'].set_data([xval],[high_load_current_PM_step_array[xval]])
        # plots['PSRR < 10kHz vs step at low load current'].set_data(xlist[:xval],low_load_current_PSRR_lt_10K_step_array[:xval])
        # plots['PSRR < 10kHz vs step at low load current cursor'].set_data([xval],[low_load_current_PSRR_lt_10K_step_array[xval]])
        # plots['PSRR < 1MHz vs step at low load current'].set_data(xlist[:xval],low_load_current_PSRR_lt_1M_step_array[:xval])
        # plots['PSRR < 1MHz vs step at low load current cursor'].set_data([xval],[low_load_current_PSRR_lt_1M_step_array[xval]])
        # plots['PSRR > 1MHz vs step at low load current'].set_data(xlist[:xval],low_load_current_PSRR_gt_1M_step_array[:xval])
        # plots['PSRR > 1MHz vs step at low load current cursor'].set_data([xval],[low_load_current_PSRR_gt_1M_step_array[xval]])
        # plots['PSRR < 10kHz vs step at high load current'].set_data(xlist[:xval],high_load_current_PSRR_lt_10K_step_array[:xval])
        # plots['PSRR < 10kHz vs step at high load current cursor'].set_data([xval],[high_load_current_PSRR_lt_10K_step_array[xval]])
        # plots['PSRR < 1MHz vs step at high load current'].set_data(xlist[:xval],high_load_current_PSRR_lt_1M_step_array[:xval])
        # plots['PSRR < 1MHz vs step at high load current cursor'].set_data([xval],[high_load_current_PSRR_lt_1M_step_array[xval]])
        # plots['PSRR > 1MHz vs step at high load current'].set_data(xlist[:xval],high_load_current_PSRR_gt_1M_step_array[:xval])
        # plots['PSRR > 1MHz vs step at high load current cursor'].set_data([xval],[high_load_current_PSRR_gt_1M_step_array[xval]])

        # plots['Gain vs freq at low load current'].set_data(low_load_current_STB_freq_array[xval],[low_load_current_Gain_freq_array[xval]])
        # plots['Gain vs freq at high load current'].set_data(high_load_current_STB_freq_array[xval],[high_load_current_Gain_freq_array[xval]])
        # plots['PM vs freq at low load current'].set_data(low_load_current_STB_freq_array[xval],[low_load_current_PM_freq_array[xval]])
        # plots['PM vs freq at high load current'].set_data(high_load_current_STB_freq_array[xval],[high_load_current_PM_freq_array[xval]])
        # plots['PSRR vs freq at low load current'].set_data(low_load_current_AC_freq_array[xval],[low_load_current_PSRR_freq_array[xval]])
        # plots['PSRR vs freq at high load current'].set_data(high_load_current_AC_freq_array[xval],[high_load_current_PSRR_freq_array[xval]])



        plots['Reward'].set_data(xlist[:xval],reward_step_array[:xval])
        plots['Reward cursor'].set_data([xval],[reward_step_array[xval]])

        
        writer.grab_frame(transparent=True)
