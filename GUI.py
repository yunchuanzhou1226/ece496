from tkinter import *
from tkinter import ttk
# from models import ActorCriticGCN
# from circuit_graph import GraphLDO
# from gymnasium.envs.registration import register
# from utils import ActionNormalizer
# import torch
# import gymnasium as gym
# import numpy as np
# from tabulate import tabulate
from main_v2 import *

#default value

specs_dict = {
    "Vdrop_target": 0.2,
    "PSRR_target_1kHz": 10**(-30/20), 
    "PSRR_target_10kHz": -30, 
    "PSRR_target_1MHz": -20,
    "PSRR_target_above_1MHz":  0, 
    "PSRR_1kHz": 1e3, "PSRR_10kHz": 1e4, "PSRR_1MHz": 1e6,
    "phase_margin_target": 45, 
    "Vreg": 1.2, "Vref": 1.2, "GND": 0, "Vdd": 1.4,
    "Vload_reg_delta_target": 0.05, "Iq_target": 200e-6,
    "Vline_reg_delta_target": 0.05,
    "transient": True
}

range_dict = {
    "Vdrop_target": ['0.18','0.22'],
    "PSRR_target_1kHz": ['10**(-60/20)','10**(0/20)'],
    "PSRR_target_10kHz": ['-60','0'],
    "PSRR_target_1MHz": ['-60','0'],
    "PSRR_target_above_1MHz":  ['-60','0'],
    "phase_margin_target": ['0','90'],
    "Vreg": ['1.2','1.2'], "GND": ['0','0'], "Vdd": ['1.4','1.4'],
    "Vload_reg_delta_target": ['0.01','0.05'], "Iq_target": ['100e-6','400e-6'],
    "Vline_reg_delta_target": ['0.01','0.05']
}



def fetch(entries,root):
    #print(trans_box.get())
    text = 'All specs are set successfully, training starts.'
    #check specs range
    flag = True
    if trans_box.get() == 'On':
        transient = True
    else:
        transient = False

    if save_box.get() == 'On':
        save = True
    else:
        save = False

    

    training_step = eval(step_box.get())
    if isinstance(training_step, int) and training_step > 0:
        pass
    else:
        flag = False
        return
 

    

    specs_dict["transient"] = transient
    specs_dict['GND'] = 0
    specs_dict['Vline_reg_delta_target'] = 1.2*0.05
    specs_dict['PSRR_target_1kHz'] = 10**(-30/20)
    #count = 0
    for entry in entries:
        # print(entry[0])
        var = entry[0]
        # if var == 'GND':
        #     specs_dict[var] = 0
        #     continue
        # if var == 'Vline_reg_delta_target':
        #     specs_dict[var] = 1.2*0.05
        #     continue
        # if var == 'PSRR_target_1kHz':
        #     specs_dict[var] = 10**(-30/20)
        #     continue
        lower_bound = eval(range_dict[var][0])
        upper_bound = eval(range_dict[var][1])
        try:
            num  = eval(entry[1].get())
            if var =='Vload_reg_delta_target':
                specs_dict[var] = 1.2*num
            elif var =='PSRR_target_10kHz':
                specs_dict[var] =  10**(num/20)
            elif var =='PSRR_target_1MHz':
                specs_dict[var] = 10**(num/20)
            elif var =='PSRR_target_above_1MHz':
                specs_dict[var] = 10**(num/20)
            else:
                specs_dict[var] = num
        except:
            text = 'Error, please check your input of '+var+'.'
            flag = False
            break
        if num < lower_bound or num > upper_bound:
            text = 'The spec of '+var+' is not reasonable, please try another value'
            flag = False
            break


    # print(len(specs_dict))
    # import sys
    # sys.exit(0)
    # lb = Label(root, text=text,     
	# 		width=100,               
	# 		height=10,              
	# 		justify='left',         
	# 		anchor='nw',
    #         font = ('Arial',15)            
    #         )
    # lb.place(x=600,y=690)
    lb.configure(text=text)

    if flag == False :
        return

    _,best_action,_,best_info = transfer_learning(specs_dict, "step_models/saved_agents/DDPGAgent_2024-03-19-02_reward=-0.24_step=5500.pkl", num_steps = training_step, save = save)

    best_action = np.around(best_action, decimals=2)

    ##################
    lb.configure(text='Training finishes!')
    result_text = f"Optimized circuit components parameters(size in um):\nM1&M2 width: {best_action[0]}  M1&M2 length: {best_action[1]}\
    M3&M4 width: {best_action[2]}  M3&M4 length: {best_action[3]}\nM5 width: {best_action[4]}  M5 length: {best_action[5]}\
    Mp width: {best_action[6]}  Mp length: {best_action[7]}  Mp multiplicity: {best_action[8]}\
    \nVb: {best_action[9]}  Rfb multiplicity: {best_action[10]}\
    Cfb multiplicity: {best_action[11]}  Cdec multiplicity: {best_action[12]}\
    \nCircuit performance:\
    \nDrop-out voltage (mV): {round(best_info['Drop-out voltage (mV)'])}\
    \nPSRR worst at low load current (dB) < 10kHz: {round(best_info['PSRR worst at low load current (dB) < 10kHz'], 2)}\
    PSRR worst at high load current (dB) < 10kHz: {round(best_info['PSRR worst at high load current (dB) < 10kHz'], 2)}\
    \nPSRR worst at low load current (dB) < 1MHz: {round(best_info['PSRR worst at low load current (dB) < 1MHz'], 2)}\
    PSRR worst at high load current (dB) < 1MHz: {round(best_info['PSRR worst at high load current (dB) < 1MHz'], 2)}\
    \nPSRR worst at low load current (dB) > 1MHz: {round(best_info['PSRR worst at low load current (dB) > 1MHz'], 2)}\
    PSRR worst at high load current (dB) > 1MHz: {round(best_info['PSRR worst at high load current (dB) > 1MHz'], 2)}\
    \nLoop-gain PM at low load current (deg): {round(best_info['Loop-gain PM at low load current (deg)'], 2)}\
    Loop-gain PM at high load current (deg): {round(best_info['Loop-gain PM at high load current (deg)'], 2)}\
    \nIq (uA): {round(best_info['Iq (uA)'], 2)}\
    Cdec (pF): {round(best_info['Cdec (pF)'], 2)}\
    high_load_Vreg: {round(best_info['high_load_Vreg'], 2)}\
    low_load_Vreg: {round(best_info['low_load_Vreg'], 2)}"

    results_lb = Label(root, text=result_text,     
			width=120,               
			height=50,              
			justify='left',         
			anchor='nw',
            font = ('Arial',10)            
            )
    results_lb.place(x=720,y=600)
    ###############

    

    



def makeform(root, vars):
   entries = []
   for var in vars:
      row = Frame(root)
      lab = Label(row, width=25, text=var, anchor='w',font=("Arial", 10, "bold"))
      #lab.config(bg = 'Palegreen',highlightbackground = 'Palegreen')
      ent = Entry(row,width=25,font=("Arial", 10, "bold"))
      #set default value
      ent.insert(0,specs_dict[var])
      row.pack(side=TOP, padx=10, pady=10)
      lab.pack(side=LEFT)
      ent.pack(side=RIGHT, expand=YES)
      entries.append((var, ent))
   return entries

vars = ["Vdd","Vreg","Vdrop_target","PSRR_target_10kHz", "PSRR_target_1MHz","PSRR_target_above_1MHz","phase_margin_target",   "Vload_reg_delta_target", "Iq_target"]
tran = "On","Off"
root = Tk()
root.title("Capstone")
root.geometry('1980x1320')
#root.configure(bg="Palegreen")
ents = makeform(root, vars)
#root.bind('<Return>', (lambda event, e=ents: fetch(e)))
b1 = Button(root, text='Train',width='8',height='4',font=("Arial", 10, "bold"),
      command=(lambda e=ents,r=root: fetch(e,r)))
b2 = Button(root, text='Quit', width='8',height='4',command=root.quit,font=("Arial", 10, "bold"))
default_value = StringVar(value="100")
step_box = Entry(width=25,font=("Arial", 10, "bold"),textvariable=default_value)
step_box_label = Label(width=25, text='training steps', anchor='w',font=("Arial", 10, "bold"))


trans_box =  ttk.Combobox(root,width=25,values = tran,font=("Arial", 10, "bold"))
trans_box.set('Off')
trans_box.configure(state = "readonly")
trans_lab = Label(root, width=25, text='Transient mode', anchor='w',font=("Arial", 10, "bold"))
save_box =  ttk.Combobox(root,width=25,values = tran,font=("Arial", 10, "bold"))
save_box.set('Off')
save_box.configure(state = "readonly")
save_lab = Label(root, width=25, text='Save result', anchor='w',font=("Arial", 10, "bold"))
#b1.configure(bg="white")
#b2.configure(bg="white")
b2.place(x=750,y=520)
b1.place(x=550,y=520)
trans_box.place(x=720,y=390)
trans_lab.place(x=540,y=390)
save_box.place(x=720,y=430)
save_lab.place(x=540,y=430)
step_box.place(x=720,y=470)
step_box_label.place(x=540,y=470)
lb = Label(root, text='Please input the specs and start training',     
			width=100,               
			height=10,              
			justify='left',         
			anchor='nw',
            font = ('Arial',15)            
            )
lb.place(x=400,y=650)
root.mainloop()



# row = Frame(root)
# lab = Label(row, width=25, text='Transient mode', anchor='w')
# row.pack(side=TOP, padx=5, pady=5)
# lab.pack(side=LEFT)
# trans_box.pack(side=RIGHT)




#     print('mmm')
#     ###register
#     model_weight = "/groups/czzzgrp/step_models/saved_weights/Actor_2024-02-07-04_reward=-0.26_step=7500.pth"
#     ldo_graph = GraphLDO()
#     act_norm = ActionNormalizer(ldo_graph.action_space_low, ldo_graph.action_space_high)
#     model = ActorCriticGCN.Actor(ldo_graph)
#     model.load_state_dict(torch.load(model_weight))
#     env_id = 'ldo-v0'
#     env_dict = gym.envs.registration.registry.copy()
#     print("De-register any environment with same id")
#     for env in env_dict:
#         if env_id in env:
#             print("Remove {} from registry".format(env))
#             del gym.envs.registration.registry[env]
#     register(
#         id = env_id,
#         entry_point = 'ldo_env:LDOEnv',
#         max_episode_steps = 10
#     )
#     env = gym.make(env_id, **specs_dict)
#     print("Register the environment success")
# ###do_design()
#     np.random.seed(496)
#     state, info = env.reset()
#     state = np.float64(state)

#     selected_action = model(torch.FloatTensor(state)).detach().numpy()  # in (-1, 1)
#     selected_action = selected_action.flatten()
#     print_result_design(act_norm.action(selected_action))

#     count = 0
#     best_reward = float("-inf")
#     best_action = None
#     while info['reward'] != 0.0 and count < 10:
#         state, _, _, _, _ = env.unwrapped.step(selected_action)
#         state = np.float64(state)
#         selected_action = model( torch.FloatTensor(state)).detach().numpy()  # in (-1, 1)
#         selected_action = selected_action.flatten()
#         selected_action = np.random.uniform(np.clip(selected_action-0.05, -1, 1), 
#                                             np.clip(selected_action+0.05, -1, 1))
#         print_result_design(act_norm.action(selected_action))

#         if (info['reward'] > best_reward):
#             best_reward = info['reward']
#             best_action = selected_action

#         count += 1
#     ##need to add codes to save training results


# def makeform(root, vars):
#     frame = tk.Frame(root)
#     frame.grid(row=0, column=0)
#     entries = []
#     for i, var in enumerate(vars):
#         row = i // 3
#         column = i % 3
#       row = Frame(root)
#       lab = Label(row, width=25, text=var, anchor='w',font=("Arial", 10, "bold"))
#       ent = Entry(row,width=25,font=("Arial", 10, "bold"))
#       #set default value
#       ent.insert(0,specs_dict[var])
#       row.pack(side=TOP, padx=10, pady=10)
#       lab.pack(side=LEFT)
#       ent.pack(side=RIGHT, expand=YES)
#       entries.append((var, ent))
#    return entries