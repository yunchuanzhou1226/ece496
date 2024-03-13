from tkinter import *
from tkinter import ttk
from models import ActorCriticGCN
from circuit_graph import GraphLDO
from gymnasium.envs.registration import register
from utils import ActionNormalizer
import torch
import gymnasium as gym
import numpy as np
from tabulate import tabulate

#default value
specs_dict = {
    "Vdrop_target": 0.2,
    "PSRR_target_1kHz": 10**(-30/20), 
    "PSRR_target_10kHz": 10**(-30/20), 
    "PSRR_target_1MHz": 10**(-20/20),
    "PSRR_target_above_1MHz":  10**(-5/20), 
    "PSRR_1kHz": 1e3, "PSRR_10kHz": 1e4, "PSRR_1MHz": 1e6,
    "phase_margin_target": 45, 
    "Vreg": 1.2, "Vref": 1.2, "GND": 0, "Vdd": 1.4,
    "Vload_reg_delta_target": 1.2*0.02, "Iq_target": 200e-6,
    "Vline_reg_delta_target": 1.2*0.02,
}

def print_result_design(action):
    print(tabulate(
            [
                ['M1 & M2 width'    , action[0]         , 'um'],
                ['M1 & M2 length'   , action[1]         , 'um'],
                ['M3 & M4 width'    , action[2]         , 'um'],
                ['M3 & M4 length'   , action[3]         , 'um'],
                ['M5 width'         , action[4]         , 'um'],
                ['M5 length'        , action[5]         , 'um'],
                ['Mp width'         , action[6]         , 'um'],
                ['Mp length'        , action[7]         , 'um'],
                ['Mp multiplier'    , int(action[8])    , ''  ],
                ['Vb'               , action[9]         , 'V' ],
                ['Rfb multiplier'   , int(action[10])   , ''  ],
                ['Cfb multiplier'   , int(action[11])   , ''  ],
                ['Cdec multiplier'  , int(action[12])   , ''  ],
            ],
        headers=['Parameter', 'number', 'unit'], tablefmt='orgtbl', numalign='right', floatfmt=".2f"
        ))
    return

range_dict = {
    "Vdrop_target": ['0.18','0.22'],
    "PSRR_target_1kHz": ['0.9*10**(-30/20)','1.1*10**(-30/20)'],
    "PSRR_target_10kHz": ['0.9*10**(-30/20)','1.1*10**(-30/20)'],
    "PSRR_target_1MHz": ['0.9*10**(-20/20)','1.1*10**(-20/20)'],
    "PSRR_target_above_1MHz":  ['0.9*10**(-5/20)','1.1*10**(-5/20)'],
    "PSRR_1kHz": ['0.9*1e3','1.1*1e3'], "PSRR_10kHz": ['0.9*1e4','1.1*1e4'], "PSRR_1MHz": ['0.9*1e6','1.1*1e6'],
    "phase_margin_target": ['0.9*45','1.1*45'],
    "Vreg": ['0.9*1.2','1.1*1.2'], "Vref": ['0.9*1.2','1.1*1.2'], "GND": ['0','0'], "Vdd": ['0.9*1.4','1.1*1.4'],
    "Vload_reg_delta_target": ['0.9*1.2*0.02','1.1*1.2*0.02'], "Iq_target": ['0.9*200e-6','1.1*200e-6'],
    "Vline_reg_delta_target": ['0.9*1.2*0.02','1.1*1.2*0.02']
}


def fetch(entries,root):
    text = 'All specs are set successfully, training starts.'
    #check specs range
    flag = True
    # bounds = np.zeros([16,2])
    # for i in range(16):
    #     bounds[i][0] = 1
    
    #count = 0
    for entry in entries:
        var = entry[0]
        lower_bound = eval(range_dict[var][0])
        upper_bound = eval(range_dict[var][1])
        try:
            num  = eval(entry[1].get())
            specs_dict[var] = num
        except:
            text = 'Error, please check your input of '+var+'.'
            flag = False
            break
        if num < lower_bound or num > upper_bound:
            text = 'The spec of '+var+' is not reasonable, please try another value'
            flag = False
            break


    #print(specs_dict)
    lb = Label(root, text=text,     
			width=100,               
			height=10,              
			justify='left',         
			anchor='nw',
            font = ('Arial',15)            
            )
    lb.place(x=500,y=650)

    if flag == False :
        return

    ###register
    model_weight = "/groups/czzzgrp/step_models/saved_weights/Actor_2024-02-07-04_reward=-0.26_step=7500.pth"
    ldo_graph = GraphLDO()
    act_norm = ActionNormalizer(ldo_graph.action_space_low, ldo_graph.action_space_high)
    model = ActorCriticGCN.Actor(ldo_graph)
    model.load_state_dict(torch.load(model_weight))
    env_id = 'ldo-v0'
    env_dict = gym.envs.registration.registry.copy()
    print("De-register any environment with same id")
    for env in env_dict:
        if env_id in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry[env]
    register(
        id = env_id,
        entry_point = 'ldo_env:LDOEnv',
        max_episode_steps = 10
    )
    env = gym.make(env_id, **specs_dict)
    print("Register the environment success")
###do_design()
    np.random.seed(496)
    state, info = env.reset()
    state = np.float64(state)

    selected_action = model(torch.FloatTensor(state)).detach().numpy()  # in (-1, 1)
    selected_action = selected_action.flatten()
    print_result_design(act_norm.action(selected_action))

    count = 0
    best_reward = float("-inf")
    best_action = None
    while info['reward'] != 0.0 and count < 10:
        state, _, _, _, _ = env.unwrapped.step(selected_action)
        state = np.float64(state)
        selected_action = model( torch.FloatTensor(state)).detach().numpy()  # in (-1, 1)
        selected_action = selected_action.flatten()
        selected_action = np.random.uniform(np.clip(selected_action-0.05, -1, 1), 
                                            np.clip(selected_action+0.05, -1, 1))
        print_result_design(act_norm.action(selected_action))

        if (info['reward'] > best_reward):
            best_reward = info['reward']
            best_action = selected_action

        count += 1
    ##need to add codes to save training results


def makeform(root, vars):
   entries = []
   for var in vars:
      row = Frame(root)
      lab = Label(row, width=25, text=var, anchor='w')
      ent = Entry(row)
      #set default value
      ent.insert(0,specs_dict[var])
      row.pack(side=TOP, padx=5, pady=5)
      lab.pack(side=LEFT)
      ent.pack(side=RIGHT, expand=YES)
      entries.append((var, ent))
   return entries

vars = "Vdrop_target","PSRR_target_1kHz", "PSRR_target_10kHz", "PSRR_target_1MHz","PSRR_target_above_1MHz", "PSRR_1kHz", "PSRR_10kHz", "PSRR_1MHz","phase_margin_target", "Vreg", "Vref", "GND", "Vdd","Vload_reg_delta_target", "Iq_target","Vline_reg_delta_target"


root = Tk()
root.title("Capstone")
root.geometry('1200x800')
#root.configure(bg="white")
ents = makeform(root, vars)
#root.bind('<Return>', (lambda event, e=ents: fetch(e)))
b1 = Button(root, text='Train',width='6',height='6',
      command=(lambda e=ents,r=root: fetch(e,r)))
b2 = Button(root, text='Quit', width='6',height='6',command=root.quit)
#b1.configure(bg="white")
#b2.configure(bg="white")
b1.place(x=450,y=500)
b2.place(x=550,y=500)

root.mainloop()