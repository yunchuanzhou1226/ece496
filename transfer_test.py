import torch
import numpy as np
import os
from pathlib import Path
import json
from tabulate import tabulate
import gymnasium as gym
import pickle as pkl

import pickle

from circuit_graph import GraphLDO
from utils import ActionNormalizer
from ddpg import DDPGAgent
from models import ActorCriticGCN

from ldo_env import LDOEnv

from datetime import datetime
date = datetime.today().strftime('%Y-%m-%d-%H')
save_path = Path(__file__).parents[1].joinpath("step_models/transient_models")
def start_design(weight_path, max_steps, save = False):

    # PWD = os.getcwd()
    # SPICE_NETLIST_DIR = f'{PWD}/simulations'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    transfer_agent_path = Path(__file__).parents[1].joinpath(weight_path)
    with open(transfer_agent_path, 'rb') as pickle_agent:
        transfer_agent = pkl.load(pickle_agent)

    """ Regsiter the environemnt to gymnasium""" 
    from gymnasium.envs.registration import register

    env_id = 'ldo-v0'

    env_dict = gym.envs.registration.registry.copy()

    print("De-register any environment with same id")
    for env in env_dict:
        if env_id in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry[env]

    # register environment with new spec
    specs_dict = {
        "Vdrop_target": 0.2,
        "PSRR_target_1kHz": 10**(-30/20), 
        "PSRR_target_10kHz": 10**(-30/20), 
        "PSRR_target_1MHz": 10**(-30/20),
        "PSRR_target_above_1MHz":  10**(-10/20), 
        "PSRR_1kHz": 1e3, "PSRR_10kHz": 1e4, "PSRR_1MHz": 1e6,
        "phase_margin_target": 90, 
        "Vreg": 1.2, "Vref": 1.2, "GND": 0, "Vdd": 1.4,
        "Vload_reg_delta_target": 1.2*0.02, "Iq_target": 200e-6,
        "Vline_reg_delta_target": 1.2*0.02,
        "transient": True
    }

    register(
            id = env_id,
            entry_point = 'ldo_env:LDOEnv',
            max_episode_steps = 10
    )

    env = gym.make(env_id, **specs_dict)
    print("Register the environment success")

    CktGraph = GraphLDO
    ldo_graph = GraphLDO()

    agent = DDPGAgent(
        env, 
        CktGraph(),
        transfer_agent.actor,
        transfer_agent.critic,
        100000,
        transfer_agent.batch_size,
        transfer_agent.noise_sigma,
        transfer_agent.noise_sigma_min,
        transfer_agent.noise_sigma_decay,
        initial_random_steps=5,
        noise_type=transfer_agent.noise_type, 
    )

    # train the agent
    num_steps = max_steps
    agent.train(num_steps, step_save=False)
    memory = agent.memory
    rews_buf = memory.rews_buf[:num_steps]
    info  = memory.info_buf[:num_steps]
    best_design = np.argmax(rews_buf)
    best_action = memory.acts_buf[best_design]
    best_reward = np.max(rews_buf)
    agent.env.step(best_action) # run the simulations

    if save:
        model_weight_actor = agent.actor.state_dict()
        save_name_actor = f"Actor_{date}_unreal_reward={best_reward:.2f}.pth"
        
        model_weight_critic = agent.critic.state_dict()
        save_name_critic = f"Critic_{date}_unreal_reward={best_reward:.2f}.pth"
        
        torch.save(model_weight_actor, save_path.joinpath(save_name_actor))
        torch.save(model_weight_critic, save_path.joinpath(save_name_critic))
        print("Actor and Critic weights have been saved!")

        # save memory
        with open(save_path.joinpath(f'memory_{date}_unreal_reward={best_reward:.2f}.pkl'), 'wb') as memory_file:
            pickle.dump(memory, memory_file)

        np.save(save_path.joinpath(f'rews_buf_{date}_unreal_reward={best_reward:.2f}'), rews_buf)

        # save agent
        with open(save_path.joinpath(f'DDPGAgent_{date}_unreal_reward={best_reward:.2f}.pkl'), 'wb') as agent_file:
            pickle.dump(agent, agent_file)

    return best_reward, best_action, best_design, info

model_path = "step_models/saved_agents/DDPGAgent_2024-02-07-02_reward=-0.26_step=5500.pkl"
#best_reward, best_action, best_design, info = start_design(model_path, 200, save = True)


train_memory = Path(__file__).parents[1].joinpath("step_models/transient_models/memory_2024-03-04-01_unreal_reward=-3.15.pkl")
agent_path = Path(__file__).parents[1].joinpath("step_models/transient_models/DDPGAgent_2024-03-04-01_unreal_reward=-3.15.pkl")
with open(train_memory , 'rb') as pickle_file:
    memory = pickle.load(pickle_file)
with open(agent_path , 'rb') as pickle_file:
    best_agent = pickle.load(pickle_file)
rews_buf = memory.rews_buf[:100]
best_design = np.argmax(rews_buf)
best_action = memory.acts_buf[best_design]
best_reward = np.max(rews_buf)
best_info  = memory.info_buf[best_design]

print(best_agent.env.unwrapped.low_load_current_tran_vreg_result_high - best_agent.env.unwrapped.low_load_current_tran_vreg_result_low)
print(best_agent.env.unwrapped.high_load_current_tran_vreg_result_high - best_agent.env.unwrapped.high_load_current_tran_vreg_result_low)

final_table = tabulate(
            [
                ['Drop-out voltage (mV)', best_info['Drop-out voltage (mV)'], "<200"],
                # ['Drop-out voltage (mV)', best_agent.env.Vdrop*1e3, best_agent.env.Vdrop_target*1e3],


                ['PSRR worst at low load current (dB) < 10kHz', best_info['PSRR worst at low load current (dB) < 10kHz'], f'< {20*np.log10(best_agent.env.unwrapped.PSRR_target_10kHz)}'],
                ['PSRR worst at low load current (dB) < 1MHz', best_info['PSRR worst at low load current (dB) < 1MHz'], f'< {20*np.log10(best_agent.env.unwrapped.PSRR_target_1MHz)}'],
                ['PSRR worst at low load current (dB) > 1MHz', best_info['PSRR worst at low load current (dB) > 1MHz'], f'< {20*np.log10(best_agent.env.unwrapped.PSRR_target_above_1MHz)}'],
                ['PSRR worst at high load current (dB) < 10kHz', best_info['PSRR worst at high load current (dB) < 10kHz'], f'< {20*np.log10(best_agent.env.unwrapped.PSRR_target_10kHz)}'],
                ['PSRR worst at high load current (dB) < 1MHz', best_info['PSRR worst at high load current (dB) < 1MHz'], f'< {20*np.log10(best_agent.env.unwrapped.PSRR_target_1MHz)}'],
                ['PSRR worst at high load current (dB) > 1MHz', best_info['PSRR worst at high load current (dB) > 1MHz'], f'< {20*np.log10(best_agent.env.unwrapped.PSRR_target_above_1MHz)}'],
                
                ['Loop-gain PM at low load current (deg)', best_info['Loop-gain PM at low load current (deg)'], f'> {best_agent.env.unwrapped.phase_margin_target}'],
                ['Loop-gain PM at high load current (deg)', best_info['Loop-gain PM at high load current (deg)'], f'> {best_agent.env.unwrapped.phase_margin_target}'],
                
                ['Iq (uA)', best_info['Iq (uA)'], f'< {best_agent.env.unwrapped.Iq_target*1e6}'],
                # ['Iq (uA)', best_agent.env.Iq*1e6, best_agent.env.Iq_target*1e6],
                ['Cdec (pF)', best_info['Cdec (pF)'], 'N/A'],

                ['low_load_Vreg',best_agent.env.unwrapped.low_load_current_tran_vreg_result_high - best_agent.env.unwrapped.low_load_current_tran_vreg_result_low, "<0.06"],
                ['high_load_Vreg', best_agent.env.unwrapped.high_load_current_tran_vreg_result_high - best_agent.env.unwrapped.high_load_current_tran_vreg_result_low, "<0.06"],
                
                # ['Cdec area score', best_agent.env.Cdec_area_score, 'N/A'],
                ['Reward', best_info['reward'], '']
            ],
        headers=['param', 'num', 'target'], tablefmt='orgtbl', numalign='right', floatfmt=".2f"
        )

best_action = ActionNormalizer(best_agent.env.unwrapped.action_space_low, best_agent.env.unwrapped.action_space_high).action(best_action)
print(final_table)
print(best_agent.env.unwrapped.low_load_current_tran_vreg_result_high - best_agent.env.unwrapped.low_load_current_tran_vreg_result_low)
print(best_agent.env.unwrapped.high_load_current_tran_vreg_result_high - best_agent.env.unwrapped.high_load_current_tran_vreg_result_low)


"""
candidate_models = ["step_models/saved_agents/DDPGAgent_2024-02-07-02_reward=-0.26_step=4500.pkl",
                    "step_models/saved_agents/DDPGAgent_2024-02-07-02_reward=-0.26_step=5500.pkl",
                    "step_models/saved_agents/DDPGAgent_2024-02-07-03_reward=-0.26_step=6500.pkl",
                    "step_models/saved_agents/DDPGAgent_2024-02-07-04_reward=-0.26_step=7500.pkl",
                    "step_models/saved_agents/DDPGAgent_2024-02-07-05_reward=-0.26_step=9000.pkl"
                    ]
discarded_model = []
num_it = [10, 10, 30, 50]
candidate_best_reward = [-100] * len(candidate_models)
candidate_best_action = [None] * len(candidate_models)
one_model_left = False

for it in num_it:
    if (not one_model_left):
        for i in range(0, len(candidate_models)):
            if (candidate_models[i] in discarded_model):
                print("model dicarded")
                continue
            print(candidate_models[i])
            curr_reward, curr_act = start_design(candidate_models[i], it)
            if (curr_reward > candidate_best_reward[i]):
                candidate_best_reward[i] = curr_reward
                candidate_best_action[i] = curr_act
            else:
                discarded_model.append(candidate_models[i])
                if (len(candidate_models) - len(discarded_model) == 1):
                    one_model_left = True
    else:
        for i in range(0, len(candidate_models)):
            if (candidate_models[i] not in discarded_model):
                curr_reward, curr_act = start_design(candidate_models[i], it)
                if (curr_reward > candidate_best_reward[i]):
                    candidate_best_reward[i] = curr_reward
                    candidate_best_action[i] = curr_act
            else:
                continue

    print(candidate_best_reward)
    print(candidate_best_action)
    print(len(discarded_model))
"""