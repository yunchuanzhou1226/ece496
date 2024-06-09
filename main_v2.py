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
from utils import OutputParser, generate_transistor_internal_parameter_mean_std, save_action_to_schematic, ActionNormalizer
from ddpg import DDPGAgent
from models import ActorCriticGCN

from ldo_env import LDOEnv

from datetime import datetime

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

default_specs_dict = {
    "Vdrop_target": 0.2,
    "PSRR_target_1kHz": 10**(-30/20), 
    "PSRR_target_10kHz": 10**(-30/20), 
    "PSRR_target_1MHz": 10**(-20/20),
    "PSRR_target_above_1MHz":  10**(0/20), 
    "PSRR_1kHz": 1e3, "PSRR_10kHz": 1e4, "PSRR_1MHz": 1e6,
    "phase_margin_target": 45, 
    "Vreg": 1.2, "Vref": 1.2, "GND": 0, "Vdd": 1.4,
    "Vload_reg_delta_target": 1.2*0.02, "Iq_target": 200e-6,
    "Vline_reg_delta_target": 1.2*0.02,
    "transient": True
}

date = datetime.today().strftime('%Y-%m-%d-%H')
transfer_model_path = "step_models/saved_agents/DDPGAgent_2024-03-19-02_reward=-0.24_step=5500.pkl"

def train(num_steps = 10000, batch_size = 128, initial_random_steps = 1000, save = True, step_save = False):
    CktGraph = GraphLDO
    SCH_PATH = (
        Path(__file__)
        .parents[1]
        .joinpath(CktGraph.schematic_path)
    )
    GNN = ActorCriticGCN # you can select other GNN
    rew_eng = CktGraph().rew_eng

    """ Regsiter the environemnt to gymnasium """

    from gymnasium.envs.registration import register


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
            max_episode_steps = 10,
    )
    env = gym.make(env_id) 
    print("Register environment success")

    """ Run intial op experiment """
    mean_std_collected = True
    if not mean_std_collected:
        generate_transistor_internal_parameter_mean_std(CktGraph(), num_rnd_sample = 2000)

    """ Do the training """
    # parameters
    memory_size = 100000
    noise_sigma = 2 # noise volume
    noise_sigma_min = 0.1
    noise_sigma_decay = 0.999 # if 1 means no decay
    noise_type = 'uniform' 

    agent = DDPGAgent(
        env, 
        CktGraph(),
        GNN().Actor(CktGraph()),
        GNN().Critic(CktGraph()),
        memory_size, 
        batch_size,
        noise_sigma,
        noise_sigma_min,
        noise_sigma_decay,
        initial_random_steps=initial_random_steps,
        noise_type=noise_type, 
    )

    # train the agent
    agent.train(num_steps=num_steps, step_save=step_save)

    """ Replay the best results """
    memory = agent.memory
    rews_buf = memory.rews_buf[:num_steps]
    info  = memory.info_buf[:num_steps]
    best_design = np.argmax(rews_buf)
    best_action = memory.acts_buf[best_design]
    best_reward = np.max(rews_buf)
    agent.env.step(best_action) # run the simulations

    results = OutputParser(CktGraph())
    op_results = results.dcOp()

    # saved agent's actor and critic network, save memory buffer and agent
    if save == True:
        model_weight_actor = agent.actor.state_dict()
        save_name_actor = f"Actor_{CktGraph().__class__.__name__}_{date}_noise={noise_type}_reward={best_reward:.2f}_{GNN().__class__.__name__}_rew_eng={rew_eng}.pth"
        
        model_weight_critic = agent.critic.state_dict()
        save_name_critic = f"Critic_{CktGraph().__class__.__name__}_{date}_noise={noise_type}_reward={best_reward:.2f}_{GNN().__class__.__name__}_rew_eng={rew_eng}.pth"
        
        torch.save(model_weight_actor, SCH_PATH.joinpath("saved_weights/" + save_name_actor))
        torch.save(model_weight_critic, SCH_PATH.joinpath("saved_weights/" + save_name_critic))
        print("Actor and Critic weights have been saved!")

        # save memory
        with open(SCH_PATH.joinpath(f'saved_memories/memory_{CktGraph().__class__.__name__}_{date}_noise={noise_type}_reward={best_reward:.2f}_{GNN().__class__.__name__}_rew_eng={rew_eng}.pkl'), 'wb') as memory_file:
            pickle.dump(memory, memory_file)

        np.save(SCH_PATH.joinpath(f'saved_memories/rews_buf_{CktGraph().__class__.__name__}_{date}_noise={noise_type}_reward={best_reward:.2f}_{GNN().__class__.__name__}_rew_eng={rew_eng}'), rews_buf)

        # save agent
        with open(SCH_PATH.joinpath(f'saved_agents/DDPGAgent_{CktGraph().__class__.__name__}_{date}_noise={noise_type}_reward={best_reward:.2f}_{GNN().__class__.__name__}_rew_eng={rew_eng}.pkl'), 'wb') as agent_file:
            pickle.dump(agent, agent_file)

        save_action_to_schematic(CktGraph(), best_action)

def transfer_learning(specs_dict, weight_path, num_steps = 100, save = True):        
    save_path = Path(__file__).parents[1].joinpath("step_models/transient_models")
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
        noise_sigma_decay = 0.94,
        initial_random_steps=5,
        noise_type=transfer_agent.noise_type, 
    )

    # train the agent
    agent.train(num_steps, step_save=False)
    memory = agent.memory
    rews_buf = memory.rews_buf[:num_steps]
    info  = memory.info_buf[:num_steps]
    best_design = np.argmax(rews_buf)
    best_action = memory.acts_buf[best_design]
    best_reward = np.max(rews_buf)
    best_info = memory.info_buf[best_design]
    agent.env.step(best_action) # run the simulations
    
    if save:
        model_weight_actor = agent.actor.state_dict()
        save_name_actor = f"Actor_{date}_TL_reward={best_reward:.2f}.pth"
        
        model_weight_critic = agent.critic.state_dict()
        save_name_critic = f"Critic_{date}_TL_reward={best_reward:.2f}.pth"
        
        torch.save(model_weight_actor, save_path.joinpath(save_name_actor))
        torch.save(model_weight_critic, save_path.joinpath(save_name_critic))
        print("Actor and Critic weights have been saved!")

        # save memory
        with open(save_path.joinpath(f'memory_{date}_TL_reward={best_reward:.2f}.pkl'), 'wb') as memory_file:
            pickle.dump(memory, memory_file)

        np.save(save_path.joinpath(f'rews_buf_{date}_TL_reward={best_reward:.2f}'), rews_buf)

        # save agent
        with open(save_path.joinpath(f'DDPGAgent_{date}_TL_reward={best_reward:.2f}.pkl'), 'wb') as agent_file:
            pickle.dump(agent, agent_file)

        # save back to Cadence
        save_action_to_schematic(CktGraph(), best_action)

    # normalize agent
    best_action = ActionNormalizer(agent.env.unwrapped.action_space_low, agent.env.unwrapped.action_space_high).action(best_action)

    return best_reward, best_action, best_design, best_info 

