import torch
import numpy as np
import os
from pathlib import Path
import json
from tabulate import tabulate
import gymnasium as gym

import pickle

from circuit_graph import GraphLDO
from utils import OutputParser, generate_transistor_internal_parameter_mean_std, save_action_to_schematic
from ddpg import DDPGAgent
from models import ActorCriticGCN

from ldo_env import LDOEnv

from datetime import datetime
date = datetime.today().strftime('%Y-%m-%d-%H')

# PWD = os.getcwd()
# SPICE_NETLIST_DIR = f'{PWD}/simulations'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
num_steps = 10000
memory_size = 100000
batch_size = 128
noise_sigma = 2 # noise volume
noise_sigma_min = 0.1
noise_sigma_decay = 0.999 # if 1 means no decay
initial_random_steps = 1000
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
agent.train(num_steps=num_steps, step_save=True)

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
save = True
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

