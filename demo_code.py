from models import ActorCriticGCN
from circuit_graph import GraphLDO
from gymnasium.envs.registration import register
from utils import ActionNormalizer, save_action_to_schematic
import torch
import gymnasium as gym
import numpy as np
from tabulate import tabulate

#model_weight = "/groups/czzzgrp/Cadence_lib/Simulation/Low_Dropout_With_Diff_Pair/spectre/schematic/saved_weights/Actor_GraphLDO_2024-01-19-20_noise=uniform_reward=-0.24_ActorCriticGCN_rew_eng=True.pth"



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


def do_design(weight_path, num_steps):
    model_weight = weight_path

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

    register(
            id = env_id,
            entry_point = 'ldo_env:LDOEnv',
            max_episode_steps = 10
    )

    env = gym.make(env_id, **specs_dict)
    print("Register the environment success")
    
    np.random.seed(496)
    state, info = env.reset()
    state = np.float64(state)

    selected_action = model(torch.FloatTensor(state)).detach().numpy()  # in (-1, 1)
    selected_action = selected_action.flatten()
    # print_result_design(act_norm.action(selected_action))

    count = 0
    best_reward = float("-inf")
    best_action = None
    while info['reward'] != 0.0 and count < num_steps:
        state, _, _, _, _ = env.unwrapped.step(selected_action)
        state = np.float64(state)
        selected_action = model( torch.FloatTensor(state)).detach().numpy()  # in (-1, 1)
        selected_action = selected_action.flatten()
        selected_action = np.random.uniform(np.clip(selected_action-0.05, -1, 1), 
                                            np.clip(selected_action+0.05, -1, 1))
        # print_result_design(act_norm.action(selected_action))

        if (info['reward'] > best_reward):
            best_reward = info['reward']
            best_action = selected_action

        count += 1
    print("+++++++++++++++++++++++++++++++++++")
    print(best_reward)
    print("+++++++++++++++++++++++++++++++++++")    
    return best_reward, best_action

    #save_action_to_schematic(ldo_graph, best_action)
candidate_models = ["/groups/czzzgrp/step_models/saved_weights/Actor_2024-02-07-01_reward=-0.26_step=4000.pth",
                    "/groups/czzzgrp/step_models/saved_weights/Actor_2024-02-07-02_reward=-0.26_step=4500.pth",
                    "/groups/czzzgrp/step_models/saved_weights/Actor_2024-02-07-03_reward=-0.26_step=6000.pth",
                    "/groups/czzzgrp/step_models/saved_weights/Actor_2024-02-07-04_reward=-0.26_step=7500.pth",
                    "/groups/czzzgrp/step_models/saved_weights/Actor_2024-02-07-05_reward=-0.26_step=9000.pth"
                    ]
discarded_model = []
candidate_best_reward = [-100] * len(candidate_models)
candidate_best_action = [None] * len(candidate_models)
for i in range(0, len(candidate_models)):
    if (candidate_models[i] in discarded_model):
        print("model dicarded")
        continue
    print(candidate_models[i])
    curr_reward, curr_act = do_design(candidate_models[i], 10)
    if (curr_reward > candidate_best_reward[i]):
        candidate_best_reward[i] = curr_reward
        candidate_best_action[i] = curr_act
    else:
        discarded_model.append(candidate_models[i])

print(candidate_best_reward)
