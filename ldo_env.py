import torch
import numpy as np
import os
import json
import gymnasium as gym
from pathlib import Path
from tabulate import tabulate
from gymnasium import spaces
from datetime import datetime

from circuit_graph import GraphLDO 
from utils import OutputParser, ActionNormalizer
from utils import run_sim_with_para, save_param_to_cadence
#from dev_params import DeviceParams

CktGraph = GraphLDO

class LDOEnv(gym.Env, CktGraph): #DeviceParams

    def __init__(self, Vdrop_target=0.2,
                PSRR_target_1kHz = 10**(-30/20), PSRR_target_10kHz=10**(-30/20), PSRR_target_1MHz=10**(-20/20),
                PSRR_target_above_1MHz =  10**(-5/20), PSRR_1kHz = 1e3, PSRR_10kHz = 1e4, PSRR_1MHz = 1e6,
                phase_margin_target = 45, Vreg = 1.2, Vref = 1.2, GND = 0, Vdd = 1.4,
                Vload_reg_delta_target = 1.2*0.02, Iq_target = 200e-6, Vline_reg_delta_target=1.2*0.02, transient = False):
        gym.Env.__init__(self)
        CktGraph.__init__(self)
        #DeviceParams.__init__(self, self.ckt_hierarchy)

        self.CktGraph = CktGraph()
        self.SCH_PATH = (
            Path(__file__)
            .parents[1]
            .joinpath(self.CktGraph.schematic_path)
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float128)
        self.action_space = spaces.Box(low=-1, high=1, shape=self.action_shape, dtype=np.float64)
        self.transient = transient

        # move target to here
        self.Vdrop_target = Vdrop_target  # drop-out voltage
        
        self.PSRR_target_1kHz = PSRR_target_1kHz # in linear scale, equals -30dB
        self.PSRR_target_10kHz = PSRR_target_10kHz # in linear scale, equals -30dB
        self.PSRR_target_1MHz =  PSRR_target_1MHz # in linear scale, equals -20dB
        self.PSRR_target_above_1MHz =  PSRR_target_above_1MHz # in linear scale, equals -5dB
        self.PSRR_1kHz = PSRR_1kHz #  from DC to 1kHz
        self.PSRR_10kHz = PSRR_10kHz #  from DC to 10kHz
        self.PSRR_1MHz = PSRR_1MHz # from DC to 1 MHz
        
        self.phase_margin_target = phase_margin_target # 45 degree PM minimum, this is for the loop-gain
        self.Vreg = Vreg # regulated output
        self.Vref = Vref
        self.GND = GND
        self.Vdd = Vdd
        
        self.Vload_reg_delta_target = Vload_reg_delta_target # load regulartion variation is at most 2% of Vreg when it is switched from ILmin to ILmax
        self.Iq_target = Iq_target #200uA quiescent current maximum
        self.Vline_reg_delta_target = Vline_reg_delta_target  # line reg voltage to be around at most 2% of Vreg when it is at both ILmin and ILmax

        
    def _initialize_simulation(self):
        self.W_M1, self.L_M1, \
        self.W_M3, self.L_M3, \
        self.W_M5, self.L_M5, \
        self.W_Mp, self.L_Mp, self.M_Mp, \
        self.Vb, \
        self.M_Rfb, self.M_Cfb, \
        self.M_Cdec, = \
        np.array([44.47075700387359, 1.618438869714737, 
                  77.25937259197235, 0.5010918378829956,
                  30.432650923728943, 0.5906739979982376,
                  53.32021648064256, 0.5122907906770706, 125,
                  1.1166186541318894,
                  2, 14,
                  55])      

        """Run the initial simulations."""  
        action = np.array([self.W_M1, self.L_M1,
                           self.W_M3, self.L_M3,
                           self.W_M5, self.L_M5,
                           self.W_Mp, self.L_Mp, self.M_Mp,
                           self.Vb,
                           self.M_Rfb,
                           self.M_Cfb,
                           self.M_Cdec])
        
        self.do_simulation(action)
        
    def _do_simulation(self, action: np.array, load_current = "LOW"):
        """
         W_M1 = W_M2, L_M1 = L_M2
         W_M3 = W_M4, L_M3 = L_M4
              
        """ 
        M_M1   = self.CktGraph.M_M1_low
        M_M3   = self.CktGraph.M_M3_low
        M_M5   = self.CktGraph.M_M5_low
        W_Rfb  = self.CktGraph.W_Rfb
        L_Rfb  = self.CktGraph.L_Rfb
        W_Cfb  = self.CktGraph.W_Cfb
        L_Cfb  = self.CktGraph.L_Cfb
        W_Cdec = self.CktGraph.W_Cdec
        L_Cdec = self.CktGraph.L_Cdec
        Vref   = self.CktGraph.Vref
        if load_current == "LOW":
            Il = self.CktGraph.Il_low
        else:
            Il = self.CktGraph.Il_high
        
        
        

        W_M1, L_M1, \
        W_M3, L_M3, \
        W_M5, L_M5,  \
        W_Mp, L_Mp, M_Mp, \
        Vb, \
        M_Rfb, M_Cfb, \
        M_Cdec = action 

        M_Mp = int(M_Mp)
        M_Rfb = int(M_Rfb)
        M_Cfb = int(M_Cfb)
        M_Cdec = int(M_Cdec)

        para_dict = dict(M12 = dict(w = W_M1,
                                    l = L_M1,
                                    m = M_M1),
                         M34 = dict(w = W_M3,
                                    l = L_M3,
                                    m = M_M3),
                          M5 = dict(w = W_M5,
                                    l = L_M5,
                                    m = M_M5),
                          Mp = dict(w = W_Mp,
                                    l = L_Mp,
                                    m = M_Mp),
                         Rfb = dict(w = W_Rfb,
                                    l = L_Rfb,
                                    m = M_Rfb),
                         Cfb = dict(w = W_Cfb,
                                    l = L_Cfb,
                                    m = M_Cfb),
                        Cdec = dict(w = W_Cdec,
                                    l = L_Cdec,
                                    m = M_Cdec),
                          Vb = dict(v = Vb),
                        Vref = dict(v = Vref),
                          Il = dict(i = Il)
                        )
        # save_param_to_cadence(para_dict, self.SCH_PATH)
        run_sim_with_para(para_dict, self.SCH_PATH)
        

    def do_simulation(self, action):
        self._do_simulation(action,"LOW")
        self.low_load_current_sim_results = OutputParser(self.CktGraph)
        self.low_load_current_stb_results = self.low_load_current_sim_results.stb()
        self.low_load_current_dc_results = self.low_load_current_sim_results.dc()
        self.low_load_current_ac_results = self.low_load_current_sim_results.ac()
        self.low_load_current_op_results = self.low_load_current_sim_results.dcOp()
        self.low_load_current_tran_results = self.low_load_current_sim_results.tran()
        self._do_simulation(action,"HIGH")
        self.high_load_current_sim_results = OutputParser(self.CktGraph)
        self.high_load_current_stb_results = self.high_load_current_sim_results.stb()
        self.high_load_current_dc_results = self.high_load_current_sim_results.dc()
        self.high_load_current_ac_results = self.high_load_current_sim_results.ac()
        self.high_load_current_op_results = self.high_load_current_sim_results.dcOp()
        self.high_load_current_tran_results = self.high_load_current_sim_results.tran()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_simulation()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def close(self):
        return None
    
    def step(self, action):
        action = ActionNormalizer(action_space_low  = self.action_space_low, 
                                  action_space_high = self.action_space_high).action(action) # convert [-1.1] range back to normal range
        action = action.astype(object)
        
        print(f"action: {action}")

        self.W_M1, self.L_M1, \
        self.W_M3, self.L_M3, \
        self.W_M5, self.L_M5, \
        self.W_Mp, self.L_Mp, self.M_Mp, \
        self.Vb, \
        self.M_Rfb, self.M_Cfb, \
        self.M_Cdec = action
        
        ''' run simulations '''
        self.do_simulation(action)
        
        ''' get observation '''
        observation = self._get_obs()
        info = self._get_info()

        reward = self.reward
        
        if reward >= 0:
            terminated = True
        else:
            terminated = False
        
        # print any information you want here    
        print(tabulate(
            [
                ['Drop-out voltage (mV)', self.Vdrop*1e3, 200],
                # ['Drop-out voltage (mV)', self.Vdrop*1e3, self.Vdrop_target*1e3],


                ['PSRR worst at low load current (dB) < 10kHz', 20*np.log10(self.low_load_current_PSRR_worst_below_10kHz), f'< {20*np.log10(self.PSRR_target_10kHz)}'],
                ['PSRR worst at low load current (dB) < 1MHz', 20*np.log10(self.low_load_current_PSRR_worst_below_1MHz), f'< {20*np.log10(self.PSRR_target_1MHz)}'],
                ['PSRR worst at low load current (dB) > 1MHz', 20*np.log10(self.low_load_current_PSRR_worst_above_1MHz), f'< {20*np.log10(self.PSRR_target_above_1MHz)}'],
                ['PSRR worst at high load current (dB) < 10kHz', 20*np.log10(self.high_load_current_PSRR_worst_below_10kHz), f'< {20*np.log10(self.PSRR_target_10kHz)}'],
                ['PSRR worst at high load current (dB) < 1MHz', 20*np.log10(self.high_load_current_PSRR_worst_below_1MHz), f'< {20*np.log10(self.PSRR_target_1MHz)}'],
                ['PSRR worst at high load current (dB) > 1MHz', 20*np.log10(self.high_load_current_PSRR_worst_above_1MHz), f'< {20*np.log10(self.PSRR_target_above_1MHz)}'],
                
                ['Loop-gain PM at low load current (deg)', self.low_load_current_phase_margin, f'> {self.phase_margin_target}'],
                ['Loop-gain PM at high load current (deg)', self.high_load_current_phase_margin, f'> {self.phase_margin_target}'],
                
                ['Iq (uA)', self.Iq*1e6, f'< {self.Iq_target*1e6}'],
                # ['Iq (uA)', self.Iq*1e6, self.Iq_target*1e6],
                ['Cdec (pF)', self.OP_Cdec*1e12, 'N/A'],
                
                ['Vdrop score', self.Vdrop_score, '0'],
                
                ['PSRR worst at low load current (dB) < 10kHz score', self.low_load_current_PSRR_worst_below_10kHz_score, '0'],
                ['PSRR worst at low load current (dB) < 1MHz score', self.low_load_current_PSRR_worst_below_1MHz_score, '0'],
                ['PSRR worst at low load current (dB) > 1MHz score', self.low_load_current_PSRR_worst_above_1MHz_score, '0'],
                ['PSRR worst at high load current (dB) < 10kHz score', self.high_load_current_PSRR_worst_below_10kHz_score, '0'],
                ['PSRR worst at high load current (dB) < 1MHz score', self.high_load_current_PSRR_worst_below_1MHz_score, '0'],
                ['PSRR worst at high load current (dB) > 1MHz score', self.high_load_current_PSRR_worst_above_1MHz_score, '0'],
                
                ['PM at low load current score', self.low_load_current_phase_margin_score, '0'],
                ['PM at high load current score', self.high_load_current_phase_margin_score, '0'],
                ['Iq score', self.Iq_score, '0'],

                ['low_load_Vreg', self.low_load_current_tran_vreg_score, 0.06],
                ['high_load_Vreg', self.high_load_current_tran_vreg_score, 0.06],
                
                # ['Cdec area score', self.Cdec_area_score, 'N/A'],
                ['Reward', reward, '']
            ],
        headers=['param', 'num', 'target'], tablefmt='orgtbl', numalign='right', floatfmt=".2f"
        ))
        return observation, reward, terminated, False, info
        
    def _get_obs(self):
        # get the mean and std for normalizing transistor internal prarmeters
        try:
            mean_std_path = self.SCH_PATH.joinpath("transistor_internal_parameter_mean_std.json")
            with open(mean_std_path, "r") as outfile: 
                self.op_mean_std = json.load(outfile)
            self.op_mean = self.op_mean_std['OP_M_mean']
            self.op_std = self.op_mean_std['OP_M_std']
            self.op_mean = np.array([self.op_mean['id'], self.op_mean['gm'], self.op_mean['gds'], self.op_mean['vth'], self.op_mean['vdsat'], self.op_mean['vds'], self.op_mean['vgs']])
            self.op_std  = np.array([ self.op_std['id'],  self.op_std['gm'],  self.op_std['gds'],  self.op_std['vth'],  self.op_std['vdsat'],  self.op_std['vds'],  self.op_std['vgs']])
        except:
            print('You need to run <generate_transistor_internal_parameter_mean_std> to generate mean and std for transistor .OP parameters')
        
        # get observation from dcOp info   
        # OP_results = dict(M1 = dict(ids = 0, gm = 0, gds = 0, vth = 0, vdsat = 0, vds = 0, vgs = 0),
        #                   M2 = dict(ids = 0, gm = 0, gds = 0, vth = 0, vdsat = 0, vds = 0, vgs = 0),
        #                   M3 = dict(ids = 0, gm = 0, gds = 0, vth = 0, vdsat = 0, vds = 0, vgs = 0),
        #                   M4 = dict(ids = 0, gm = 0, gds = 0, vth = 0, vdsat = 0, vds = 0, vgs = 0),
        #                   M5 = dict(ids = 0, gm = 0, gds = 0, vth = 0, vdsat = 0, vds = 0, vgs = 0),
        #                   Mp = dict(ids = 0, gm = 0, gds = 0, vth = 0, vdsat = 0, vds = 0, vgs = 0),
        #                   Vb = dict(v = 0),
        #                   Vdd = dict(v = 0),
        #                   Rfb = dict(res = 0),
        #                   Cfb = dict(cap = 0)
        #                   )
        
        self.OP_M1 = self.low_load_current_op_results['M1']
        self.OP_M1_norm = (np.array([self.OP_M1['id'],
                                self.OP_M1['gm'],
                                self.OP_M1['gds'],
                                self.OP_M1['vth'],
                                self.OP_M1['vdsat'],
                                self.OP_M1['vds'],
                                self.OP_M1['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M2 = self.low_load_current_op_results['M2']
        self.OP_M2_norm = (np.array([self.OP_M2['id'],
                                self.OP_M2['gm'],
                                self.OP_M2['gds'],
                                self.OP_M2['vth'],
                                self.OP_M2['vdsat'],
                                self.OP_M2['vds'],
                                self.OP_M2['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M3 = self.low_load_current_op_results['M3']
        self.OP_M3_norm = (np.abs([self.OP_M3['id'],
                                self.OP_M3['gm'],
                                self.OP_M3['gds'],
                                self.OP_M3['vth'],
                                self.OP_M3['vdsat'],
                                self.OP_M3['vds'],
                                self.OP_M3['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M4 = self.low_load_current_op_results['M4']
        self.OP_M4_norm = (np.abs([self.OP_M4['id'],
                                self.OP_M4['gm'],
                                self.OP_M4['gds'],
                                self.OP_M4['vth'],
                                self.OP_M4['vdsat'],
                                self.OP_M4['vds'],
                                self.OP_M4['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M5 = self.low_load_current_op_results['M5']
        self.OP_M5_norm = (np.abs([self.OP_M5['id'],
                                self.OP_M5['gm'],
                                self.OP_M5['gds'],
                                self.OP_M5['vth'],
                                self.OP_M5['vdsat'],
                                self.OP_M5['vds'],
                                self.OP_M5['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_Mp = self.low_load_current_op_results['Mp']
        self.OP_Mp_norm = (np.array([self.OP_Mp['id'],
                                self.OP_Mp['gm'],
                                self.OP_Mp['gds'],
                                self.OP_Mp['vth'],
                                self.OP_Mp['vdsat'],
                                self.OP_Mp['vds'],
                                self.OP_Mp['vgs']
                                ]) - self.op_mean)/self.op_std
        
        self.Vb = self.low_load_current_op_results['Vb']['v']
        self.Vdd = self.Vdd

        self.OP_Rfb = self.low_load_current_op_results['Rfb']['res']
        self.OP_Rfb_norm = ActionNormalizer(action_space_low=self.Rfb_low, action_space_high=self.Rfb_high).reverse_action(self.OP_Rfb) # convert to (-1, 1)

        self.OP_Cfb = self.low_load_current_op_results['Cfb']['cap']   
        self.OP_Cfb_norm = ActionNormalizer(action_space_low=self.Cfb_low, action_space_high=self.Cfb_high).reverse_action(self.OP_Cfb) # convert to (-1, 1)

        self.OP_Cdec = self.low_load_current_op_results['Cdec']['cap']   
        self.OP_Cdec_norm = ActionNormalizer(action_space_low=self.Cdec_low, action_space_high=self.Cdec_high).reverse_action(self.OP_Cdec) # convert to (-1, 1)

        
        # [OP_Vb,OP_Vdd,OP_GND,     OP_Rfb_norm, OP_Cfb_norm, OP_CL_norm,      OP_M_norm]
        # state shall be in the order of node (node0, node1, ...)
        observation = np.array([[0,0,0,0,0,0,                 self.OP_M1_norm[0],self.OP_M1_norm[1],self.OP_M1_norm[2],self.OP_M1_norm[3],self.OP_M1_norm[4],self.OP_M1_norm[5],self.OP_M1_norm[6]],
                                [0,0,0,0,0,0,                 self.OP_M2_norm[0],self.OP_M2_norm[1],self.OP_M2_norm[2],self.OP_M2_norm[3],self.OP_M2_norm[4],self.OP_M2_norm[5],self.OP_M2_norm[6]],
                                [0,0,0,0,0,0,                 self.OP_M3_norm[0],self.OP_M3_norm[1],self.OP_M3_norm[2],self.OP_M3_norm[3],self.OP_M3_norm[4],self.OP_M3_norm[5],self.OP_M3_norm[6]],
                                [0,0,0,0,0,0,                 self.OP_M4_norm[0],self.OP_M4_norm[1],self.OP_M4_norm[2],self.OP_M4_norm[3],self.OP_M4_norm[4],self.OP_M4_norm[5],self.OP_M4_norm[6]],
                                [0,0,0,0,0,0,                 self.OP_M5_norm[0],self.OP_M5_norm[1],self.OP_M5_norm[2],self.OP_M5_norm[3],self.OP_M5_norm[4],self.OP_M5_norm[5],self.OP_M5_norm[6]],
                                [0,0,0,0,0,0,                 self.OP_Mp_norm[0],self.OP_Mp_norm[1],self.OP_Mp_norm[2],self.OP_Mp_norm[3],self.OP_Mp_norm[4],self.OP_Mp_norm[5],self.OP_Mp_norm[6]],
                                [self.Vb,0,0,0,0,0,           0,0,0,0,0,0,0],
                                [0,self.Vdd,0,0,0,0,          0,0,0,0,0,0,0],
                                [0,0,self.GND,0,0,0,          0,0,0,0,0,0,0],
                                [0,0,0,self.OP_Rfb_norm,0,0,  0,0,0,0,0,0,0],
                                [0,0,0,0,self.OP_Cfb_norm,0,  0,0,0,0,0,0,0],
                                [0,0,0,0,0,self.OP_Cdec_norm, 0,0,0,0,0,0,0]])
        # clip the obs for better regularization
        observation = np.clip(observation, -5, 5)
        
        return observation
        
    def _get_info(self):
        ############################### TEMP #########################################
        # PSRR (10k) & phase margin at Iload = 10uA  PM = 45, PSRR = 30dB
        # later transfer learning PM = 60 , PSRR = 15dB
        # run another run without transfer learining to compare convergence rate
        ##############################################################################
        '''Evaluate the performance'''
        ''' DC performance '''
        idx = int(self.Vdd/0.01 - 0.5/0.01) # since I sweep Vdc from 0.5V - 2.5V to avoid some bad DC points 
        self.Vdrop =  abs(self.Vdd - self.low_load_current_dc_results[1][idx])
        

        # idx = [i for i,j in enumerate(self.low_load_current_dc_results[1]) if (j - self.Vref) >= 0]
        # if len(idx) > 1:
        #     idx = idx[0]
        # self.Vdrop =  abs(self.low_load_current_dc_results[0][idx] - self.low_load_current_dc_results[1][idx])
        
        # self.Vdrop_score = np.min([(self.Vdrop_target - self.Vdrop) / (self.Vdrop_target + self.Vdrop), 0])


        self.Vdrop_score = np.min([(self.Vdrop_target - self.Vdrop) / (self.Vdrop_target + self.Vdrop), 0])
    
        # AC Scores
        ''' PSRR performance with low load current'''
        freq = self.low_load_current_ac_results[0]
        self.low_load_current_psrr_results = self.low_load_current_ac_results[1]
        # @ 10 kHz
        idx_10kHz = int(10 * np.log10(self.PSRR_10kHz))
        # @ 1 MHz
        idx_1MHz = int(10 * np.log10(self.PSRR_1MHz))
        self.low_load_current_PSRR_worst_below_10kHz = max(self.low_load_current_psrr_results[:idx_10kHz]) # in linear scale
        self.low_load_current_PSRR_worst_below_1MHz = max(self.low_load_current_psrr_results[:idx_1MHz]) # in linear scale
        self.low_load_current_PSRR_worst_above_1MHz = max(self.low_load_current_psrr_results[idx_1MHz:]) # in linear scale
    
        if self.rew_eng == True:
            # @ 10 kHz
            if 20*np.log10(self.low_load_current_PSRR_worst_below_10kHz) > 0:
                self.low_load_current_PSRR_worst_below_10kHz_score = -10 
            else:
                self.low_load_current_PSRR_worst_below_10kHz_score = np.min([(self.PSRR_target_10kHz - self.low_load_current_PSRR_worst_below_10kHz) / (self.low_load_current_PSRR_worst_below_10kHz + self.PSRR_target_10kHz), 0])
                self.low_load_current_PSRR_worst_below_10kHz_score *= 0.5 # give a weights
    
            # @ 1MHz
            if 20*np.log10(self.low_load_current_PSRR_worst_below_1MHz) > 0:
                self.low_load_current_PSRR_worst_below_1MHz_score = -10
            else:
                self.low_load_current_PSRR_worst_below_1MHz_score =  np.min([(self.PSRR_target_1MHz - self.low_load_current_PSRR_worst_below_1MHz) / (self.low_load_current_PSRR_worst_below_1MHz + self.PSRR_target_1MHz), 0])
        
            # beyond 1 MHz
            if 20*np.log10(self.low_load_current_PSRR_worst_above_1MHz) > 0:
                self.low_load_current_PSRR_worst_above_1MHz_score = -10 
            else:
                self.low_load_current_PSRR_worst_above_1MHz_score =  np.min([(self.PSRR_target_above_1MHz - self.low_load_current_PSRR_worst_above_1MHz) / (self.low_load_current_PSRR_worst_above_1MHz + self.PSRR_target_above_1MHz), 0])
        else:
            # @ 10 kHz
            self.low_load_current_PSRR_worst_below_10kHz_score =  np.min([(self.PSRR_target_10kHz - self.low_load_current_PSRR_worst_below_10kHz) / (self.low_load_current_PSRR_worst_below_10kHz + self.PSRR_target_10kHz), 0])
            # @ 1MHz
            self.low_load_current_PSRR_worst_below_1MHz_score =  np.min([(self.PSRR_target_1MHz - self.low_load_current_PSRR_worst_below_1MHz) / (self.low_load_current_PSRR_worst_below_1MHz + self.PSRR_target_1MHz), 0])
            # beyond 1 MHz
            self.low_load_current_PSRR_worst_above_1MHz_score =  np.min([(self.PSRR_target_above_1MHz - self.low_load_current_PSRR_worst_above_1MHz) / (self.low_load_current_PSRR_worst_above_1MHz + self.PSRR_target_above_1MHz), 0])
        

        ''' PSRR performance with high load current'''
        freq = self.high_load_current_ac_results[0]
        self.high_load_current_psrr_results = self.high_load_current_ac_results[1]
        # @ 10 kHz
        idx_10kHz = int(10 * np.log10(self.PSRR_10kHz))
        # @ 1 MHz
        idx_1MHz = int(10 * np.log10(self.PSRR_1MHz))
        self.high_load_current_PSRR_worst_below_10kHz = max(self.high_load_current_psrr_results[:idx_10kHz]) # in linear scale
        self.high_load_current_PSRR_worst_below_1MHz = max(self.high_load_current_psrr_results[:idx_1MHz]) # in linear scale
        self.high_load_current_PSRR_worst_above_1MHz = max(self.high_load_current_psrr_results[idx_1MHz:]) # in linear scale
    
        if self.rew_eng == True:
            # @ 10 kHz
            if 20*np.log10(self.high_load_current_PSRR_worst_below_10kHz) > 0:
                self.high_load_current_PSRR_worst_below_10kHz_score = -10 
            else:
                self.high_load_current_PSRR_worst_below_10kHz_score = np.min([(self.PSRR_target_10kHz - self.high_load_current_PSRR_worst_below_10kHz) / (self.high_load_current_PSRR_worst_below_10kHz + self.PSRR_target_10kHz), 0])
                self.high_load_current_PSRR_worst_below_10kHz_score *= 0.5 # give a weights
    
            # @ 1MHz
            if 20*np.log10(self.high_load_current_PSRR_worst_below_1MHz) > 0:
                self.high_load_current_PSRR_worst_below_1MHz_score = -10
            else:
                self.high_load_current_PSRR_worst_below_1MHz_score =  np.min([(self.PSRR_target_1MHz - self.high_load_current_PSRR_worst_below_1MHz) / (self.high_load_current_PSRR_worst_below_1MHz + self.PSRR_target_1MHz), 0])
        
            # beyond 1 MHz
            if 20*np.log10(self.high_load_current_PSRR_worst_above_1MHz) > 0:
                self.high_load_current_PSRR_worst_above_1MHz_score = -10 
            else:
                self.high_load_current_PSRR_worst_above_1MHz_score =  np.min([(self.PSRR_target_above_1MHz - self.high_load_current_PSRR_worst_above_1MHz) / (self.high_load_current_PSRR_worst_above_1MHz + self.PSRR_target_above_1MHz), 0])
        else:
            # @ 10 kHz
            self.high_load_current_PSRR_worst_below_10kHz_score =  np.min([(self.PSRR_target_10kHz - self.high_load_current_PSRR_worst_below_10kHz) / (self.high_load_current_PSRR_worst_below_10kHz + self.PSRR_target_10kHz), 0])
            # @ 1MHz
            self.high_load_current_PSRR_worst_below_1MHz_score =  np.min([(self.PSRR_target_1MHz - self.high_load_current_PSRR_worst_below_1MHz) / (self.high_load_current_PSRR_worst_below_1MHz + self.PSRR_target_1MHz), 0])
            # beyond 1 MHz
            self.high_load_current_PSRR_worst_above_1MHz_score =  np.min([(self.PSRR_target_above_1MHz - self.high_load_current_PSRR_worst_above_1MHz) / (self.high_load_current_PSRR_worst_above_1MHz + self.PSRR_target_above_1MHz), 0])
        
        
        
        ''' Loop-gain phase margin with low load current'''
        # STB results
        low_load_current_stb_freq = self.low_load_current_stb_results[0]
        self.low_load_current_loop_mag_results = self.low_load_current_stb_results[1]
        self.low_load_current_loop_phase_results = self.low_load_current_stb_results[2]
        self.low_load_current_loop_gain_mag = 20*np.log10(self.low_load_current_loop_mag_results)
        self.low_load_current_loop_gain_phase = self.low_load_current_loop_phase_results # in rad
        if self.low_load_current_loop_gain_mag[0] < 0: # if DC gain is smaller than 0 dB
            self.low_load_current_phase_margin = 0 # phase margin becomes meaningless 
        else:  
            try:
                # find the index when mag cross the 0dB
                idx = [i for i,j in enumerate(self.low_load_current_loop_gain_mag[:-1] * self.low_load_current_loop_gain_mag[1:]) if j<0][0]+1
                low_load_current_phase_margin = np.min(self.low_load_current_loop_gain_phase[:idx])
            except: # this rarely happens: unity gain is larger than the frequency sweep
                idx = len(self.low_load_current_loop_gain_phase)
                low_load_current_phase_margin = np.min(self.low_load_current_loop_gain_phase[:idx])
            
            if low_load_current_phase_margin > 180 or low_load_current_phase_margin < 0:
                self.low_load_current_phase_margin = 0
            else:
                self.low_load_current_phase_margin = low_load_current_phase_margin

        if self.rew_eng == True:
            if self.low_load_current_phase_margin < 20:
                self.low_load_current_phase_margin_score = -10
            elif self.low_load_current_phase_margin < self.phase_margin_target:
                self.low_load_current_phase_margin_score = -1 + np.min([(self.low_load_current_phase_margin - self.phase_margin_target) / (self.low_load_current_phase_margin + self.phase_margin_target), 0]) 
            else:
                self.low_load_current_phase_margin_score = np.min([(self.low_load_current_phase_margin - self.phase_margin_target) / (self.low_load_current_phase_margin + self.phase_margin_target), 0]) # larger PM is better
        else:
            self.low_load_current_phase_margin_score = np.min([(self.low_load_current_phase_margin - self.phase_margin_target) / (self.low_load_current_phase_margin + self.phase_margin_target), 0]) # larger PM is better
        

        ''' Loop-gain phase margin with high load current'''
        # STB results
        high_load_current_stb_freq = self.high_load_current_stb_results[0]
        self.high_load_current_loop_mag_results = self.high_load_current_stb_results[1]
        self.high_load_current_loop_phase_results = self.high_load_current_stb_results[2]
        self.high_load_current_loop_gain_mag = 20*np.log10(self.high_load_current_loop_mag_results)
        self.high_load_current_loop_gain_phase = self.high_load_current_loop_phase_results # in rad
        if self.high_load_current_loop_gain_mag[0] < 0: # if DC gain is smaller than 0 dB
            self.high_load_current_phase_margin = 0 # phase margin becomes meaningless 
        else:  
            try:
                # find the index when mag cross the 0dB
                idx = [i for i,j in enumerate(self.high_load_current_loop_gain_mag[:-1] * self.high_load_current_loop_gain_mag[1:]) if j<0][0]+1
                high_load_current_phase_margin = np.min(self.high_load_current_loop_gain_phase[:idx])
            except: # this rarely happens: unity gain is larger than the frequency sweep
                idx = len(self.high_load_current_loop_gain_phase)
                high_load_current_phase_margin = np.min(self.high_load_current_loop_gain_phase[:idx])
            
            if high_load_current_phase_margin > 180 or high_load_current_phase_margin < 0:
                self.high_load_current_phase_margin = 0
            else:
                self.high_load_current_phase_margin = high_load_current_phase_margin

        if self.rew_eng == True:
            if self.high_load_current_phase_margin < 20:
                self.high_load_current_phase_margin_score = -10
            elif self.high_load_current_phase_margin < self.phase_margin_target:
                self.high_load_current_phase_margin_score = -1 + np.min([(self.high_load_current_phase_margin - self.phase_margin_target) / (self.high_load_current_phase_margin + self.phase_margin_target), 0]) 
            
            else:
                self.high_load_current_phase_margin_score = np.min([(self.high_load_current_phase_margin - self.phase_margin_target) / (self.high_load_current_phase_margin + self.phase_margin_target), 0]) # larger PM is better
        else:
            self.high_load_current_phase_margin_score = np.min([(self.high_load_current_phase_margin - self.phase_margin_target) / (self.high_load_current_phase_margin + self.phase_margin_target), 0]) # larger PM is better

        if self.transient:
            ''' Transient Vreg with high load current'''
            self.high_load_current_tran_time         = self.high_load_current_tran_results[0]
            self.high_load_current_tran_vreg_results = self.high_load_current_tran_results[1]

            idx = int(30/0.01 - 5/0.01) # since I simulated 20us for transient, take the final cycle.
            self.high_load_current_tran_vreg_result_high  = self.high_load_current_tran_vreg_results[idx]
            self.high_load_current_tran_vreg_result_low   = self.high_load_current_tran_vreg_results[-1]
            self.high_load_current_tran_vreg_score = np.min([(self.high_load_current_tran_vreg_result_low - self.high_load_current_tran_vreg_result_high) / (self.high_load_current_tran_vreg_result_low + self.high_load_current_tran_vreg_result_high), 0])

            ''' Transient Vreg with low load current'''
            self.low_load_current_tran_time          = self.low_load_current_tran_results[0]
            self.low_load_current_tran_vreg_results  = self.low_load_current_tran_results[1]

            self.low_load_current_tran_vreg_result_high  = self.low_load_current_tran_vreg_results[idx]
            self.low_load_current_tran_vreg_result_low   = self.low_load_current_tran_vreg_results[-1]
            self.low_load_current_tran_vreg_score = np.min([(self.low_load_current_tran_vreg_result_low - self.low_load_current_tran_vreg_result_high) / (self.low_load_current_tran_vreg_result_low + self.low_load_current_tran_vreg_result_high), 0])

        """ Quiescent current exclude load current """
        self.Iq = self.OP_M3['id'] + self.OP_M4['id']
        self.Iq = abs(self.Iq)
        self.Iq_score = np.min([(self.Iq_target - self.Iq) / (self.Iq_target + self.Iq), 0]) # smaller iq is better
        
        """ Decap score """
        self.Cdec_area_score = (self.Cdec_low - self.OP_Cdec) / (self.Cdec_low + self.OP_Cdec)
    
        """ Total reward """
        self.reward =   self.low_load_current_PSRR_worst_below_10kHz_score + \
                        self.low_load_current_PSRR_worst_below_1MHz_score + \
                        self.low_load_current_PSRR_worst_above_1MHz_score + \
                        self.low_load_current_phase_margin_score + \
                        self.high_load_current_PSRR_worst_below_10kHz_score + \
                        self.high_load_current_PSRR_worst_below_1MHz_score + \
                        self.high_load_current_PSRR_worst_above_1MHz_score + \
                        self.high_load_current_phase_margin_score + \
                        self.Iq_score + \
                        self.Vdrop_score
        # if transient is required                
        if self.transient:
            self.reward +=  self.high_load_current_tran_vreg_score + \
                            self.low_load_current_tran_vreg_score
        
        # if self.reward >= 0:
        #     self.reward = self.reward + self.Cdec_area_score + 10
                    
        return {
                'Drop-out voltage (mV)': self.Vdrop*1e3,
                
                'PSRR worst at low load current (dB) < 10kHz': 20*np.log10(self.low_load_current_PSRR_worst_below_10kHz),
                'PSRR worst at low load current (dB) < 1MHz' : 20*np.log10(self.low_load_current_PSRR_worst_below_1MHz),
                'PSRR worst at low load current (dB) > 1MHz' : 20*np.log10(self.low_load_current_PSRR_worst_above_1MHz),
                'PSRR worst at high load current (dB) < 10kHz': 20*np.log10(self.high_load_current_PSRR_worst_below_10kHz),
                'PSRR worst at high load current (dB) < 1MHz' : 20*np.log10(self.high_load_current_PSRR_worst_below_1MHz),
                'PSRR worst at high load current (dB) > 1MHz' : 20*np.log10(self.high_load_current_PSRR_worst_above_1MHz),
                
                'Loop-gain PM at low load current (deg)': self.low_load_current_phase_margin, 
                'Loop-gain PM at high load current (deg)': self.high_load_current_phase_margin, 

                'Iq (uA)': self.Iq*1e6, 
                'Cdec (pF)': self.OP_Cdec*1e12,

                "low_load_current_stb_results" : self.low_load_current_stb_results,
                "low_load_current_dc_results"  : self.low_load_current_dc_results,
                "low_load_current_ac_results"  : self.low_load_current_ac_results,
                "low_load_current_op_results"  : self.low_load_current_op_results,

                "high_load_current_stb_results": self.high_load_current_stb_results,
                "high_load_current_dc_results" : self.high_load_current_dc_results,
                "high_load_current_ac_results" : self.high_load_current_ac_results,
                "high_load_current_op_results" : self.high_load_current_op_results,

                "high_load_Vreg": self.high_load_current_tran_vreg_result_high - self.high_load_current_tran_vreg_result_low,
                "low_load_Vreg": self.low_load_current_tran_vreg_result_high - self.low_load_current_tran_vreg_result_low,

                "reward": self.reward,
            }