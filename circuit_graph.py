import torch
import numpy as np

"""
Vdrop_target = 0.2  # drop-out voltage
        
PSRR_target_1kHz = 10**(-30/20) # in linear scale, equals -30dB
PSRR_target_10kHz = 10**(-30/20) # in linear scale, equals -30dB
PSRR_target_1MHz =  10**(-20/20) # in linear scale, equals -20dB
PSRR_target_above_1MHz =  10**(-5/20) # in linear scale, equals -5dB
PSRR_1kHz = 1e3 #  from DC to 1kHz
PSRR_10kHz = 1e4 #  from DC to 10kHz
PSRR_1MHz = 1e6  # from DC to 1 MHz

phase_margin_target = 45 # 45 degree PM minimum, this is for the loop-gain
Vreg = 1.2 # regulated output
Vref = 1.2
GND = 0
Vdd = Vref + Vdrop_target

Vload_reg_delta_target = Vreg * 0.02 # load regulartion variation is at most 2% of Vreg when it is switched from ILmin to ILmax
Iq_target = 200e-6 #200uA quiescent current maximum
Vline_reg_delta_target =  Vreg * 0.02  # line reg voltage to be around at most 2% of Vreg when it is at both ILmin and ILmax
"""

class GraphLDO:


    '''
    At: Il = 20mA
    PSRR and PM meet the previous spec


    Other algorithm
    

    '''
    """

        Here is a graph discription for the simple LDO:


                       Vdd 
             _____________________________________________
             |            |                              |
           M4_____________M3                             |
             |____|       |                              |
             |            |______________________________Mp
             |            |        |____Rfb__Cfb_________|
        Vreg-M1___________M2-Vref                        |_______________Vreg
                 |                                       |       |
            Vb---M5                                      |       |
                 |                                      Cdec     IL
                 |                                        |      |
                 |                                        |      |
                  ------------------------------------------------
                        GND

    node 0 will be M1
    node 1 will be M2
    node 2 will be M3
    node 3 will be M4
    node 4 will be M5
    node 5 will be Mp
    node 6 will be Vb
    node 7 will be Vdd
    node 8 will be GND
    node 9 will be Rfb
    node 10 will be Cfb
    node 11 will be Cdec

    """
    schematic_path = "Cadence_lib/Simulation/Low_Dropout_With_Diff_Pair/spectre/schematic"
    def __init__(self, Vdrop_target=0.2,
                PSRR_target_1kHz = 10**(-30/20), PSRR_target_10kHz=10**(-30/20), PSRR_target_1MHz=10**(-20/20),
                PSRR_target_above_1MHz =  10**(-5/20), PSRR_1kHz = 1e3, PSRR_10kHz = 1e4, PSRR_1MHz = 1e6,
                phase_margin_target = 45, Vreg = 1.2, Vref = 1.2, GND = 0, Vdd = 1.4,
                Vload_reg_delta_target = 1.2*0.02, Iq_target = 200e-6, Vline_reg_delta_target=1.2*0.02):

        # self.schematic_path = "Cadence_lib/Simulation/Low_Dropout_With_Diff_Pair/spectre/schematic"
        # self.device = torch.device(
        #     "cuda:0" if torch.cuda.is_available() else "cpu"
        # )
        
        self.device = torch.device(
           "cpu"
        )
# Rfb 1/gm Mp
# Cfb 0.22 Cdec
# Cdec 100p
        # we do not include R here since, it is not straght forward to get the resistance from resistor
        # in SKY130 PDK
        # self.ckt_hierarchy = (('M1','x1.x1.XM1','nfet_g5v0d10v5','m'),
        #               ('M2','x1.x1.XM2','nfet_g5v0d10v5','m'),
        #               ('M3','x1.x1.XM3','pfet_g5v0d10v5','m'),
        #               ('M4','x1.x1.XM4','pfet_g5v0d10v5','m'),
        #               ('M5','x1.x1.XM5','nfet_g5v0d10v5','m'),
        #               ('Mp','x1.XMp','pfet_g5v0d10v5','m'),
        #               ('Vb','','Vb','v'),
                      
        #               ('Cfb','x1.XCfb','cap_mim_m3_1','c'),
        #               ('Cdec','XCdec','cap_mim_m3_1','c')
        #              )    

        self.op = {'M1':{},
				   'M2':{},
				   'M3':{},
				   'M4':{},
				   'M5':{},
				   'Mp':{},
				   'Vb':{},
				   'Rfb':{},
				   'Cfb':{},
				   'Cdec':{}
				   }
        self.op_mean_std = ['M1','M2','M3','M4','M5','Mp']

        self.edge_index = torch.tensor([
            [0,1], [1,0], [0,2], [2,0], [0,3], [3,0], [0,4], [4,0], [0,5], [5,0], [0,10], [10,0], [0,11], [11,0],    
            [1,2], [2,1], [1,4], [4,1], [1,5], [5,1], [1,9], [9,1],
            [2,3], [3,2], [2,5], [5,2], [2,7], [7,2], [2,9], [9,2],
            [3,7], [7,3],
            [4,6], [6,4], [4,8], [8,4],
            [5,7], [7,5], [5,9], [9,5],
            [5,10], [10,5], [5,11], [11,5],
            [9,10], [10,9],
            [10,11], [11,10],
            [11,8], [8,11]
            ], dtype=torch.long).t().to(self.device)
        
        # # sorted based on if it is the small signal path
        # # small signal path: 0; biasing path: 1
        self.edge_type = torch.tensor([
            0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0,
            1, 1,
            1, 1, 1, 1, 
            1, 1, 0, 0, 0, 0, 0, 0, 
            0, 0,
            0, 0,
            1, 1,
            ]).to(self.device)
        
        self.num_relations = 2
        self.num_nodes = 12
        self.num_node_features = 13
        self.obs_shape = (self.num_nodes, self.num_node_features)
        
        """Select an action from the input state."""
        # M1 and M2 have the same L and W
        self.L_M1_low = 0.28    # um
        self.L_M1_high = 2      # um
        self.W_M1_low = 1       # um
        self.W_M1_high = 100    # um
        self.M_M1_low = 1 
        self.M_M1_high = 10 

        # M3 and M4 have the same L and W
        self.L_M3_low = 0.28    # um
        self.L_M3_high = 2      # um
        self.W_M3_low = 1       # um
        self.W_M3_high = 100    # um
        self.M_M3_low = 1 
        self.M_M3_high = 10 

        # M5 and the bias voltage
        self.L_M5_low = 0.28    # um
        self.L_M5_high = 2      # um
        self.W_M5_low = 1       # um
        self.W_M5_high = 100    # um
        self.M_M5_low = 1 
        self.M_M5_high = 10 
        self.Vb_low = 0.4       # V
        self.Vb_high = 1.2      # V

        # Mp
        self.L_Mp_low = 0.28    # um
        self.L_Mp_high = 2      # um
        self.W_Mp_low = 10      # um
        self.W_Mp_high = 200    # um
        self.M_Mp_low = 1   # 100
        self.M_Mp_high = 1000 # 2000

        # Il
        self.Il_low = '10u'
        self.Il_high = '10m'

        # Cdec
        self.L_Cdec = 30 # each unit cap is 30um by 30um
        self.W_Cdec = 30
        self.M_Cdec_low = 10 #10 calculated 55, up 56, low 54
        self.M_Cdec_high = 100  #300 # copies of unit cap
        self.Cdec_low = self.M_Cdec_low * (self.L_Cdec * self.W_Cdec * 2e-15 + (self.L_Cdec + self.W_Cdec)*0.38e-15)
        self.Cdec_high = self.M_Cdec_high * (self.L_Cdec * self.W_Cdec * 2e-15 + (self.L_Cdec + self.W_Cdec)*0.38e-15)

        # Rfb
        self.W_Rfb = 1.8 
        self.L_Rfb = 18.0 
        self.M_Rfb_low = 1 
        self.M_Rfb_high = 40 
        self.Rsheet = 605 
        self.Rfb_low =  self.Rsheet * self.L_Rfb / self.W_Rfb / self.M_Rfb_high  
        self.Rfb_high = self.Rsheet * self.L_Rfb / self.W_Rfb / self.M_Rfb_low 
        
        # Cfb
        self.W_Cfb = 10
        self.L_Cfb = 10
        self.M_Cfb_low = 1 #1 calculated 14, up 15, low 13
        self.M_Cfb_high = 50 #100
        self.Cfb_low = self.M_Cfb_low * (self.L_Cfb * self.W_Cfb * 2e-15 + (self.L_Cfb + self.W_Cfb) *0.38e-15)
        self.Cfb_high = self.M_Cfb_high * (self.L_Cfb * self.W_Cfb*2e-15 + (self.L_Cfb + self.W_Cfb) *0.38e-15)
 
        self.action_space_low = np.array([ self.W_M1_low, self.L_M1_low,        # M1_2
                                           self.W_M3_low, self.L_M3_low,        # M3_4
                                           self.W_M5_low, self.L_M5_low,        # M5
                                           self.W_Mp_low, self.L_Mp_low, self.M_Mp_low, # Mp
            	                           self.Vb_low,     # Vb
                                           self.M_Rfb_low,  # Rfb
                                           self.M_Cfb_low,  # Cfb
                                           self.M_Cdec_low,        # Cdec
                                         ]) 
        
        self.action_space_high = np.array([self.W_M1_high, self.L_M1_high,        # M1_2
                                           self.W_M3_high, self.L_M3_high,        # M3_4
                                           self.W_M5_high, self.L_M5_high,        # M5
                                           self.W_Mp_high, self.L_Mp_high, self.M_Mp_high, # Mp
                                           self.Vb_high,     # Vb
                                           self.M_Rfb_high,  # Rfb
                                           self.M_Cfb_high,  # Cfb
                                           self.M_Cdec_high,        # Cdec
                                         ]) 
        
        self.action_dim =len(self.action_space_low)
        self.action_shape = (self.action_dim,)    
        
        """Some target specifications for the final design"""
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

        """If you want to apply the reward engineering"""
        self.rew_eng = True