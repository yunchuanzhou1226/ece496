import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import numpy as np
import pickle
import sys
import pandas as pd


sys.path.append('..')


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

# M1 and M2 have the same L and W
L_M1_low = 0.28    # um
L_M1_high = 2      # um
W_M1_low = 1       # um
W_M1_high = 100    # um
M_M1_low = 1 
M_M1_high = 10 

# M3 and M4 have the same L and W
L_M3_low = 0.28    # um
L_M3_high = 2      # um
W_M3_low = 1       # um
W_M3_high = 100    # um
M_M3_low = 1 
M_M3_high = 10 

# M5 and the bias voltage
L_M5_low = 0.28    # um
L_M5_high = 2      # um
W_M5_low = 1       # um
W_M5_high = 100    # um
M_M5_low = 1 
M_M5_high = 10 
Vb_low = 0.4       # V
Vb_high = 1.2      # V

# Mp
L_Mp_low = 0.28    # um
L_Mp_high = 2      # um
W_Mp_low = 10      # um
W_Mp_high = 200    # um
M_Mp_low = 1   # 100
M_Mp_high = 1000 # 2000

# Cdec
L_Cdec = 30 # each unit cap is 30um by 30um
W_Cdec = 30
M_Cdec_low = 10 #10 calculated 55, up 56, low 54
M_Cdec_high = 100  #300 # copies of unit cap
Cdec_low = M_Cdec_low * (L_Cdec * W_Cdec * 2e-15 + (L_Cdec + W_Cdec)*0.38e-15)
Cdec_high = M_Cdec_high * (L_Cdec * W_Cdec * 2e-15 + (L_Cdec + W_Cdec)*0.38e-15)

 # Rfb
W_Rfb = 1.8 
L_Rfb = 18.0 
M_Rfb_low = 1 
M_Rfb_high = 40 
Rsheet = 605 
Rfb_low =  Rsheet * L_Rfb / W_Rfb / M_Rfb_high  
Rfb_high = Rsheet * L_Rfb / W_Rfb / M_Rfb_low 
        
# Cfb
W_Cfb = 10
L_Cfb = 10
M_Cfb_low = 1 #1 calculated 14, up 15, low 13
M_Cfb_high = 50 #100
Cfb_low = M_Cfb_low * (L_Cfb * W_Cfb * 2e-15 + (L_Cfb + W_Cfb) *0.38e-15)
Cfb_high = M_Cfb_high * (L_Cfb * W_Cfb*2e-15 + (L_Cfb + W_Cfb) *0.38e-15)

# action space
action_space_low = np.array([ W_M1_low, L_M1_low,        # M1_2
                                           W_M3_low, L_M3_low,        # M3_4
                                           W_M5_low, L_M5_low,        # M5
                                           W_Mp_low, L_Mp_low, M_Mp_low, # Mp
            	                           Vb_low,     # Vb
                                           M_Rfb_low,  # Rfb
                                           M_Cfb_low,  # Cfb
                                           M_Cdec_low,        # Cdec
                                         ]) 
        
action_space_high = np.array([W_M1_high, L_M1_high,        # M1_2
                                           W_M3_high, L_M3_high,        # M3_4
                                           W_M5_high, L_M5_high,        # M5
                                           W_Mp_high, L_Mp_high, M_Mp_high, # Mp
                                           Vb_high,     # Vb
                                           M_Rfb_high,  # Rfb
                                           M_Cfb_high,  # Cfb
                                           M_Cdec_high,        # Cdec
                                         ]) 


class ActionNormalizer():
    """Rescale and relocate the actions."""
    def __init__(self, action_space_low, action_space_high):
        
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space_low
        high = self.action_space_high 

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space_low
        high = self.action_space_high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action

actions = [ActionNormalizer(action_space_low,action_space_high).action(x) for x in memory.acts_buf[:num_steps]]
info = memory.info_buf[:num_steps]
reward_array = np.array([info[i]['reward'] for i in range(num_steps)])

df = pd.DataFrame()
df["M1 & M2 Width"]     = [x[0] for x in actions]
df["M1 & M2 Length"]    = [x[1] for x in actions]
df["M3 & M4 Width"]     = [x[2] for x in actions]
df["M3 & M4 Length"]    = [x[3] for x in actions]
df["M5 Width"]          = [x[4] for x in actions]
df["M5 Length"]         = [x[5] for x in actions]
df["Mp Width"]          = [x[6] for x in actions]
df["Mp Length"]         = [x[7] for x in actions]
df["Mp Multiplier"]     = [x[8] for x in actions]
df["Vb"]                = [x[9] for x in actions]
df["Rfb Multiplier"]    = [x[10] for x in actions]
df["Cfb Multiplier"]    = [x[11] for x in actions]
df["Cdec Multiplier"]   = [x[12] for x in actions]
df["Reward"]            = list(reward_array)


# print(df)


# fig = px.parallel_coordinates(df, color=df["Reward"],
#                               dimensions=["M1 & M2 Width", "M1 & M2 Length",
#                                           "M3 & M4 Width", "M3 & M4 Length", 
#                                           "M5 Width", "M5 Length", 
#                                           "Mp Width", "Mp Length", "Mp Multiplier",
#                                           "Vb", 
#                                           "Rfb Multiplier", 
#                                           "Cfb Multiplier",
#                                           "Cdec Multiplier"],
#                               color_continuous_scale=px.colors.diverging.Portland)
# fig.show()


fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df['Reward'],
                   colorscale = 'Portland',
                   showscale = True),
        dimensions=list([dict(range = [W_M1_low,W_M1_high],     label = 'M1 & M2 Width',  values = df['M1 & M2 Width']),
                         dict(range = [L_M1_low,L_M1_high],     label = 'M1 & M2 Length', values = df['M1 & M2 Length']),
                         dict(range = [W_M3_low,W_M3_high],     label = 'M3 & M4 Width',  values = df['M3 & M4 Width']),
                         dict(range = [L_M3_low,L_M3_high],     label = 'M3 & M4 Length', values = df['M3 & M4 Length']),
                         dict(range = [W_M5_low,W_M5_high],     label = "M5 Width",       values = df["M5 Width"]),
                         dict(range = [L_M5_low,L_M5_high],     label = "M5 Length",      values = df["M5 Length"]),
                         dict(range = [W_Mp_low,W_Mp_high],     label = "Mp Width",       values = df["Mp Width"]),
                         dict(range = [L_Mp_low,L_Mp_high],     label = "Mp Length",      values = df["Mp Length"]),
                         dict(range = [M_Mp_low,M_Mp_high],     label = "Mp Multiplier",  values = df["Mp Multiplier"]),
                         dict(range = [Vb_low,Vb_high],         label = "Vb",             values = df["Vb"]),
                         dict(range = [M_Rfb_low,M_Rfb_high],   label = "Rfb Multiplier", values = df["Rfb Multiplier"]),
                         dict(range = [M_Cfb_low,M_Cfb_high],   label = "Cfb Multiplier", values = df["Cfb Multiplier"]),
                         dict(range = [M_Cdec_low,M_Cdec_high], label = "Cdec Multiplier",values = df["Cdec Multiplier"])])
    )
)

fig.update_layout(
    plot_bgcolor = 'rgba(0,0,0,0)',
    paper_bgcolor = 'rgba(0,0,0,0)'
)
fig.update_yaxes(showline=True, linewidth=10, linecolor='black')

# fig.show()
pio.write_image(fig, "./pictures/parallel_Coord_plot.png", width=6*600, height=4*600, scale=1)