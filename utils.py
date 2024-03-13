from pathlib import Path
from copy import deepcopy
from psf_utils import PSF
import numpy as np
import json 
import os
import random
import statistics


# SCH_PATH = (
#     Path(__file__)
#     .parents[1]
#     .joinpath("Cadence_lib/Simulation/Low_Dropout_With_Diff_Pair/spectre/schematic")
# )

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.longdouble):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


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


# modified the parameter in input.scs
def save_param_to_cadence(para_dict, SCH_PATH):
    scs_path = SCH_PATH.joinpath("netlist/netlist")
    lines = []
    try:
        scs_file = open(scs_path, "r")
        lines = scs_file.readlines()

        # modified lines
        lines[4] = f"IL (Vreg 0) isource dc={para_dict['Il']['i']} type=dc\n"
        lines[5] = f"Vref_source (Vref 0) vsource dc={para_dict['Vref']['v']} type=dc\n"
        lines[6] = f"Vb (net09 0) vsource dc={para_dict['Vb']['v']} type=dc\n"
        lines[7] = f"Rfb (net06 net020 ) rnwsti l={para_dict['Rfb']['l']}u w={para_dict['Rfb']['w']}u mf={para_dict['Rfb']['m']}\n"
        lines[8] = f"\n"
        lines[9] = f"Cdec (Vreg 0) mimcap_sin lt={para_dict['Cdec']['l']}u wt={para_dict['Cdec']['w']}u mimflag=3 mf={para_dict['Cdec']['m']} mismatchflag=0\n"
        lines[10] = f"Cfb (net020 Vreg) mimcap_sin lt={para_dict['Cfb']['l']}u wt={para_dict['Cfb']['w']}u mimflag=3 mf={para_dict['Cfb']['m']} mismatchflag=0\n"
        lines[11] = f"IPRB0 (Vreg net016) iprobe\n"
        lines[12] = f"Vdd (VDD 0) vsource dc=Supply_Voltage mag=1 type=sine\n"
        lines[13] = f"M2 (net06 Vref net17 net17) nch_25 l={para_dict['M12']['l']}u w={para_dict['M12']['w']}u m={para_dict['M12']['m']} nf=1 sd=310.0n \\\n"
        lines[14] = f"        ad=2.99e-12 as=2.99e-12 pd=26.46u ps=26.46u nrd=0.0119231 \\\n"
        lines[15] = f"        nrs=0.0119231 sa=230.0n sb=230.0n sca=0 scb=0 scc=0\n"
        lines[16] = f"M5 (net17 net09 0 0) nch_25 l={para_dict['M5']['l']}u w={para_dict['M5']['w']}u m={para_dict['M5']['m']} nf=1 sd=310.0n \\\n"
        lines[17] = f"        ad=3.2821e-12 as=3.2821e-12 pd=29.0u ps=29.0u nrd=0.010862 \\\n"
        lines[18] = f"        nrs=0.010862 sa=230.0n sb=230.0n sca=0 scb=0 scc=0\n"
        lines[19] = f"M1 (net3 net016 net17 net17) nch_25 l={para_dict['M12']['l']}u w={para_dict['M12']['w']}u m={para_dict['M12']['m']} nf=1 sd=310.0n \\\n"
        lines[20] = f"        ad=2.99e-12 as=2.99e-12 pd=26.46u ps=26.46u nrd=0.0119231 \\\n"
        lines[21] = f"        nrs=0.0119231 sa=230.0n sb=230.0n sca=0 scb=0 scc=0\n"
        lines[22] = f"M4 (net06 net3 VDD VDD) pch_25 l={para_dict['M34']['l']}u w={para_dict['M34']['w']}u m={para_dict['M34']['m']} nf=1 sd=310.0n ad=1.84e-11 \\\n"
        lines[23] = f"        as=1.84e-11 pd=160.46u ps=160.46u nrd=0.0019375 nrs=0.0019375 \\\n"
        lines[24] = f"        sa=230.0n sb=230.0n sca=0 scb=0 scc=0\n"
        lines[25] = f"M3 (net3 net3 VDD VDD) pch_25 l={para_dict['M34']['l']}u w={para_dict['M34']['w']}u m={para_dict['M34']['m']} nf=1 sd=310.0n ad=1.84e-11 \\\n"
        lines[26] = f"        as=1.84e-11 pd=160.46u ps=160.46u nrd=0.0019375 nrs=0.0019375 \\\n"
        lines[27] = f"        sa=230.0n sb=230.0n sca=0 scb=0 scc=0\n"
        lines[28] = f"Mp (Vreg net06 VDD VDD) pch_25 l={para_dict['Mp']['l']}u w={para_dict['Mp']['w']}u m={para_dict['Mp']['m']} nf=1 sd=310.0n \\\n"
        lines[29] = f"        ad=1.46395e-11 as=1.46395e-11 pd=127.76u ps=127.76u nrd=0.00243519 \\\n"
        lines[30] = f"        nrs=0.00243519 sa=230.0n sb=230.0n sca=0 scb=0 scc=0\n"
    except:
        print("ERROR")



def run_sim_with_para(para_dict, SCH_PATH):
    scs_path = SCH_PATH.joinpath("netlist/input.scs")
    lines = []
    try:
        scs_file = open(scs_path, "r")
        lines = scs_file.readlines()

        # modified  lines
        lines[58] = f"IL (Vreg 0) isource dc={para_dict['Il']['i']} type=dc\n"
        lines[59] = f"Vref_source (Vref 0) vsource dc={para_dict['Vref']['v']} type=dc\n"
        lines[60] = f"Vb (net09 0) vsource dc={para_dict['Vb']['v']} type=dc\n"
        lines[61] = f"Rfb (net06 net020 ) rnwsti l={para_dict['Rfb']['l']}u w={para_dict['Rfb']['w']}u mf={para_dict['Rfb']['m']}\n"
        lines[62] = f"\n"
        lines[63] = f"Cdec (Vreg 0) mimcap_sin lt={para_dict['Cdec']['l']}u wt={para_dict['Cdec']['w']}u mimflag=3 mf={para_dict['Cdec']['m']} mismatchflag=0\n"
        lines[64] = f"Cfb (net10 Vreg) mimcap_sin lt={para_dict['Cfb']['l']}u wt={para_dict['Cfb']['w']}u mimflag=3 mf={para_dict['Cfb']['m']} mismatchflag=0\n"
        lines[65] = f"IPRB0 (Vreg net016) iprobe\n"
        lines[66] = f"Vdd (VDD 0) vsource dc=1.4 mag=1 type=pulse val0=1.4 val1=2.8 period=10u\n"
        
        
        # M2 NMOS Parameters
        lines[67] = f"M2 (net06 Vref net17 net17) nch_25 l={para_dict['M12']['l']}u w={para_dict['M12']['w']}u m={para_dict['M12']['m']} nf=1 sd=310.0n \\\n"
        lines[68] = f"        ad=2.99e-12 as=2.99e-12 pd=26.46u ps=26.46u nrd=0.0119231 \\\n"
        lines[69] = f"        nrs=0.0119231 sa=230.0n sb=230.0n sca=0 scb=0 scc=0\n"
        
        # M5 NMOS Parameters
        lines[70] = f"M5 (net17 net09 0 0) nch_25 l={para_dict['M5']['l']}u w={para_dict['M5']['w']}u m={para_dict['M5']['m']} nf=1 sd=310.0n ad=3.2821e-12 \\\n"
        lines[71] = f"        as=3.2821e-12 pd=29.0u ps=29.0u nrd=0.010862 nrs=0.010862 sa=230.0n \\\n"
        lines[72] = f"        sb=230.0n sca=0 scb=0 scc=0\n"
        
        # M1 NMOS Parameters
        lines[73] = f"M1 (net3 net016 net17 net17) nch_25 l={para_dict['M12']['l']}u w={para_dict['M12']['w']}u m={para_dict['M12']['m']} nf=1 sd=310.0n \\\n"
        lines[74] = f"        ad=2.99e-12 as=2.99e-12 pd=26.46u ps=26.46u nrd=0.0119231 \\\n"
        lines[75] = f"        nrs=0.0119231 sa=230.0n sb=230.0n sca=0 scb=0 scc=0\n"
        
        # M4 PMOS Parameters
        lines[76] = f"M4 (net06 net3 VDD VDD) pch_25 l={para_dict['M34']['l']}u w={para_dict['M34']['w']}u m={para_dict['M34']['m']} nf=1 sd=310.0n ad=1.84e-11 \\\n"
        lines[77] = f"        as=1.84e-11 pd=160.46u ps=160.46u nrd=0.0019375 nrs=0.0019375 sa=230.0n \\\n"
        lines[78] = f"        sb=230.0n sca=0 scb=0 scc=0\n"

        # M3 PMOS Parameters
        lines[79] = f"M3 (net3 net3 VDD VDD) pch_25 l={para_dict['M34']['l']}u w={para_dict['M34']['w']}u m={para_dict['M34']['m']} nf=1 sd=310.0n ad=1.84e-11 \\\n"
        lines[80] = f"        as=1.84e-11 pd=160.46u ps=160.46u nrd=0.0019375 nrs=0.0019375 sa=230.0n \\\n"
        lines[81] = f"        sb=230.0n sca=0 scb=0 scc=0\n"

        # Gate PMOS Parameters
        lines[82] = f"Mp (Vreg net06 VDD VDD) pch_25 l={para_dict['Mp']['l']}u w={para_dict['Mp']['w']}u m={para_dict['Mp']['m']} nf=1 sd=310.0n \\\n"
        lines[83] = f"        ad=1.46395e-11 as=1.46395e-11 pd=127.76u ps=127.76u nrd=0.00243519 \\\n"
        lines[84] = f"        nrs=0.00243519 sa=230.0n sb=230.0n sca=0 scb=0 scc=0\n"

        lines[85] = f"simulatorOptions options reltol=1e-3 vabstol=1e-6 iabstol=1e-12 temp=27 \\\n"
        lines[86] = f"    tnom=27 scalem=1.0 scale=1.0 gmin=1e-12 rforce=1 maxnotes=5 maxwarns=5 \\\n"
        lines[87] = f'    digits=5 cols=80 pivrel=1e-3 sensfile="../psf/sens.output" \\\n'
        lines[88] = f"    checklimitdest=psf\n"

        # DC Operating Simulation Parameters
        lines[89] = f'dcOp dc write="spectre.dc" maxiters=150 maxsteps=10000 annotate=status\n'
        lines[90] = f"dcOpInfo info what=oppoint where=rawfile\n"

        # DC Simulation Parameters
        lines[91] = f"dc dc param=Supply_Voltage start=0.5 stop=2.5 step=0.01 oppoint=rawfile \\\n"
        lines[92] = f"    maxiters=150 maxsteps=10000 annotate=status\n"

        # Transient Simulation Parameters
        lines[93] = f'tran tran stop=30u step=0.01u maxstep=0.01u minstep=0.01u \\\n'
        lines[94] = f'    write="spectre.ic" writefinal="spectre.fc" annotate=status maxiters=5 \n'
        lines[95] = f"finalTimeOP info what=oppoint where=rawfile\n"

        # AC Simulation Parameters
        lines[96] = f"ac ac start=1 stop=100G dec=10 oppoint=rawfile annotate=status \n"

        # STB Simulation Parameters
        lines[97] = f'stb stb start=1 stop=100G dec=10 probe=IPRB0 annotate=status \n'

        print("Parameters modified sucessfully, use new parameters to simulate...")
        scs_file.close()

        # write back to input.scs
        scs_file = open(scs_path, "w")
        scs_file.writelines(lines)
        scs_file.close()

        netlist_path = SCH_PATH.joinpath("netlist")
        psf_path = SCH_PATH.joinpath("psf")

        try:
            os.system(
                f"spectre -f psfascii =log {netlist_path}/input.log -r {psf_path} {netlist_path}/input.scs"
            )

            print("Simulation sucess.")
        except:
            print("Cadence simulation failed.")

    except:
        print("ERROR")

def run_sim_with_para_new(para_dict, SCH_PATH, component_to_modify):

    scs_path = SCH_PATH.joinpath("netlist/input.scs")
    new_lines = []
    scs_file = open(scs_path, "r")
    lines = scs_file.readlines()

    for line in lines:
        curr_line_first_word = line.split(" ", 1)[0]
        if (curr_line_first_word in component_to_modify):
            if (curr_line_first_word == 'M1' or curr_line_first_word == 'M2'):
                dict_term = 'M12'
            elif (curr_line_first_word == 'M3' or curr_line_first_word == 'M4'):
                dict_term = 'M34'
            else:
                dict_term = curr_line_first_word

            curr_line_split = line.split()
            curr_line_split = curr_line_split[1:]

            for word in curr_line_split:
                split_by_equal = word.split("=")
                if (split_by_equal[0] in para_dict[dict_term]):
                    curr_line_first_word += f" {split_by_equal[0]}={para_dict[dict_term][split_by_equal[0]]}"
                else:
                    curr_line_first_word += (" " + word)
            curr_line_first_word += "\n"
            new_lines.append(curr_line_first_word)
        else:
            new_lines.append(line)
    scs_file.close()

    scs_path = SCH_PATH.joinpath("netlist/test.scs")
    # write back to input.scs
    scs_file = open(scs_path, "w")
    scs_file.writelines(new_lines)
    scs_file.close()

    netlist_path = SCH_PATH.joinpath("netlist")
    psf_path = SCH_PATH.joinpath("psf")


# from circuit_graph imort GraphLDO
# CktGraph = GraphLDO
# SCH_PATH = (
#     Path(__file__)
#     .parents[1]
#     .joinpath(CktGraph.schematic_path)
# )
# para_dict = dict()
# para_dict["M12"] = dict()
# para_dict["M34"] = dict()
# para_dict["M5"] = dict()
# para_dict["Mp"] = dict()
# para_dict["IL"] = dict()
# para_dict['Vref_source'] = dict()
# para_dict["Rfb"] = dict()
# para_dict["Cfb"] = dict()
# para_dict["Cdec"] = dict()
# para_dict["Vb"] = dict()

# para_dict["M12"]["l"] = 1
# para_dict["M12"]["w"] = 1
# para_dict["M12"]["m"] = 1
# para_dict["M34"]["l"] = 1
# para_dict["M34"]["w"] = 1
# para_dict["M34"]["m"] = 1
# para_dict["M5"]["l"] = 1
# para_dict["M5"]["w"] = 1
# para_dict["M5"]["m"] = 1
# para_dict["Mp"]["l"] = 1
# para_dict["Mp"]["w"] = 1
# para_dict["Mp"]["m"] = 1

# para_dict["Rfb"]["l"] = 1
# para_dict["Rfb"]["w"] = 1
# para_dict["Rfb"]["mf"] = 1
# para_dict["Cfb"]["lt"] = 1
# para_dict["Cfb"]["wt"] = 1
# para_dict["Cfb"]["mf"] = 1
# para_dict["Cdec"]["lt"] = 1
# para_dict["Cdec"]["wt"] = 1
# para_dict["Cdec"]["mf"] = 1

# para_dict['IL']['dc'] = 1
# para_dict['Vb']['dc'] = 1
# para_dict['Vref_source']['dc'] = 1


# run_sim_with_para_new(para_dict, SCH_PATH, ["M1", "M2", "M5", "Mp", "Rfb", "Cfb", "Cdec", "IL", "Vb", 'Vref_source'])



# parse all outputs
class OutputParser:
    def __init__(self, CktGraph):
        self.op = CktGraph.op
        self.SCH_PATH = (
            Path(__file__)
            .parents[1]
            .joinpath(CktGraph.schematic_path)
        )   
        self.psf_tran_path = self.SCH_PATH.joinpath("psf/tran.tran")
        self.psf_ac_path   = self.SCH_PATH.joinpath("psf/ac.ac")
        self.psf_dc_path   = self.SCH_PATH.joinpath("psf/dc.dc")
        self.psf_dcOp_path = self.SCH_PATH.joinpath("psf/dcOpInfo.info")
        self.psf_stb_path  = self.SCH_PATH.joinpath("psf/stb.stb")
    
    # STB analysis result parser

    def stb(self):
        try:
            stb_file = open(self.psf_stb_path, "r")
            lines_stb = stb_file.readlines()
            current_section = "HEADER"
            stb_freq = []
            loopGain_mag = []
            loopGain_ph = []
            for line in lines_stb:
                line = line[:-1]
                if line == "TYPE":
                    current_section = "TYPE"
                    continue
                elif line == "SWEEP":
                    current_section = "SWEEP"
                    continue
                elif line == "TRACE":
                    current_section = "TRACE"
                    continue
                elif line == "VALUE":
                    current_section = "VALUE"
                    continue
                elif line == "END":
                    break
                
                if current_section == 'VALUE':
                    if 'freq' in line:
                        stb_freq.append(float(line.split()[1]))
                    elif 'loopGain' in line:
                        loopGain = line.split(' ')
                        loopGain = complex(float(loopGain[1][1:]), float(loopGain[2][:-1]))
                        loopGain_mag.append(np.absolute(loopGain))
                        loopGain_ph.append(np.angle(loopGain, deg=True))

            return stb_freq, loopGain_mag, loopGain_ph
        except:
            print("ERROR: no .STB simulation results.")
    
    # AC analysis result parser
    def ac(self):
        try:
            ac_file = open(self.psf_ac_path, "r")
            lines_ac = ac_file.readlines()
            current_section = "HEADER"
            freq = []
            Vout_mag = []
            Vout_ph = []
            for line in lines_ac:
                line = line[:-1]
                if line == "TYPE":
                    current_section = "TYPE"
                    continue
                elif line == "SWEEP":
                    current_section = "SWEEP"
                    continue
                elif line == "TRACE":
                    current_section = "TRACE"
                    continue
                elif line == "VALUE":
                    current_section = "VALUE"
                    continue
                elif line == "END":
                    break
                
                if current_section == 'VALUE':
                    if 'freq' in line:
                        freq.append(float(line.split()[1]))
                    elif 'Vreg' in line:
                        Vac = line.split(' ')
                        Vac = complex(float(Vac[1][1:]), float(Vac[2][:-1]))
                        Vout_mag.append(np.absolute(Vac))
                        Vout_ph.append(np.angle(Vac))

            return freq, Vout_mag, Vout_ph
        except:
            print("ERROR: no .AC simulation results.")
            
    # DC analysis result parser
    def dc(self):
        try:
            dc_file = open(self.psf_dc_path, "r")
            lines_dc = dc_file.readlines()
            current_section = "HEADER"
            Vdd_dc = []
            Vreg_dc = []
            for line in lines_dc:
                line = line[:-1]
                if line == "TYPE":
                    current_section = "TYPE"
                    continue
                elif line == "SWEEP":
                    current_section = "SWEEP"
                    continue
                elif line == "TRACE":
                    current_section = "TRACE"
                    continue
                elif line == "VALUE":
                    current_section = "VALUE"
                    continue
                elif line == "END":
                    break

                if current_section == "VALUE":
                    if 'VDD' in line:
                        Vdd_dc.append(float(line.split(' ')[1]))
                    elif 'Vreg' in line:
                        Vreg_dc.append(float(line.split(' ')[1]))
            
            return Vdd_dc, Vreg_dc
        except:
            print("ERROR: no .DC simulation results.")
    
    # Transient analysis result parser
    def tran(self):
        try:
            tran_file = open(self.psf_tran_path, "r")
            lines_tran = tran_file.readlines()
            current_section = "HEADER"
            time = []
            Vreg_tran = []
            for line in lines_tran:
                line = line[:-1]
                if line == "TYPE":
                    current_section = "TYPE"
                    continue
                elif line == "SWEEP":
                    current_section = "SWEEP"
                    continue
                elif line == "TRACE":
                    current_section = "TRACE"
                    continue
                elif line == "VALUE":
                    current_section = "VALUE"
                    continue
                elif line == "END":
                    break

                if current_section == "VALUE":
                    if 'time' in line:
                        time.append(float(line.split(' ')[1]))
                    elif 'Vreg' in line:
                        Vreg_tran.append(float(line.split(' ')[1]))            
            return time, Vreg_tran
        except:
                print("ERROR: no .TRAN simulation results.")

    # DCOP analysis result parser
    def dcOp(self):
        # data dict
        self.DCOP_TYPE_DICT = {}
        '''
        {
            Component type1: {
                measurement1:{
                    idex in the measurement-value list in VALUE_DICT,
                    Unit,
                    Description,
                }
                measurement2:{
                    idex in the measurement-value list in VALUE_DICT,
                    Unit,
                    Description,
                },
                ...
            },

            Component type2: {
                measurement1:{
                    idex in the measurement-value list in VALUE_DICT,
                    Unit,
                    Description,
                },
                measurement2:{
                    idex in the measurement-value list in VALUE_DICT,
                    Unit,
                    Description,
                },
                ...
            },
            ...
        }
        '''

        self.DCOP_VALUE_DICT = {}
        '''
        {
            Component Name1:{
                Type of the Component,
                Measurement-value list
            },
            Component Name2:{
                Type of the Component,
                Measurement-value list
            },
            ...
        }
        '''
        try:
            dcop_file = open(self.psf_dcOp_path, "r")
            lines_dcop = dcop_file.readlines()
            current_section = "HEADER"
            current_type = ""
            current_measurement = ""
            current_component = ""
            measurement_idx = 0
            for line in lines_dcop:
                line = line[:-1]
                if line == "TYPE":
                    current_section = "TYPE"
                    continue
                elif line == "VALUE":
                    current_section = "VALUE"
                    continue
                elif line == "END":
                    break
                
                # store data structure for each type of components
                if current_section == "TYPE":
                    if "STRUCT" in line:
                        current_type = line.split()[0][1:-1]
                        self.DCOP_TYPE_DICT[current_type] = {}
                        measurement_idx = 0
                    elif (")" not in line) and ("PROP" in line):
                        current_measurement = line.split()[0][1:-1]
                        self.DCOP_TYPE_DICT[current_type][current_measurement] = {}
                        self.DCOP_TYPE_DICT[current_type][current_measurement][
                            "idx"
                        ] = measurement_idx
                        measurement_idx += 1
                    elif (")" not in line) and ("key" not in line):
                        self.DCOP_TYPE_DICT[current_type][current_measurement][
                            line.split()[0][1:-1]
                        ] = line.split()[1][1:-1]
                # store data for each components
                elif current_section == "VALUE":
                    line_split = line.split()
                    if (
                        len(line_split) > 1
                        and line_split[1][1:-1] in self.DCOP_TYPE_DICT
                        and (")" not in line)
                        and ("model" not in line)
                    ):
                        current_component = line.split()[0][1:-1]
                        current_component = current_component.split('.')[0]
                        current_component_type = line.split()[1][1:-1]
                        self.DCOP_VALUE_DICT[current_component] = {}
                        self.DCOP_VALUE_DICT[current_component]["type"] = current_component_type
                        self.DCOP_VALUE_DICT[current_component]["values"] = []
                    elif (")" not in line) and ("model" not in line):
                        self.DCOP_VALUE_DICT[current_component]["values"].append(float(line))

            for component_name in self.op:
                for measurement_name in self.DCOP_TYPE_DICT[self.DCOP_VALUE_DICT[component_name]["type"]]:
                    self.op[component_name][measurement_name] = self.DCOP_VALUE_DICT[component_name]["values"][self.DCOP_TYPE_DICT[self.DCOP_VALUE_DICT[component_name]["type"]][measurement_name]["idx"]]

            return self.op

        except:
            print("ERROR: no .DCOP simulation results.")


# # parse the data from dcOpInfo.info
# def parse_dcOp(requested_values, SCH_PATH):
#     # data dict
#     TYPE_DICT = {}
#     '''
#     {
#         Component type1: {
#             measurement1:{
#                 idex in the measurement-value list in VALUE_DICT,
#                 Unit,
#                 Description,
#             }
#             measurement2:{
#                 idex in the measurement-value list in VALUE_DICT,
#                 Unit,
#                 Description,
#             },
#             ...
#         },

#         Component type2: {
#             measurement1:{
#                 idex in the measurement-value list in VALUE_DICT,
#                 Unit,
#                 Description,
#             },
#             measurement2:{
#                 idex in the measurement-value list in VALUE_DICT,
#                 Unit,
#                 Description,
#             },
#             ...
#         },
#         ...
#     }
#     '''

#     VALUE_DICT = {}
#     '''
#     {
#         Component Name1:{
#             Type of the Component,
#             Measurement-value list
#         },
#         Component Name2:{
#             Type of the Component,
#             Measurement-value list
#         },
#         ...
#     }
#     '''


#     dcop_path = SCH_PATH.joinpath("psf/dcOpInfo.info")
#     lines = []
#     try:
#         scs_file = open(dcop_path, "r")
#         lines = scs_file.readlines()
#         current_section = "HEADER"
#         current_type = ""
#         current_measurement = ""
#         current_component = ""
#         measurement_idx = 0
#         for line in lines:
#             line = line[:-1]
#             if line == "TYPE":
#                 current_section = "TYPE"
#                 continue
#             elif line == "VALUE":
#                 current_section = "VALUE"
#                 continue
#             elif line == "END":
#                 break
            
#             # store data structure for each type of components
#             if current_section == "TYPE":
#                 if "STRUCT" in line:
#                     current_type = line.split()[0][1:-1]
#                     TYPE_DICT[current_type] = {}
#                     measurement_idx = 0
#                 elif (")" not in line) and ("PROP" in line):
#                     current_measurement = line.split()[0][1:-1]
#                     TYPE_DICT[current_type][current_measurement] = {}
#                     TYPE_DICT[current_type][current_measurement][
#                         "idx"
#                     ] = measurement_idx
#                     measurement_idx += 1
#                 elif (")" not in line) and ("key" not in line):
#                     TYPE_DICT[current_type][current_measurement][
#                         line.split()[0][1:-1]
#                     ] = line.split()[1][1:-1]
#             # store data for each components
#             elif current_section == "VALUE":
#                 line_split = line.split()
#                 if (
#                     len(line_split) > 1
#                     and line_split[1][1:-1] in TYPE_DICT
#                     and (")" not in line)
#                     and ("model" not in line)
#                 ):
#                     current_component = line.split()[0][1:-1]
#                     current_component_type = line.split()[1][1:-1]
#                     VALUE_DICT[current_component] = {}
#                     VALUE_DICT[current_component]["type"] = current_component_type
#                     VALUE_DICT[current_component]["values"] = []
#                 elif (")" not in line) and ("model" not in line):
#                     VALUE_DICT[current_component]["values"].append(np.float128(line))

#         for component_name in requested_values:
#             for measurement_name in requested_values[component_name]:
#                 requested_values[component_name][measurement_name] = VALUE_DICT[component_name]["values"][TYPE_DICT[VALUE_DICT[component_name]["type"]][measurement_name]["idx"]]

#         return requested_values

#     except:
#         print("ERROR")


def generate_transistor_internal_parameter_mean_std(ckt_graph, num_rnd_sample = 1000):
    SCH_PATH = (
                Path(__file__)
                .parents[1]
                .joinpath(ckt_graph.schematic_path)
    )
    sim_result = OutputParser(ckt_graph)
    mean_std = dict(OP_M_mean = dict(), OP_M_std = dict())

    for idx in range(num_rnd_sample):
        para_dict = dict(M12 = dict(w = round(random.uniform(ckt_graph.W_M1_low,   ckt_graph.W_M1_high),   3),
                                    l = round(random.uniform(ckt_graph.L_M1_low,   ckt_graph.L_M1_high),   3),
                                    m =       random.randint(ckt_graph.M_M1_low,   ckt_graph.M_M1_high)),
                         M34 = dict(w = round(random.uniform(ckt_graph.W_M3_low,   ckt_graph.W_M3_high),   3),
                                    l = round(random.uniform(ckt_graph.L_M3_low,   ckt_graph.L_M3_high),   3),
                                    m =       random.randint(ckt_graph.M_M3_low,   ckt_graph.M_M3_high)),
                          M5 = dict(w = round(random.uniform(ckt_graph.W_M5_low,   ckt_graph.W_M5_high),   3),
                                    l = round(random.uniform(ckt_graph.L_M5_low,   ckt_graph.L_M5_high),   3),
                                    m =       random.randint(ckt_graph.M_M5_low,   ckt_graph.M_M5_high)),
                          Mp = dict(w = round(random.uniform(ckt_graph.W_Mp_low,   ckt_graph.W_Mp_high),   3),
                                    l = round(random.uniform(ckt_graph.L_Mp_low,   ckt_graph.L_Mp_high),   3),
                                    m =       random.randint(ckt_graph.M_Mp_low,   ckt_graph.M_Mp_high)),
                         Rfb = dict(w =                      ckt_graph.W_Rfb,
                                    l =                      ckt_graph.L_Rfb,
                                    m =       random.randint(ckt_graph.M_Rfb_low,  ckt_graph.M_Rfb_high)),
                         Cfb = dict(w =                      ckt_graph.W_Cfb,
                                    l =                      ckt_graph.L_Cfb,
                                    m =       random.randint(ckt_graph.M_Cfb_low,  ckt_graph.M_Cfb_high)),
                        Cdec = dict(w =                      ckt_graph.W_Cdec,
                                    l =                      ckt_graph.L_Cdec,
                                    m =       random.randint(ckt_graph.M_Cdec_low, ckt_graph.M_Cdec_high)),
                          Vb = dict(v = round(random.uniform(ckt_graph.Vb_low,     ckt_graph.Vb_high),     3)),
                        Vref = dict(v =                      ckt_graph.Vref),
                          Il = dict(i = ckt_graph.Il_low if random.randint(0,1) else ckt_graph.Il_high)
                        )

        run_sim_with_para(para_dict, SCH_PATH)
        OP_results = sim_result.dcOp()

        # initialize mean_std dict
        if idx == 0:
            for measurement_name in OP_results[ckt_graph.op_mean_std[0]]:
                mean_std['OP_M_mean'][measurement_name] = []
                mean_std['OP_M_std'][measurement_name] = []

        # record the values
        for component_name in OP_results:
            if component_name not in ckt_graph.op_mean_std:
                continue
            for measurement_name in OP_results[component_name]:
                mean_std['OP_M_mean'][measurement_name].append(OP_results[component_name][measurement_name])
                mean_std['OP_M_std'][measurement_name].append(OP_results[component_name][measurement_name])

    # calculate mean and std
    for measurement_name in OP_results[ckt_graph.op_mean_std[0]]:
        mean_std['OP_M_mean'][measurement_name] = statistics.mean(mean_std['OP_M_mean'][measurement_name])
        mean_std['OP_M_std'][measurement_name] = statistics.stdev(mean_std['OP_M_std'][measurement_name])

    mean_std_path = SCH_PATH.joinpath("transistor_internal_parameter_mean_std.json")
    with open(mean_std_path, "w") as outfile: 
        json.dump(mean_std, outfile, cls=NpEncoder)





'''
RUN the following command in virtuoso 
loadi("/groups/czzzgrp/anaconda3/envs/my-env/lib/python3.11/site-packages/skillbridge/server/python_server.il")
pyStartServer
'''


from skillbridge import Workspace


def edit_schematic(inst_name, properties, 
                   lib_name = "Capstone_Project", 
                   lib_cell_name = "Low_Dropout_With_Diff_Pair", 
                   lib_cell_view_name = "schematic"):
    # open the schematic
    ws = Workspace.open()
    cell_view = ws.db.open_cell_view_by_type(lib_name, lib_cell_name, lib_cell_view_name, "", "a")
    insts = cell_view.instances

    # change the instance of the given name
    for inst in insts:
        if inst.name == inst_name:
            info = "INFO: In lib: {} cell: {} view: {} modified {} ".format(lib_name, lib_cell_name, lib_cell_view_name, inst_name)
            
            # change every property listed in the dictionary
            for prop in properties:
                if properties[prop] != '' and inst[prop] != None:
                    inst[prop] = properties[prop]
                    info += "{}: {} ".format(prop, inst[prop])
            print(info)
            break
    
    # check and save the schematic
    ws.db.check(cell_view)
    ws.db.save(cell_view)


def save_action_to_schematic(CktGraph, action):
    # save the action back into the schematic
    action = ActionNormalizer(action_space_low  = CktGraph.action_space_low, 
                              action_space_high = CktGraph.action_space_high).action(action) # convert [-1.1] range back to normal range
    action = action.astype(object)
    W_M1, L_M1, \
    W_M3, L_M3, \
    W_M5, L_M5,  \
    W_Mp, L_Mp, M_Mp, \
    Vb, \
    M_Rfb, M_Cfb, \
    M_Cdec = action
    
    edit_schematic('M1', dict(l = "{}u".format(round(L_M1, 4)), 
                              w = "{}u".format(round(W_M1, 4)),
                              wf = "{}u".format(round(W_M1, 4))))
    edit_schematic('M2', dict(l = "{}u".format(round(L_M1, 4)), 
                              w = "{}u".format(round(W_M1, 4)),
                              wf = "{}u".format(round(W_M1, 4))))
    edit_schematic('M3', dict(l = "{}u".format(round(L_M3, 4)), 
                              w = "{}u".format(round(W_M3, 4)),
                              wf = "{}u".format(round(W_M3, 4))))
    edit_schematic('M4', dict(l = "{}u".format(round(L_M3, 4)), 
                              w = "{}u".format(round(W_M3, 4)),
                              wf = "{}u".format(round(W_M3, 4))))
    edit_schematic('M5', dict(l = "{}u".format(round(L_M5, 4)), 
                              w = "{}u".format(round(W_M5, 4)),
                              wf = "{}u".format(round(W_M5, 4))))
    edit_schematic('Mp', dict(l = "{}u".format(round(L_Mp, 4)), 
                              w = "{}u".format(round(W_Mp, 4)),
                              simM = int(M_Mp),
                              totalM = int(M_Mp),
                              wf = "{}u".format(round(W_Mp, 4))))
    edit_schematic('Vb', dict(vdc = Vb))
    edit_schematic('Rfb', dict(m = int(M_Rfb)))
    edit_schematic('Cfb', dict(m = int(M_Cfb)))
    edit_schematic('Cdec', dict(m = int(M_Cdec)))

# from circuit_graph import GraphLDO
# CktGraph = GraphLDO
# save_action_to_schematic(CktGraph, [4.707513183355331, 1.595399378538132, 88.81715071201324, 0.2823687756061556,
#  40.0684275329113, 1.2994877582788469, 66.32999673485756, 0.34650112867355365,
#  38.99372512102127, 1.075568437576294, 35.627860367298126, 33.65817591547966,
#  97.54893034696579])