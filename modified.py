from pathlib import Path
from copy import deepcopy
import os
from psf_utils import PSF, Quantity
from inform import display, Error
from cmath import phase
from math import degrees
import matplotlib.pyplot as plt
import numpy as np


SCH_PATH = (
    Path(__file__)
    .parents[1]
    .joinpath("Cadence_lib/Simulation/Low_Dropout_With_Diff_Pair/spectre/schematic")
)

# parameter setup
PARA_DICT = {
    "M12": {"w": 13, "l": 2, "m": 1},
    "M34": {"w": 80, "l": 0.5, "m": 1},
    "M5": {"w": 14.27, "l": 1.22, "m": 1},
    "Mp": {"w": 63.65, "l": 0.5, "m": 142},
    "Rfb": {"w": 0.35, "l": 1, "m": 1},
    "Cfb": {"w": 10, "l": 10, "m": 4},
    "Cdec": {"w": 30, "l": 30, "m": 262},
    "Vb": {"v": 1.37},
    "Vref": {"v": 1.8},
}


# modified the parameter in input.scs
def modify_input(para_dict):
    p = deepcopy(para_dict)
    scs_path = SCH_PATH.joinpath("netlist/input.scs")
    lines = []
    try:
        scs_file = open(scs_path, "r")
        lines = scs_file.readlines()

        # modified  lines
        lines[58] = f"IL (Vreg 0) isource dc=10u type=dc\n"
        lines[59] = f"Vref_source (Vref 0) vsource dc={p['Vref']['v']} type=dc\n"
        lines[60] = f"Vb (net09 0) vsource dc={p['Vb']['v']} type=dc\n"
        lines[61] = f"Rfb (net6 net10) resistor r=1K l={p['Rfb']['l']}u w={p['Rfb']['w']}u m={p['Rfb']['m']}\n"
        lines[62] = f"Cdec (Vreg 0) capacitor c=1p l={p['Cdec']['l']}u w={p['Cdec']['w']}u m={p['Cdec']['m']}\n"
        lines[63] = f"Cfb (net10 Vreg) capacitor c=1p l={p['Cfb']['l']}u w={p['Cfb']['w']}u m={p['Cfb']['m']}\n"
        lines[64] = f"Vdd (VDD 0) vsource dc=Supply_Voltage mag=1 type=sine\n"
        
        # M2 NMOS Parameters
        lines[65] = f"M2 (net6 Vref net17 net17) nch_25 l={p['M12']['l']}u w={p['M12']['w']}u m={p['M12']['m']} nf=1 sd=310.0n \\\n"
        lines[66] = f"        ad=6.9e-13 as=6.9e-13 pd=6.46u ps=6.46u nrd=0.0516667 \\\n"
        lines[67] = f"        nrs=0.0516667 sa=230.0n sb=230.0n sca=0 scb=0 scc=0\n"
        
        # M5 NMOS Parameters
        lines[68] = f"M5 (net17 net09 0 0) nch_25 l={p['M5']['l']}u w={p['M5']['w']}u m={p['M5']['m']} nf=1 sd=310.0n ad=9.2e-13 \\\n"
        lines[69] = f"        as=9.2e-13 pd=8.46u ps=8.46u nrd=0.03875 nrs=0.03875 sa=230.0n \\\n"
        lines[70] = f"        sb=230.0n sca=0 scb=0 scc=0\n"
        
        # M1 NMOS Parameters
        lines[71] = f"M1 (net3 Vreg net17 net17) nch_25 l={p['M12']['l']}u w={p['M12']['w']}u m={p['M12']['m']} nf=1 sd=310.0n \\\n"
        lines[72] = f"        ad=6.9e-13 as=6.9e-13 pd=6.46u ps=6.46u nrd=0.0516667 \\\n"
        lines[73] = f"        nrs=0.0516667 sa=230.0n sb=230.0n sca=0 scb=0 scc=0\n"
        
        # M4 PMOS Parameters
        lines[74] = f"M4 (net6 net3 VDD VDD) pch_25 l={p['M34']['l']}u w={p['M34']['w']}u m={p['M34']['m']} nf=1 sd=310.0n ad=9.2e-12 \\\n"
        lines[75] = f"        as=9.2e-12 pd=80.46u ps=80.46u nrd=0.003875 nrs=0.003875 sa=230.0n \\\n"
        lines[76] = f"        sb=230.0n sca=0 scb=0 scc=0\n"

        # M3 PMOS Parameters
        lines[77] = f"M3 (net3 net3 VDD VDD) pch_25 l={p['M34']['l']}u w={p['M34']['w']}u m={p['M34']['m']} nf=1 sd=310.0n ad=9.2e-12 \\\n"
        lines[78] = f"        as=9.2e-12 pd=80.46u ps=80.46u nrd=0.003875 nrs=0.003875 sa=230.0n \\\n"
        lines[79] = f"        sb=230.0n sca=0 scb=0 scc=0\n"

        # Gate PMOS Parameters
        lines[80] = f"Mp (Vreg net6 VDD VDD) pch_25 l={p['Mp']['l']}u w={p['Mp']['w']}u m={p['Mp']['m']} nf=1 sd=310.0n \\\n"
        lines[81] = f"        ad=9.2e-12 as=9.2e-12 pd=80.46u ps=80.46u nrd=0.003875 \\\n"
        lines[82] = f"        nrs=0.003875 sa=230.0n sb=230.0n sca=0 scb=0 scc=0\n"

        lines[83] = f"simulatorOptions options reltol=1e-3 vabstol=1e-6 iabstol=1e-12 temp=27 \\\n"
        lines[84] = f"    tnom=27 scalem=1.0 scale=1.0 gmin=1e-12 rforce=1 maxnotes=5 maxwarns=5 \\\n"
        lines[85] = f'    digits=5 cols=80 pivrel=1e-3 sensfile="../psf/sens.output" \\\n'
        lines[86] = f"    checklimitdest=psf\n"

        # DC Operating Simulation Parameters
        lines[87] = f'dcOp dc write="spectre.dc" maxiters=150 maxsteps=10000 annotate=status\n'
        lines[88] = f"dcOpInfo info what=oppoint where=rawfile\n"

        # DC Simulation Parameters
        lines[89] = f"dc dc param=Supply_Voltage start=0.5 stop=2.5 lin=100 oppoint=rawfile \\\n"
        lines[90] = f"    maxiters=150 maxsteps=10000 annotate=status\n"

        # Transient Simulation Parameters
        lines[91] = f'\\\\tran tran stop=3u write="spectre.ic" writefinal="spectre.fc" \\\n'
        lines[92] = f"\\\\    annotate=status maxiters=5\n"
        lines[93] = f"\\\\finalTimeOP info what=oppoint where=rawfile\n"

        # AC Simulation Parameters
        lines[94] = f"ac ac start=10u stop=100G annotate=status\n"

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

            print("Simulation sucess. Check plot for result.")
        except:
            print("Cadence simulation failed.")

    except:
        print("ERROR")
    # test_file = open("test.txt", "w")
    # test_file.writelines(lines)
    # test_file.close()
    # except:
    # print("ERROR")


def parse_psf():
    # psf path
    psf_tran_path = SCH_PATH.joinpath("psf/tran.tran")
    psf_ac_path = SCH_PATH.joinpath("psf/ac.ac")
    psf_dc_path = SCH_PATH.joinpath("psf/dc.dc")
    psf_dcOp_path = SCH_PATH.joinpath("psf/dcOp.dc")

    # parse psf file
    psf_tran = PSF(psf_tran_path)
    psf_ac = PSF(psf_ac_path)
    psf_dc = PSF(psf_dc_path)
    psf_dcOp = PSF(psf_dcOp_path)

    # tran analysis
    sweep = psf_tran.get_sweep()
    Vreg = psf_tran.get_signal("Vreg")

    plt.plot(sweep.abscissa, Vreg.ordinate, linewidth=2, label=Vreg.name)
    plt.title("Vreg")
    plt.xlabel(f"{sweep.name} ({PSF.units_to_unicode(sweep.units)})")
    plt.ylabel(f"{Vreg.name} ({PSF.units_to_unicode(Vreg.units)})")
    plt.savefig("pictures/tran/Vreg.png", dpi = 500)
    plt.clf()

    # ac analysis
    sweep = psf_ac.get_sweep()
    Vreg = psf_ac.get_signal("Vreg")

    plt.plot(
        sweep.abscissa, 20 * np.log10(abs(Vreg.ordinate)), linewidth=2, label="PRSS"
    )
    plt.xscale("log")
    plt.title("PRSS")
    plt.xlabel(f"{sweep.name} ({PSF.units_to_unicode(sweep.units)})")
    plt.ylabel(f'PRSS ({PSF.units_to_unicode(Vreg.units)})')
    plt.savefig("pictures/ac/PRSS.png", dpi = 500)
    plt.clf()

    # ac phase gain
    Vreg_phase = [degrees(phase(i)) for i in Vreg.ordinate]
    plt.plot(
        sweep.abscissa, Vreg_phase, linewidth=2, label="Phase"
    )
    plt.xscale("log")
    plt.title("Phase")
    plt.xlabel(f"{sweep.name} ({PSF.units_to_unicode(sweep.units)})")
    plt.ylabel(f'Phase (degree)')
    plt.savefig("pictures/ac/Phase.png", dpi = 500)
    plt.clf()

    # dc Vreg analysis
    sweep = psf_dc.get_sweep()
    Vreg_dc = psf_dc.get_signal("Vreg")

    plt.plot(sweep.abscissa, Vreg_dc.ordinate, linewidth=2, label=Vreg_dc.name)
    plt.title("Vreg")
    plt.xlabel(f"{sweep.name} ({PSF.units_to_unicode(sweep.units)})")
    plt.ylabel(f"{Vreg_dc.name} ({PSF.units_to_unicode(Vreg_dc.units)})")
    plt.savefig("pictures/dc/Vreg.png", dpi = 500)
    plt.clf()

    # figure = plt.figure()
    # axes = figure.add_subplot(1,1,1)
    # axes.plot(sweep.abscissa, out.ordinate, linewidth=2, label=out.name)
    # axes.set_title('Vreg Output')
    # axes.set_xlabel(f'{sweep.name} ({PSF.units_to_unicode(sweep.units)})')
    # axes.set_ylabel(f'{out.name} ({PSF.units_to_unicode(out.units)})')
    # plt.show()

modify_input(PARA_DICT)
"""
psf_dcOp_path = SCH_PATH.joinpath("psf/dcOpInfo.info")
psf_dcOp = PSF(psf_dcOp_path)
with Quantity.prefs(map_sf=Quantity.map_sf_to_greek):
    for signal in sorted(psf_dcOp.all_signals(), key=lambda s: s.name):
        name = f'{signal.name}'
        print(f'{name} = {signal.ordinate}')
"""