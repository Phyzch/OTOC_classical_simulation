import numpy
import os
from Evolve_dynamics import Plot_Trajectory
from Analyze_Sensitivity import Analyze_Sensitivity_number_operator, Analyze_Stability_Matrix_change_action, Analyze_Stability_Matrix_for_xp
from Analyze_Sensitivity_SCCL2 import SCCL2_Analyze_Sensitivity_number_operator, SCCL2_Analyze_Stability_Matrix_for_xp, Plot_Trajectory_SCCL2

def main():
    # Plot_Trajectory()

    # Analyze_Sensitivity_number_operator()

    # Analyze_Stability_Matrix_change_action()

    # folder_path_SW_model = "/home/phyzch/PycharmProjects/OTOC_classical simulation/result/SW model/V=7/"
    # Analyze_Stability_Matrix_for_xp(folder_path_SW_model)

    # SCCL2_Analyze_Sensitivity_number_operator()

    # SCCL2_Analyze_Stability_Matrix_for_xp()

    Plot_Trajectory_SCCL2()

main()