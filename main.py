import numpy
import os
from Evolve_dynamics import Plot_Trajectory
from Analyze_Sensitivity import Analyze_Sensitivity_number_operator, Analyze_Stability_Matrix_change_action, Analyze_Stability_Matrix_for_xp
from Analyze_Sensitivity_SCCL2 import SCCL2_Analyze_Sensitivity_number_operator, SCCL2_Analyze_Stability_Matrix_for_xp, Plot_Trajectory_SCCL2, Plot_Trajectory_SCCL2_Realistic_Hamiltonian, \
    Sample_SCCL2_Realistic_Hamiltonian_angular_velocity, Sample_SCCL2_scaling_angular_velocity , Analyze_OTOC_for_xp_for_Realistic_SCCL2_Hamiltonian

def main():
    # Plot_Trajectory()

    # Analyze_Sensitivity_number_operator()

    # Analyze_Stability_Matrix_change_action()

    # folder_path_SW_model = "/home/phyzch/PycharmProjects/OTOC_classical simulation/result/SW model/V=7/"
    # Analyze_Stability_Matrix_for_xp(folder_path_SW_model)

    # SCCL2_Analyze_Sensitivity_number_operator()

    # SCCL2_Analyze_Stability_Matrix_for_xp()

    folder_path_SCCL2_realistic_model = "/home/phyzch/PycharmProjects/OTOC_classical simulation/result/SCCL2_realistic_Hamiltonian/try/"
    # Analyze_OTOC_for_xp_for_Realistic_SCCL2_Hamiltonian(folder_path_SCCL2_realistic_model)

    # Plot_Trajectory_SCCL2()

    Plot_Trajectory_SCCL2_Realistic_Hamiltonian()

    # Sample_SCCL2_Realistic_Hamiltonian_angular_velocity()

    # Sample_SCCL2_scaling_angular_velocity()

main()