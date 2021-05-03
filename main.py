import numpy
import os
from Analyze_Sensitivity_SWW import Analyze_Sensitivity_number_operator, Analyze_Stability_Matrix_change_action, Analyze_Stability_Matrix_for_xp_SWW, Plot_Trajectory_SWW
from Analyze_Sensitivity_Other_Molecules import SCCL2_Analyze_Sensitivity_number_operator, Other_molecules_Analyze_Stability_Matrix_for_xp, Plot_Trajectory_Other_Molecules, Plot_Trajectory_SCCL2_Realistic_Hamiltonian, \
    Sample_SCCL2_Realistic_Hamiltonian_angular_velocity, Sample_SCCL2_scaling_angular_velocity , Analyze_OTOC_for_xp_for_Realistic_SCCL2_Hamiltonian

from Evolve_dynamics_Using_Bulirsch import Plot_Trajectory_Other_molecule_BS_method

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()


def main():

    # Function for SWW model.
    # Plot_Trajectory_SWW()

    # Analyze_Sensitivity_number_operator()

    # Analyze_Stability_Matrix_change_action()

    # Analyze xp sensitivity for SWW model
    folder_path_SW_model = "/home/phyzch/PycharmProjects/OTOC_classical simulation/result/SW model/V=7/"
    # Analyze_Stability_Matrix_for_xp_SWW(folder_path_SW_model)

    # Analyze number operator sensitivity for SCCL2
    # SCCL2_Analyze_Sensitivity_number_operator()

    # SCCL2 scaling Hamiltonian xp stability matrix
    folder_path_SCCL2_scaling_Other_Molecule = "/home/phyzch/PycharmProjects/OTOC_classical simulation/result/Other Molecule/Cyclopentaone/Average 2232222 BS_method (20)/"
    Other_molecules_Analyze_Stability_Matrix_for_xp(folder_path_SCCL2_scaling_Other_Molecule)


    # plot OTOC for  SCCL2 Strickler Gruebele Hamiltonian
    folder_path_SCCL2_realistic_model = "/home/phyzch/PycharmProjects/OTOC_classical simulation/result/SCCL2_realistic_Hamiltonian/try/"
    # Analyze_OTOC_for_xp_for_Realistic_SCCL2_Hamiltonian(folder_path_SCCL2_realistic_model)

    # plot trajectory for scaling Hamiltonian
    # Plot_Trajectory_Other_Molecules()
    # Plot_Trajectory_Other_molecule_BS_method()


    #plot trajectory for  Grueebele Strickler Hamiltonian
    # Plot_Trajectory_SCCL2_Realistic_Hamiltonian()

    # Sample_SCCL2_Realistic_Hamiltonian_angular_velocity()

    # Sample_SCCL2_scaling_angular_velocity()

main()