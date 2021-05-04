import numpy as np
from SCCL2_potential import Generate_n_quanta_list_for_SCCL2
from Bulirsch_Stoer_module import bulStoer
from Evolve_dynamics import Other_Molecules
import matplotlib.pyplot as plt

def Evolve_dynamics_Other_Molecule_BS_method(Initial_position, Time_step, V0, scaling_parameter, frequency, f0, nquanta_list, nquanta_list_trans):
    t0 = 0
    time_step_size = Time_step[1]

    final_time = Time_step[-1]

    Initial_position = np.array(Initial_position)

    tol = 1e-6

    time, position , finish_simulation = bulStoer(Other_Molecules, t0, Initial_position, final_time, time_step_size, args = (V0, scaling_parameter, frequency, f0, nquanta_list, nquanta_list_trans) , tol = tol )



    return time, position , finish_simulation

def Plot_Trajectory_Other_molecule_BS_method():
    # This parameter tune the chaos
    V0 = 3050

    scaling_parameter = 0.2

    # Cyclopentaone
    frequency = [3062, 2914, 1445, 1349, 1275, 1203, 1043, 905, 695]

    dof = 9

    f0 = 500

    nquanta_list = Generate_n_quanta_list_for_SCCL2(dof)

    final_time = 0.021

    Time_step = np.linspace(0, final_time, 100)

    # specify initial position and angle
    Initial_action = [0, 0, 0, 1, 0, 0, 1, 0, 0]
    Initial_angle = [2.4442179158763073, 1.0017940119372089, 4.250017300134825, 2.5048770218889747, 0.5626141486498147,
                     3.158573153008315, 0.7225434637992191, 6.038088800623476, 4.530226217933312]

    Initial_position = Initial_action + Initial_angle

    nquanta_list_trans = np.transpose(nquanta_list)
    # solve dynamics
    # sol [ time_step, 2 * dof]
    time, sol  = Evolve_dynamics_Other_Molecule_BS_method(Initial_position,Time_step, V0, scaling_parameter, frequency, f0, nquanta_list, nquanta_list_trans)


    Period = 0.03
    fig2, ax2 = plt.subplots(nrows=1, ncols=1)

    for i in range(dof):
        ax2.plot(time / Period, sol[:, i] , label=' J ' + str(i + 1) + ' (t)')

    ax2.legend(loc='best')
    ax2.set_xlabel('t/T')
    ax2.set_yscale('log')

    ax2.set_title('BS method')

    plt.show()

