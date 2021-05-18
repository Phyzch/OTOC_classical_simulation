import numpy as np
from SCCL2_potential import Generate_n_quanta_list_for_SCCL2
from Bulirsch_Stoer_module import bulStoer
from Evolve_dynamics import Other_Molecules , SCCL2_Realistic_Hamiltonian , anharmonic_oscillator
import matplotlib.pyplot as plt
import matplotlib
import os

def Evolve_dynamics_Other_Molecule_BS_method(Initial_position, Time_step, V0, scaling_parameter, frequency, f0, nquanta_list, nquanta_list_trans):
    t0 = 0
    time_step_size = Time_step[1]

    final_time = Time_step[-1]

    Initial_position = np.array(Initial_position)

    tol = 1e-6

    time, position , finish_simulation = bulStoer(Other_Molecules, t0, Initial_position, final_time, time_step_size, args = (V0, scaling_parameter, frequency, f0, nquanta_list, nquanta_list_trans) , tol = tol )



    return time, position , finish_simulation

def Evolve_dynamics_Realistic_SCCL2_BS_method(Initial_position, Time_step, frequency, Coefficient, nquanta_list):
    t0 = 0
    time_step_size = Time_step[1]

    final_time = Time_step[-1]

    Initial_position = np.array(Initial_position)

    tol = 1e-6

    time, position , finish_simulation = bulStoer( SCCL2_Realistic_Hamiltonian, t0, Initial_position, final_time, time_step_size, args = (frequency, Coefficient, nquanta_list) , tol = tol )

    return time, position , finish_simulation

def Evolve_dynamics_SWW_BS_method(Initial_position, Time_step , frequency, V_phi, D, Tuple_list):
    t0 = 0
    time_step_size = Time_step[1]

    final_time = Time_step[-1]

    Initial_position = np.array(Initial_position)

    tol = 1e-6

    time, position, finish_simulation = bulStoer( anharmonic_oscillator, t0 , Initial_position, final_time, time_step_size, args = (frequency,V_phi,D,Tuple_list)  , tol = tol )

    return time , position, finish_simulation

def Plot_Trajectory_Other_molecule_BS_method():
    matplotlib.rcParams.update({'font.size': 20})
    Iter_number = 1

    # This parameter tune the chaos
    V0 = 3050

    scaling_parameter = 0.2

    # Cyclopentane
    # frequency = [3062, 2914, 1445, 1349, 1275, 1203, 1043, 905, 695]
    # dof = 9
    # # specify initial position and angle
    # Initial_action = [0, 0, 0, 1, 0, 0, 1, 0, 0]
    # Initial_angle = [2.4442179158763073, 1.0017940119372089, 4.250017300134825, 2.5048770218889747, 0.5626141486498147,
    #                  3.158573153008315, 0.7225434637992191, 6.038088800623476, 4.530226217933312]

    # Cyclopentanone
    frequency = [2210 , 2222 , 2966 , 2945 , 2130 , 2126 , 2880 , 2880]
    dof = 8
    Initial_action = [1 ,1 ,1 ,0 ,0 ,1 ,0 ,0]
    # Initial_action = [2,2,2,3,2,2,2,2]
    Initial_angle = [np.random.random() * np.pi * 2 for i in range(dof)]
    Initial_angle = [5.50931976, 0.46196768, 6.060698 ,  1.1348094 , 5.28584671 ,4.77783969,
 5.87302753 ,1.49436109]
    f0 = 500

    folder_path = "/home/phyzch/PycharmProjects/OTOC_classical simulation/result/Other Molecule/Cyclopentanone/11100100/"

    nquanta_list = Generate_n_quanta_list_for_SCCL2(dof)

    final_time = 0.5

    Time_step = np.linspace(0, final_time, 300)

    Initial_position = Initial_action + Initial_angle

    nquanta_list_trans = np.transpose(nquanta_list)

    mean_action = np.mean(Initial_action)
    # solve dynamics
    # sol [ time_step, 2 * dof]
    sol_list = []
    dev_t_list = []
    time = 0
    for i in range(Iter_number):
        if(i!=0):
            Initial_angle = [np.random.random() * np.pi * 2 for i in range(dof)]
            Initial_position = Initial_action + Initial_angle
        time, sol, _  = Evolve_dynamics_Other_Molecule_BS_method(Initial_position,Time_step, V0, scaling_parameter, frequency, f0, nquanta_list, nquanta_list_trans)
        # Define quantity which is deviation of action
        Len = len(time)
        dev_t = []
        for j in range(Len):
            dev =  np.sqrt(np.sum (np.power(sol[j][:dof] - Initial_action , 2) ) / dof)
            dev_t.append(dev)
        dev_t_list.append(dev_t)
        sol_list.append(sol)

    sol = np.mean(sol_list , 0)

    Period = 0.03
    fig2, ax2 = plt.subplots(nrows=1, ncols=1)

    for i in range(dof):
        ax2.plot(time / Period, sol[:, i] , label=' J ' + str(i + 1) + ' (t)')

    ax2.legend(loc='best')
    ax2.set_xlabel('t/T')
    # ax2.set_yscale('log')
    ax2.set_title('BS method')

    dev_t_mean = np.mean(dev_t_list, 0)
    fig3, ax3 =  plt.subplots(nrows=1, ncols=1)
    ax3.plot(time , dev_t_mean, linewidth = 6)
    ax3.set_xlabel('t(ps)')
    ax3.set_title('< $ \sqrt{\sum_{i=1}^{N} (J_{i}(t) - J_{i}(0))^{2} / N} $>')
    # ax3.set_yscale('log')

    # save the result
    file_path = os.path.join(folder_path, "action_motion.txt")
    f = open(file_path, "w")
    f.write(str(dof) + "\n")
    Data_len = len(time)
    for i in range(Data_len):
        f.write(str(time[i]) + " ")
    f.write("\n")

    for i in range(Data_len):
        f.write(str(dev_t_mean[i]) + " ")
    f.write("\n")

    f.close()

    plt.show()

