import numpy as np
import os
from scipy.integrate import odeint
from Potential import compute_angle_velocity, compute_action_velocity
from Potential import prepare_Tuple_list
import matplotlib.pyplot as plt
from Evolve_dynamics import Evolve_dynamics
from Evolve_dynamics_Using_Bulirsch import Evolve_dynamics_SWW_BS_method
import matplotlib
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()

cf = 2 * np.pi * 0.0299792458  # conversion from cm^{-1} to ps^{-1}

def Plot_Trajectory_SWW():
    D = 32924

    # This term tune the chaos
    V_phi = 7
    # V_phi = 0

    dof = 6

    Tuple_list = prepare_Tuple_list(dof)

    frequency = [1003.1, 1003.5, 1002.9, 1002.4, 1003.8, 1001.1]  # in unit of cm^{-1}

    final_time = 0.6

    Time_step = np.linspace(0,final_time,50)

    Initial_action = [2,2,3,3,3,2]
    # Initial_angle1 = [np.random.random() * np.pi * 2 for i in range(dof)]
    Initial_angle1 = [1.26902252, 0.56623613, 0.12324133, 3.25980214, 0.11200997, 0.10889953]

    print('initial angle:')
    print(Initial_angle1)

    Initial_position = Initial_action + Initial_angle1

    sol = Evolve_dynamics(Initial_position,Time_step,frequency,V_phi,D,Tuple_list)

    Period = 0.03

    # fig, ax = plt.subplots(nrows=2,ncols=1)
    #
    # for i in range(dof):
    #     ax[0].plot(Time_step/Period, sol[:,i], label = 'J' + str(i) + ' (t)')
    #     ax[1].plot(Time_step/Period, sol[:, i + dof], label='$\phi$ ' + str(i) + " (t)")
    # ax[0].legend(loc='best')
    # ax[1].legend(loc='best')
    # ax[0].set_xlabel('t/T')
    # ax[1].set_xlabel('t/T')

    phase_jitter = 0.001
    action_jitter = 0.001

    Initial_action = [2,2,3 + action_jitter,3,3,2]

    Initial_angle2 = np.array(Initial_angle1)
    Initial_angle2 = Initial_angle2.tolist()

    Initial_position = Initial_action + Initial_angle2

    # sol1 = Evolve_dynamics(Initial_position,Time_step,frequency,V_phi,D,Tuple_list)

    # fig1, ax1 = plt.subplots(nrows=2,ncols=1)
    #
    # for i in range(dof):
    #     ax1[0].plot(Time_step/Period, sol1[:,i], label = 'J' + str(i) + ' (t)')
    #     ax1[1].plot(Time_step/Period,sol1[:,i+dof] , label= '$\phi$ '+str(i) + " (t)")
    # ax1[0].legend(loc='best')
    # ax1[1].legend(loc='best')
    # ax1[0].set_xlabel('t/T')
    # ax1[1].set_xlabel('t/T')

    # difference between trajectory
    # Sol_diff = np.array(sol1) - np.array(sol)

    fig2, ax2 = plt.subplots(nrows=2,ncols=1)

    for i in range(dof):
        ax2[0].plot(Time_step/Period , sol[:,i] , label = ' $\Delta$ J' + str(i) + ' (t)' )
        ax2[1].plot(Time_step / Period, np.sin(sol[:, i + dof]), label='$sin \phi$ ' + str(i) + " (t)")

    ax2[0].legend(loc='best')
    ax2[1].legend(loc='best')
    ax2[0].set_xlabel('t/T')
    ax2[1].set_xlabel('t/T')

    ax2[0].set_yscale('log')
    # ax2[1].set_yscale('log')

    # ax2[0].set_xscale('log')
    # ax2[1].set_xscale('log')

    # plot angle velocity and action_velocity
    fig3, ax3 = plt.subplots(nrows=3,ncols=1)
    action_t_list = [sol[:, i] for i in range(dof)]
    angle_t_list = [sol[:, i + dof] for i in range(dof)]

    # data in the form [Time_step, dof]
    action_t_list = np.transpose(action_t_list)
    angle_t_list = np.transpose(angle_t_list)

    angle_velocity_list = []
    action_velocity_list = []
    angle_combination_list = []
    Len = len(Time_step)
    for i in range(Len):
        action_t = action_t_list[i]
        angle_t = angle_t_list[i]
        angle_velocity = compute_angle_velocity(frequency,action_t,angle_t,D,V_phi,Tuple_list)
        action_velocity = compute_action_velocity(V_phi,action_t,angle_t,Tuple_list)

        # compute combination of angular velocity:
        angle_combination = angle_velocity[2] + angle_velocity[3] - angle_velocity[1]
        angle_combination_list.append(angle_combination)

        angle_velocity_list.append(angle_velocity)
        action_velocity_list.append(action_velocity)

    angle_velocity_list = np.transpose(angle_velocity_list)
    action_velocity_list = np .transpose(action_velocity_list)

    for i in range(dof):
        ax3[0].plot(Time_step / Period, action_velocity_list[i] , label = 'action velocity ' + str(i+1) )
        ax3[1].plot(Time_step/Period, angle_velocity_list[i] , label  = 'angle velocity ' + str(i+1)  )
        ax3[2].plot(Time_step/Period , angle_combination_list, label = ' angle velocity combination 2 + 3  -1 ')

    ax3[0].legend(loc='best')
    ax3[1].legend(loc='best')

    ax3[0].set_xlabel('t/T')
    ax3[1].set_xlabel('t/T')

    ax3[0].set_ylim([-2000,2000])
    ax3[1].set_ylim([-2000,2000])
    ax3[2].set_ylim([-1000,1000])


    file_path = "/home/phyzch/PycharmProjects/OTOC_classical simulation/result/SW model/V=7(small perturbation)/trajectory.txt"
    with open(file_path, "w") as f:
        Time_step_len = len(Time_step)
        for i in range(Time_step_len):
            f.write("t: " + str(Time_step[i]) + "\n")
            f.write("action:  ")
            for j in range(dof):
                f.write(str(sol[i][j]) + " ")
            f.write("\n")
            f.write("angle:  ")
            for j in range(dof, 2*dof):
                f.write( str(sol[i][j]) + " ")
            f.write("\n")

    plt.show()

def Analyze_Sensitivity_number_operator():
    '''
    compute |\n_{i}(t) / \phi_{j}|^{2} matrix at early time
    :return:
    '''
    matplotlib.rcParams.update({'font.size': 20})
    D = 32924

    # This term tune the chaos
    V_phi = 251.2

    dof = 6

    Tuple_list = prepare_Tuple_list(dof)

    frequency = [1003.1, 1003.5, 1002.9, 1002.4, 1003.8, 1001.1]  # in unit of cm^{-1}

    final_time = 0.01

    Time_step = np.linspace(0, final_time, 500)

    Initial_action = [2, 2, 3, 3, 3, 2]
    Initial_angle1 = [np.random.random() * np.pi * 2 for i in range(dof)]

    print('initial angle:')
    print(Initial_angle1)

    Initial_position = Initial_action + Initial_angle1

    sol = Evolve_dynamics(Initial_position, Time_step, frequency, V_phi, D, Tuple_list)

    Period = 0.03

    Sol_diff_list = []
    phase_jitter = 0.001

    for i in range(dof):
        phase_change = np.zeros(dof)
        phase_change[i] = phase_jitter

        Initial_angle2 = np.array(Initial_angle1) + np.array(phase_change)
        Initial_angle2 = Initial_angle2.tolist()

        Initial_position = Initial_action + Initial_angle2

        sol1 = Evolve_dynamics(Initial_position, Time_step, frequency, V_phi, D, Tuple_list)

        Sol_diff = np.array(sol1) - np.array(sol)

        Sol_diff_list.append(Sol_diff)

    # form matrix for \partial n_{i} , \partial phi_{j}
    Time_step_len = len(Time_step)

    Stability_Matrix_list = []

    for t in range(Time_step_len):
        Stability_Matrix = []
        for i in range(dof):
            Stability_Matrix.append([])

        for i in range(dof):
            for j in range(dof):
                Stability_Matrix[i].append(Sol_diff_list[j][t][i] / phase_jitter)

        Stability_Matrix_list.append(Stability_Matrix)

    # compute singular value of matrix at each time step

    Singular_value_list = []
    for t in range(Time_step_len):
        u, s, vh = np.linalg.svd(Stability_Matrix_list[t])
        s = -np.sort(-s)
        Singular_value_list.append(s)

    Singular_value_list = np.transpose(Singular_value_list)

    fig, ax = plt.subplots(nrows=2,ncols=1)
    for i in range(dof):
        ax[0].plot(np.array(Time_step) / Period ,Singular_value_list[i], label='singular value '+str(i))
        ax[1].plot(np.array(Time_step) / Period, Singular_value_list[i], label='singular value ' + str(i))

    ax[0].legend(loc= 'best',fontsize=14)
    ax[0].set_xlabel('t/T')

    ax[1].legend(loc= 'best',fontsize = 14)
    ax[1].set_xlabel('t/T')

    ax[0].set_yscale('log')

    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    plt.show()

def Analyze_Stability_Matrix_change_action():
    matplotlib.rcParams.update({'font.size': 20})
    D = 32924

    # This term tune the chaos
    V_phi = 251.2

    dof = 6

    Tuple_list = prepare_Tuple_list(dof)

    frequency = [1003.1, 1003.5, 1002.9, 1002.4, 1003.8, 1001.1]  # in unit of cm^{-1}

    final_time = 0.01

    Time_step = np.linspace(0, final_time, 500)

    Initial_action = [2, 2, 3, 3, 3, 2]
    Initial_angle1 = [np.random.random() * np.pi * 2 for i in range(dof)]

    print('initial angle:')
    print(Initial_angle1)

    Initial_position = Initial_action + Initial_angle1

    sol = Evolve_dynamics(Initial_position, Time_step, frequency, V_phi, D, Tuple_list)

    Period = 0.03

    Sol_diff_list = []
    action_jitter = 0.001

    for i in range(dof):
        action_change = np.zeros(dof)
        action_change[i] = action_jitter

        Initial_action1 = np.array(Initial_action) + np.array(action_change)
        Initial_action1 = Initial_action1.tolist()

        Initial_position = Initial_action1 + Initial_angle1

        sol1 = Evolve_dynamics(Initial_position, Time_step, frequency, V_phi, D, Tuple_list)

        Sol_diff = np.array(sol1) - np.array(sol)

        Sol_diff_list.append(Sol_diff)

    # form matrix for \partial n_{i} , \partial phi_{j}
    Time_step_len = len(Time_step)

    Stability_Matrix_list = []

    for t in range(Time_step_len):
        Stability_Matrix = []
        for i in range(dof):
            Stability_Matrix.append([])

        for i in range(dof):
            for j in range(dof):
                Stability_Matrix[i].append(Sol_diff_list[j][t][i] / action_jitter)

        Stability_Matrix_list.append(Stability_Matrix)

    # compute singular value of matrix at each time step

    Singular_value_list = []
    for t in range(Time_step_len):
        u, s, vh = np.linalg.svd(Stability_Matrix_list[t])
        s = -np.sort(-s)
        Singular_value_list.append(s)

    Singular_value_list = np.transpose(Singular_value_list)

    fig, ax = plt.subplots(nrows=1,ncols=1)
    for i in range(dof):
        ax.plot(np.array(Time_step) / Period ,Singular_value_list[i], label='singular value '+str(i))

    ax.legend(loc= 'best')
    ax.set_xlabel('t/T')

    # ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()




def Analyze_Stability_Matrix_for_xp_SWW(folder_path):
    matplotlib.rcParams.update({'font.size': 14})
    D = 32924

    # This term tune the chaos
    # V_phi = 75.7
    V_phi = 7

    dof = 6

    Tuple_list = prepare_Tuple_list(dof)

    frequency = np.array([1003.1, 1003.5, 1002.9, 1002.4, 1003.8, 1001.1])  # in unit of cm^{-1}

    final_time = 0.6

    Time_step_len = 500

    Time_step = np.linspace(0, final_time, Time_step_len)

    Largest_Lyapunov_exponent_in_all_simulation = 0
    Initial_angle_for_largest_eigenvalue = [0,0,0,0,0,0]

    Iterate_number = 40

    Largest_Eigenvalue_List = []
    Largest_Singularvalue_List = []
    Period = 0.03

    # we add new option. sample coherent state
    coherent_state_alpha_amplitude_list = np.array([0.5, 0.5, 0, 0.5, 0, 0])
    coherent_state_alpha_angle_list = [0,30,0, 0, 0, 0]
    coherent_state_alpha_angle_list = np.array(coherent_state_alpha_angle_list) /180 * np.pi
    coherent_state_x = coherent_state_alpha_amplitude_list * np.cos(coherent_state_alpha_angle_list)
    coherent_state_p = coherent_state_alpha_amplitude_list * np.sin(coherent_state_alpha_angle_list)
    std_for_xp = 1/2

    # Initial_action = [2 ,2, 0.1, 2 , 0.1, 0.1]
    # Initial_energy = np.sum( np.array(frequency) * np.array(Initial_action) )


    Eigenvalue_List_in_all_simulation = []
    Singularvalue_List_in_all_simulation = []

    random_angle_list = []
    random_action_list = []
    Iteration_number_per_core = int(Iterate_number / num_proc)
    Iterate_number = Iteration_number_per_core * num_proc

    for iter_index in range(Iteration_number_per_core):

        # Initial_angle = [np.random.random() * np.pi * 2 for i in range(dof)]

        random_number = std_for_xp * np.random.normal(0,std_for_xp, 2 * dof)
        x = coherent_state_x + random_number[:dof]
        p = coherent_state_p + random_number[dof:]
        Initial_action = np.power(x,2) + np.power(p,2)
        Initial_angle = np.arctan( p / x )
        Initial_action = Initial_action.tolist()
        Initial_angle = Initial_angle.tolist()

        random_angle_list.append(Initial_angle)
        random_action_list.append(Initial_action)

        Initial_position = Initial_action + Initial_angle

        #sol = Evolve_dynamics(Initial_position, Time_step, frequency, V_phi, D, Tuple_list)
        _, sol, finish_simulation = Evolve_dynamics_SWW_BS_method(Initial_position, Time_step, frequency, V_phi, D, Tuple_list)

        Sol_change_list = []   # list of trajectory after impose a phase or action jitter
        action_jitter = 0.0001

        for i in range(dof):
            action_change = np.zeros(dof)
            action_change[i] = action_jitter

            Initial_action1 = np.array(Initial_action) + np.array(action_change)
            Initial_action1 = Initial_action1.tolist()

            Initial_position = Initial_action1 + Initial_angle

            # sol1 = Evolve_dynamics(Initial_position, Time_step, frequency, V_phi, D, Tuple_list)
            _, sol1, finish_simulation = Evolve_dynamics_SWW_BS_method(Initial_position, Time_step, frequency, V_phi, D, Tuple_list)

            Sol_change_list.append(sol1)

        phase_jitter = 0.0001
        for i in range(dof):
            phase_change = np.zeros(dof)
            phase_change[i] = phase_jitter

            Initial_angle1 = np.array(Initial_angle) + np.array(phase_change)
            Initial_angle1 = Initial_angle1.tolist()

            Initial_position = Initial_action + Initial_angle1

            # sol1 = Evolve_dynamics(Initial_position, Time_step, frequency, V_phi, D, Tuple_list)
            _, sol1, finish_simulation = Evolve_dynamics_SWW_BS_method(Initial_position, Time_step, frequency, V_phi, D, Tuple_list)

            Sol_change_list.append(sol1)

        # we first compute \partial Q_{i} / \partial J_{j} or \partial Q_{i} / \partial theta_{j}

        # for original dynamics
        XP_matrix = []
        for i in range(2*dof):
            XP_matrix.append([])

        Time_step_len = len(Time_step)
        for t in range(Time_step_len):
            for i in range(dof):
                J = sol[t][i]
                phi = sol[t][i+dof]

                Q = np.sqrt(2*J) * np.cos(phi)
                P = np.sqrt(2*J) * np.sin(phi)

                XP_matrix[i].append(Q)
                XP_matrix[i+dof].append(P)
        XP_matrix = np.array(XP_matrix)

        Diff_XP_matrix_list = []  # change of XP matrix after we change action/ angle  :  Delta Q_{i}. or Delta P_{i}
        for i in range( 2 * dof):
            XP_matrix_new = []
            for j in range(2*dof):
                XP_matrix_new.append([])

            Sol_after_change = Sol_change_list[i]

            for t in range(Time_step_len):
                for j in range(dof):
                    J = Sol_after_change[t][j]
                    phi = Sol_after_change[t][j + dof]

                    Q = np.sqrt(2*J) * np.cos(phi)
                    P = np.sqrt(2*J) * np.sin(phi)

                    XP_matrix_new[j].append(Q)
                    XP_matrix_new[j+dof].append(P)

            XP_matrix_new = np.array(XP_matrix_new)
            Diff_XP_matrix = XP_matrix_new - XP_matrix

            Diff_XP_matrix_list.append(Diff_XP_matrix)

        #  Diff_XP_matrix_list: size [2*dof, 2*dof, Time_len]

        Stability_Matrix_list = [] # Stability_Matrix at different time step

        for t in range(Time_step_len):
            Stability_Matrix= np.zeros([2*dof, 2*dof])

            for i in range(2*dof):
                for j in range(dof):
                    # {Q_{i}(t) ,Q_{j}}
                    Stability_Matrix[i][j] = (Diff_XP_matrix_list[j][i][t] / action_jitter * np.sqrt(2 * Initial_action[j]) * np.sin(Initial_angle[j]) +
                                              Diff_XP_matrix_list[j+dof][i][t]/phase_jitter * np.cos(Initial_angle[j]) / np.sqrt(2 * Initial_action[j])  )   # {, Q_{j} }
                    # {Q_{i}(t) , P_{j}}
                    Stability_Matrix[i][j+dof] =  (Diff_XP_matrix_list[j][i][t] / action_jitter *  np.sqrt(2*Initial_action[j]) * np.cos(Initial_angle[j]) -
                                              Diff_XP_matrix_list[j+dof][i][t]/phase_jitter * np.sin(Initial_angle[j]) / np.sqrt(2*Initial_action[j]) )      # { , P_{j}}

            Stability_Matrix_list.append(Stability_Matrix)

        Singular_value_list = []
        Eigen_value_list = []
        for t in range(Time_step_len):
            u, s, vh = np.linalg.svd(Stability_Matrix_list[t])
            s = -np.sort(-s)
            Singular_value_list.append(s)
            eigenvalue = np.power(s,2)
            Eigen_value_list.append(eigenvalue)

        # Now we have Singular value at different time
        Singular_value_list = np.transpose(Singular_value_list)
        Eigen_value_list = np.transpose(Eigen_value_list)

        Largest_Lyapunov_exponent = np.log(Eigen_value_list[0][-1]) / (2*Time_step[-1])

        Eigenvalue_List_in_all_simulation.append(Eigen_value_list)
        Singularvalue_List_in_all_simulation.append(Singular_value_list)

        if(Largest_Lyapunov_exponent > Largest_Lyapunov_exponent_in_all_simulation):
            Largest_Lyapunov_exponent_in_all_simulation = Largest_Lyapunov_exponent
            Initial_angle_for_largest_eigenvalue = Initial_angle
            Largest_Eigenvalue_List = Eigen_value_list
            Largest_Singularvalue_List = Singular_value_list

    # combine results in different process together:
    Eigenvalue_List_in_all_simulation = np.real(Eigenvalue_List_in_all_simulation)
    Singularvalue_List_in_all_simulation = np.real(Singularvalue_List_in_all_simulation)
    random_angle_list = np.real(random_angle_list)
    random_action_list = np.real(random_action_list)

    # size: [num_proc, iteration_number_per_core, 2*dof, Time_step_len]
    recv_Eigenvalue_list_in_all_simulation = []
    recv_Singular_value_List_in_all_simulation = []
    if (rank == 0):
        recv_Eigenvalue_list_in_all_simulation = np.empty(
            [num_proc, Iteration_number_per_core, 2 * dof, Time_step_len], dtype=np.float64)
        recv_Singular_value_List_in_all_simulation = np.empty(
            [num_proc, Iteration_number_per_core, 2 * dof, Time_step_len], dtype=np.float64)

    comm.Gather(Eigenvalue_List_in_all_simulation, recv_Eigenvalue_list_in_all_simulation, 0)
    comm.Gather(Singularvalue_List_in_all_simulation, recv_Singular_value_List_in_all_simulation, 0)

    # angle_list : size : [num_proc, iteration_number_per_core, dof]
    recv_angle_list = []
    recv_action_list = []
    if (rank == 0):
        recv_angle_list = np.empty([num_proc, Iteration_number_per_core, dof], dtype=np.float64)
        recv_action_list = np.empty([num_proc, Iteration_number_per_core, dof], dtype=np.float64)
    comm.Gather(random_angle_list, recv_angle_list, 0)
    comm.Gather(random_action_list, recv_action_list, 0)

    if(rank == 0):

        # convert recved data to original format
        # Now shape [iterate_number , 2 * dof,  Time_step_len ]
        recv_Eigenvalue_list_shape = recv_Eigenvalue_list_in_all_simulation.shape
        Eigenvalue_List_in_all_simulation = np.reshape(recv_Eigenvalue_list_in_all_simulation,
                                                       (recv_Eigenvalue_list_shape[0] * recv_Eigenvalue_list_shape[1],
                                                        recv_Eigenvalue_list_shape[2], recv_Eigenvalue_list_shape[3]))

        recv_Singular_value_shape = recv_Singular_value_List_in_all_simulation.shape
        Singularvalue_List_in_all_simulation = np.reshape(recv_Singular_value_List_in_all_simulation,
                                                          (recv_Singular_value_shape[0] * recv_Eigenvalue_list_shape[1],
                                                           recv_Singular_value_shape[2], recv_Eigenvalue_list_shape[3])
                                                          )
        # Now shape : [Iterate_number , dof ]
        recv_angle_list_shape = recv_angle_list.shape
        random_angle_list = np.reshape(recv_angle_list,
                                       (recv_angle_list_shape[0] * recv_angle_list_shape[1], recv_angle_list_shape[2])
                                       )
        recv_action_list_shape = recv_action_list.shape
        random_action_list = np.reshape(recv_action_list,
                                        (recv_action_list_shape[0] * recv_action_list_shape[1], recv_action_list_shape[2])
                                        )

        print("random angle received:  " + str(random_angle_list))

        # Now compute Largest Singular_value_list and Largest_eigenvalue_list and Largest Lyapunov_exponent_list and their initial angles.
        Largest_Lypunov_exponent_in_all_simulation = 0
        Initial_action_for_largest_eigenvalue = []
        for i in range(Iterate_number):
            Largest_Lyapunov_exponent = np.log(Eigenvalue_List_in_all_simulation[i][0][-1]) / (2 * Time_step[-1])
            if (Largest_Lyapunov_exponent > Largest_Lypunov_exponent_in_all_simulation):
                Largest_Lypunov_exponent_in_all_simulation = Largest_Lyapunov_exponent
                Initial_angle_for_largest_eigenvalue = random_angle_list[i]
                Initial_action_for_largest_eigenvalue = random_action_list[i]
                Largest_Eigenvalue_List = Eigenvalue_List_in_all_simulation[i]
                Largest_Singularvalue_List = Singularvalue_List_in_all_simulation[i]

        Singular_value_list = Largest_Singularvalue_List
        Eigen_value_list = Largest_Eigenvalue_List

        print('initial action')
        print(Initial_action_for_largest_eigenvalue)
        print('initial angle:')
        Initial_angle = Initial_angle_for_largest_eigenvalue
        print(Initial_angle)


        # plot result
        fig, ax = plt.subplots(nrows=1, ncols=1)
        # for i in range( 2*dof):
        #     ax.plot(np.array(Time_step) / Period, Eigen_value_list[i], label='Eigenvalue ' + str(i+1)+" for $M^{2}$ ")
        ax.plot(np.array(Time_step) / Period, Eigen_value_list[0], label=' Largest Eigenvalue  for $M^{2}$ ')

        ax.legend(loc='best')
        ax.set_xlabel('t/T')

        # ax.set_xscale('log')
        ax.set_yscale('log')

        fig1, ax1 = plt.subplots(nrows=1, ncols=1)
        for i in range(2 * dof):
            ax1.plot(np.array(Time_step) / Period, Singular_value_list[i],
                     label='Singular value ' + str(i + 1) + ' for M')
        ax1.plot(np.array(Time_step) / Period, Singular_value_list[0], label='Largest Singular value for M')

        ax1.legend(loc='best')
        ax1.set_xlabel('t/T')

        # ax1.set_xscale('log')
        ax1.set_yscale('log')

        Lyapunov_exponent_list = []
        for i in range(2 * dof):
            Lyapunov_exponent = np.log(Eigen_value_list[i][1:]) / (2 * np.array(Time_step[1:]))
            Lyapunov_exponent_list.append(Lyapunov_exponent)

        # plot Lyapunov exponent
        fig2, ax2 = plt.subplots(nrows=1, ncols=1)
        for i in range(2 * dof):
            ax2.plot(np.array(Time_step[1:]) / Period, Lyapunov_exponent_list[i],
                     label='Lyapunov exponent mode ' + str(i + 1))

        # ax2.legend(loc = 'best')
        ax2.set_xlabel('t/T')
        ax2.set_title('Lyapunov exponent')

        # plot average result over angle in torus
        fig3, ax3 = plt.subplots(nrows=1, ncols=1)

        Average_Eigenvalue_list = np.mean(Eigenvalue_List_in_all_simulation, 0)
        Average_Singular_value_list = np.mean(Singularvalue_List_in_all_simulation, 0)

        # for i in range( 2*dof):
        #     ax3.plot(np.array(Time_step) / Period, Average_Eigenvalue_list[i], label='Average Eigenvalue ' + str(i+1)+" for $M^{2}$ ")

        ax3.plot(np.array(Time_step) / Period, Average_Eigenvalue_list[0],
                 label='Largest Average Eigenvalue  for $M^{2}$ ')

        ax3.set_xlabel('t/T')
        ax3.set_ylabel('Average Eigenvalue')

        ax3.set_title('Average Eigenvalue')

        # ax3.legend(loc = 'best')
        ax3.set_yscale('log')

        # Average_Lyapunov exponent
        Lyapunov_exponent_all = []
        for i in range(Iterate_number):
            Lyapunov_exponent_single_trajectory = []
            for j in range(2 * dof):
                Lyapunov_exponent = np.log(Eigenvalue_List_in_all_simulation[i][j][1:]) / (2 * np.array(Time_step[1:]))
                Lyapunov_exponent_single_trajectory.append(Lyapunov_exponent)
            Lyapunov_exponent_all.append(Lyapunov_exponent_single_trajectory)

        Average_Lyapunov_exponent = np.mean(Lyapunov_exponent_all, 0)


        # compute Intermittency : R_{n} = <f(x)^{n}> / <f(x)>^{n}
        intermittency_order = [2, 3,  5 ]
        intermittency_order_number = len(intermittency_order)
        intermittency_for_largest_eigenvalue = []
        for n in intermittency_order:
            R = np.mean( np.power(Eigenvalue_List_in_all_simulation , n) , 0 ) / np.power( np.mean(Eigenvalue_List_in_all_simulation , 0 ) , n )
            intermittency_for_largest_eigenvalue.append(R[0])

        fig5, ax5  = plt.subplots(nrows=1, ncols=1)
        for n in range(intermittency_order_number):
            ax5.plot(Time_step, intermittency_for_largest_eigenvalue[n] , label = 'n = ' + str(intermittency_order[n]))
        ax5.legend(loc = 'best')
        ax5.set_xlabel('t(ps)')
        ax5.set_ylabel('$<p^{n}> / <p>^{n} $')
        ax5.set_title('Intermittency')
        ax5.set_yscale('log')

        # save the result
        file_path = os.path.join(folder_path, "Average_Eigenvalue.txt")
        f = open(file_path, "w")
        f.write(str(dof) + "\n")
        Data_len = len(Time_step)
        for i in range(Data_len):
            f.write(str(Time_step[i] / Period) + " ")
        f.write("\n")

        for i in range(dof * 2):
            for j in range(Data_len):
                f.write(str(Average_Eigenvalue_list[i][j]) + " ")
            f.write("\n")

        f.close()

        file_path_all_state = os.path.join(folder_path, "All_trajectory_eigenvalue.txt")
        f = open(file_path_all_state, "w")
        f.write(str(Iterate_number) + "\n")
        for i in range(Data_len):
            f.write(str(Time_step[i] / Period) + " ")
        f.write("\n")
        for i in range(Iterate_number):
            for j in range(Data_len):
                f.write(str(Eigenvalue_List_in_all_simulation[i][0][j]) + " ")
            f.write("\n")
        f.close()

        file_path1 = os.path.join(folder_path, "Largest_Eigenvalue_for_chaotic_regime.txt")
        f = open(file_path1, "w")
        f.write(str(dof) + "\n")
        Data_len = len(Time_step)
        for i in range(Data_len):
            f.write(str(Time_step[i] / Period) + " ")
        f.write("\n")

        for i in range(dof * 2):
            for j in range(Data_len):
                f.write(str(Largest_Eigenvalue_List[i][j]) + " ")
            f.write("\n")

        f.close()

        file_path2 = os.path.join(folder_path, "Largest_angle.txt")
        f = open(file_path2, "w")
        f.write('angle for largest Lyapunov exponent')
        f.write('\n')
        f.write(str(Initial_angle_for_largest_eigenvalue))
        f.write('\n')
        f.write('action for largest Lyapunov exponent')
        f.write('\n')
        f.write(str(Initial_action_for_largest_eigenvalue))
        f.write('\n')
        f.write('all angle and action : ')
        for i in range(Iterate_number):
            f.write(str(random_angle_list[i]))
            f.write('\n')
            f.write(str(random_action_list[i]))
            f.write('\n')
            f.write('\n')

        f.close()

        file_path3 = os.path.join(folder_path, 'Average_over_Lyapunov_exponent.txt')
        f = open(file_path3, "w")
        f.write(str(dof) + "\n")
        Data_len = len(Time_step)
        for i in range(1, Data_len):
            f.write(str(Time_step[i] / Period) + " ")
        f.write("\n")
        for i in range(dof * 2):
            for j in range(Data_len - 1):
                f.write(str(Average_Lyapunov_exponent[i][j]) + " ")
            f.write("\n")
        f.close()

        intermittency_file = os.path.join(folder_path, 'intermittency.txt')
        with open (intermittency_file, "w") as f:
            f.write("intermittency order:  \n ")
            for i in range(intermittency_order_number):
                f.write( str(intermittency_order[i]) + " ")
            f.write('\n')
            for i in range(Time_step_len):
                f.write(str(Time_step[i]) + " ")
            f.write("\n")
            for i in range(intermittency_order_number):
                for j in range(Time_step_len):
                    f.write(str(intermittency_for_largest_eigenvalue[i][j]) + " ")
                f.write("\n")


        # save data:
        # file_path = os.path.join(folder_path,"Microcanonical Average Eigenvalue.txt")
        # f = open(file_path,"w")
        # f.write(str(dof) + "\n")
        # Data_len = len(Time_step)
        # for i in range(Data_len):
        #     f.write(str(Time_step[i] / Period) + " ")
        # f.write("\n")
        #
        # for i in range(dof * 2):
        #     for j in range(Data_len):
        #         f.write(str(Average_Eigenvalue_list[i][j]) + " ")
        #     f.write("\n")
        #
        # f.close()
        #
        # log_path = os.path.join(folder_path,"Micaocanonical Average action angle.txt")
        # f1 = open(log_path,"w")
        # f1.write(str(Iterate_number) + "\n")
        # for i in range(Iterate_number):
        #     for j in range(dof):
        #         f1.write(str(Initial_action_list[i][j]) + " ")
        #     f1.write("\n")
        #     for j in range(dof):
        #         f1.write(str(Initial_angle_list[i][j]) + " ")
        #     f1.write("\n")
        #
        # f1.close()


        plt.show()
