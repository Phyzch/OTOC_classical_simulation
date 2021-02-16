import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Evolve_dynamics import Evolve_dynamics_SCCL2 , Evolve_dynamics_SCCL2_Realistic_Hamiltonian
from SCCL2_potential import Generate_n_quanta_list_for_SCCL2, SCCL2_angle_velocity, SCCL2_action_velocity, SCCL2_Realistic_Hamiltonian_action_velocity, SCCL2_Realistic_Hamiltonian_angle_velocity
from SCCL2_potential import Read_Realistic_SCCL2

cf = 2 * np.pi * 0.0299792458  # conversion from cm^{-1} to ps^{-1}

def SCCL2_Analyze_Sensitivity_number_operator():
    '''
    compute |\n_{i}(t) / \phi_{j}|^{2} matrix at early time
    :return:
    '''
    matplotlib.rcParams.update({'font.size': 20})


    # This term tune the chaos
    V0 = 300

    scaling_parameter = 0.2

    frequency = [1149, 508, 291, 474, 843, 333]

    dof = 6

    f0 = 500

    nquanta_list = Generate_n_quanta_list_for_SCCL2(dof)

    final_time = 0.015

    Time_step = np.linspace(0, final_time, 500)

    Initial_action = [2, 2, 3, 3, 3, 2]
    Initial_angle1 = [np.random.random() * np.pi * 2 for i in range(dof)]

    print('initial angle:')
    print(Initial_angle1)

    Initial_position = Initial_action + Initial_angle1

    sol = Evolve_dynamics_SCCL2(Initial_position,Time_step,V0,scaling_parameter,frequency,f0,nquanta_list)

    Period = 0.03

    Sol_diff_list = []
    phase_jitter = 0.001

    for i in range(dof):
        phase_change = np.zeros(dof)
        phase_change[i] = phase_jitter

        Initial_angle2 = np.array(Initial_angle1) + np.array(phase_change)
        Initial_angle2 = Initial_angle2.tolist()

        Initial_position = Initial_action + Initial_angle2

        sol1 = Evolve_dynamics_SCCL2(Initial_position,Time_step,V0,scaling_parameter,frequency,f0,nquanta_list)

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


def SCCL2_Analyze_Stability_Matrix_for_xp():
    matplotlib.rcParams.update({'font.size': 14})

    # This term tune the chaos
    V0 = 300

    scaling_parameter = 0.2

    frequency = [1149, 508, 291, 474, 843, 333]

    dof = 6

    f0 = 500

    nquanta_list = Generate_n_quanta_list_for_SCCL2(dof)

    final_time = 0.03

    Time_step = np.linspace(0, final_time, 100)

    Largest_Lyapunov_exponent_in_all_simulation = 0
    Initial_angle_for_largest_eigenvalue = [0,0,0,0,0,0]

    Iterate_number = 1

    Largest_Eigenvalue_List = []
    Largest_Singularvalue_List = []
    Period = 0.03
    Initial_action = [2, 2, 3, 2 , 2 , 2]

    Eigenvalue_List_in_all_simulation = []
    Singularvalue_List_in_all_simulation = []

    for _ in range(Iterate_number):
        Initial_angle = [np.random.random() * np.pi * 2 for i in range(dof)]

        # print('initial action')
        # print(Initial_action)
        # print('initial angle:')
        # print(Initial_angle)

        Initial_position = Initial_action + Initial_angle

        sol = Evolve_dynamics_SCCL2(Initial_position,Time_step,V0,scaling_parameter,frequency,f0,nquanta_list)

        Sol_change_list = []   # list of trajectory after impose a phase or action jitter
        action_jitter = 0.01

        for i in range(dof):
            action_change = np.zeros(dof)
            action_change[i] = action_jitter

            Initial_action1 = np.array(Initial_action) + np.array(action_change)
            Initial_action1 = Initial_action1.tolist()

            Initial_position = Initial_action1 + Initial_angle

            sol1 = Evolve_dynamics_SCCL2(Initial_position,Time_step,V0,scaling_parameter,frequency,f0,nquanta_list)

            Sol_change_list.append(sol1)

        phase_jitter = 0.01
        for i in range(dof):
            phase_change = np.zeros(dof)
            phase_change[i] = phase_jitter

            Initial_angle1 = np.array(Initial_angle) + np.array(phase_change)
            Initial_angle1 = Initial_angle1.tolist()

            Initial_position = Initial_action + Initial_angle1

            sol1 = Evolve_dynamics_SCCL2(Initial_position,Time_step,V0,scaling_parameter,frequency,f0,nquanta_list)

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


    Singular_value_list = Largest_Singularvalue_List
    Eigen_value_list = Largest_Eigenvalue_List

    print('initial action')
    print(Initial_action)
    print('initial angle:')
    Initial_angle = Initial_angle_for_largest_eigenvalue
    print(Initial_angle)

    # plot result
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for i in range( 2*dof):
        ax.plot(np.array(Time_step) / Period, Eigen_value_list[i], label='Eigenvalue ' + str(i+1)+" for $M^{2}$ ")

    ax.legend(loc='best')
    ax.set_xlabel('t/T')

    # ax.set_xscale('log')
    ax.set_yscale('log')

    fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    for i in range(2 * dof):
        ax1.plot(np.array(Time_step)/Period, Singular_value_list[i],label = 'Singular value '+ str(i+1) + ' for M' )
    ax1.legend(loc = 'best')
    ax1.set_xlabel('t/T')

    # ax1.set_xscale('log')
    ax1.set_yscale('log')

    Lyapunov_exponent_list = []
    for i in range(2 * dof):
        Lyapunov_exponent = np.log(Eigen_value_list[i]) / (2 * np.array(Time_step))
        Lyapunov_exponent_list.append(Lyapunov_exponent)

    # plot Lyapunov exponent
    fig2, ax2 = plt.subplots(nrows=1,ncols=1)
    for i in range(2* dof):
        ax2.plot( np.array(Time_step)/Period, Lyapunov_exponent_list[i] , label = 'Lyapunov exponent mode ' + str(i+1))

    # ax2.legend(loc = 'best')
    ax2.set_xlabel('t/T')
    ax2.set_title('Lyapunov exponent')

    # plot average result over angle in torus
    fig3, ax3 = plt.subplots(nrows=1,ncols=1)

    Average_Eigenvalue_list = np.mean(Eigenvalue_List_in_all_simulation,0)
    Average_Singular_value_list = np.mean(Singularvalue_List_in_all_simulation,0)

    for i in range( 2*dof):
        ax3.plot(np.array(Time_step) / Period, Average_Eigenvalue_list[i], label='Average Eigenvalue ' + str(i+1)+" for $M^{2}$ ")

    ax3.set_xlabel('t/T')
    ax3.set_ylabel('Average Eigenvalue')

    ax3.set_title('Average Eigenvalue')

    # ax3.legend(loc = 'best')
    ax3.set_yscale('log')
    plt.show()


def Plot_Trajectory_SCCL2():
    # This parameter tune the chaos
    V0 = 300

    scaling_parameter = 0.2

    frequency = [1149, 508, 291, 474, 843, 333]
    # frequency = np.array([1003.1, 1003.5, 1002.9, 1002.4, 1003.8, 1001.1])

    dof = 6

    f0 = 500

    nquanta_list = Generate_n_quanta_list_for_SCCL2(dof)

    final_time = 0.22

    Time_step = np.linspace(0, final_time, 100)

    # specify initial position and angle
    Initial_action = [2, 2, 3, 3, 3, 2]
    Initial_angle1 = [np.random.random() * np.pi * 2 for i in range(dof)]
    Initial_angle1 = [0.21703024880194466, 2.045895500165229, 5.087988069355274, 6.079494689252139, 4.3671156576438594, 1.4091667949185125]

    print('initial angle:')
    print(Initial_angle1)

    Initial_position = Initial_action + Initial_angle1

    # solve dynamics
    sol = Evolve_dynamics_SCCL2(Initial_position, Time_step, V0, scaling_parameter, frequency, f0, nquanta_list)

    Period = 0.03

    phase_jitter = 0.001
    action_jitter = 0.001

    Initial_action = [2, 2, 3 + action_jitter, 3, 3, 2]

    Initial_angle2 = np.array(Initial_angle1)
    Initial_angle2 = Initial_angle2.tolist()

    Initial_position = Initial_action + Initial_angle2

    # sol1 = Evolve_dynamics_SCCL2(Initial_position, Time_step, V0, scaling_parameter, frequency, f0, nquanta_list)
    #
    # Sol_diff = np.array(sol1) - np.array(sol)

    fig2, ax2 = plt.subplots(nrows=3, ncols=1)

    for i in range(dof):
        ax2[0].plot(Time_step / Period, sol[:, i] , label=' $\Delta$ J' + str(i + 1) + ' (t)')
        ax2[1].plot(Time_step / Period, np.cos(sol[:, i + dof]) , label='$cos \phi$ ' + str(i + 1) + " (t)")
        ax2[2].plot(Time_step/ Period, np.sin(sol[:,i+dof]) , label = '$sin \phi$ ' + str(i+1) + " (t)" )

    ax2[0].legend(loc='best')
    ax2[1].legend(loc='best')
    ax2[2].legend(loc = 'best')
    ax2[0].set_xlabel('t/T')
    ax2[1].set_xlabel('t/T')
    ax2[2].set_xlabel('t/T')

    ax2[0].set_yscale('log')
    # ax2[1].set_yscale('log')

    # compute angle velocity at given time
    fig3 , ax3 = plt.subplots(nrows=2,ncols=1)
    action_t_list = [ sol[:, i ] for i in range(dof) ]
    angle_t_list = [sol[:,i+dof] for i in range(dof)]

    # data in the form [Time_step, dof]
    action_t_list = np.transpose(action_t_list)
    angle_t_list = np.transpose(angle_t_list)

    angle_velocity_list = []
    action_velocity_list = []
    Len = len(Time_step)
    for i in range(Len):
        action_t = action_t_list[i]
        angle_t = angle_t_list[i]
        angle_velocity = SCCL2_angle_velocity(action_t,angle_t,V0,scaling_parameter,frequency,f0,nquanta_list)
        action_velocity = SCCL2_action_velocity(action_t,angle_t, V0, scaling_parameter, frequency, f0, nquanta_list)

        angle_velocity_list.append(angle_velocity)
        action_velocity_list.append(action_velocity)

    angle_velocity_list = np.transpose(angle_velocity_list)
    action_velocity_list = np.transpose(action_velocity_list)

    for i in range(dof):
        ax3[0].plot(Time_step / Period, action_velocity_list[i] * cf , label = 'action velocity ' + str(i+1))
        ax3[1].plot(Time_step/ Period, angle_velocity_list[i] * cf, label = 'angle velocity  ' + str(i+1) )

    ax3[0].legend(loc='best')
    ax3[1].legend(loc='best')
    ax3[0].set_xlabel('t/T')
    ax3[1].set_xlabel('t/T')

    ax3[0].set_ylim([-400,400])
    ax3[1].set_ylim([-400,400])



    plt.show()

def Plot_Trajectory_SCCL2_Realistic_Hamiltonian():

    frequency, Coefficient, nquanta_list = Read_Realistic_SCCL2()

    Coefficient = np.array(Coefficient) * 1

    dof = 6
    final_time = 0.06

    Time_step = np.linspace(0, final_time, 100)

    Iteration_number = 1

    # specify initial position and angle
    phase_jitter = 0.001
    action_jitter = 0.001
    Initial_action = [2, 2, 3, 3, 3, 2]
    # Initial_action = [6.3809, 3.2045, 5.5516, 2.8994, 5.1202, 6.0236]
    Initial_action = [6.2187, 5.5134, 1.0357, 3.2284, 4.9875, 2.896]
    Initial_action1 = [2, 2, 3, 3, 3 + action_jitter, 2]

    max_action_in_all_simulation = 0
    Index = -1 # Index for max action angle.
    max_sol = []
    max_Sol_diff = []
    Initial_angle_list = []


    for i in range(Iteration_number):
        # Initial_angle = [np.random.random() * np.pi * 2 for i in range(dof)]
        Initial_angle = [14.753, 6.216, 4.992, 9.7515, 9.314, 9.846]
        Initial_angle = [31.4, 6.04, 8.8, 16.3, 19.86, 14.386]
        Initial_angle = [31.93947782165516, 6.097124720812282, 8.845719425253785, 16.528919283844296, 20.21703311486376, 14.099748439492283]
        # for j in range(dof):
        #     Initial_angle[j] = (0.02 * (np.random.random()-0.5) * 2 +1) * Initial_angle[j]

        Initial_angle_list.append(Initial_angle)
        print(Initial_angle)
        # print('initial angle:' + str(Initial_angle1))

        Initial_position = Initial_action + Initial_angle

        # solve dynamics
        sol = Evolve_dynamics_SCCL2_Realistic_Hamiltonian(Initial_position,Time_step,frequency,Coefficient,nquanta_list)

        max_action = np.max( [sol[:][i] for i in range(dof)] )

        # Initial_angle1 = Initial_angle
        # Initial_position1 = Initial_action1 + Initial_angle1
        # sol1 = Evolve_dynamics_SCCL2_Realistic_Hamiltonian(Initial_position1, Time_step, frequency, Coefficient,
        #                                                    nquanta_list)
        #
        # Sol_diff = np.array(sol1) - np.array(sol)
        #
        # max_action = np.max([ max(Sol_diff[:][i]) for i in range(dof)])
        if(max_action > max_action_in_all_simulation):
            max_action_in_all_simulation = max_action
            max_sol = sol
            # max_Sol_diff = Sol_diff
            Index = i

    print('Initial angle: ' + str(Initial_angle_list[Index]))
    sol = max_sol
    Sol_diff = max_Sol_diff

    Period = 0.03



    fig2, ax2 = plt.subplots(nrows=3, ncols=1)

    for i in range(dof):
        ax2[0].plot(Time_step/Period , sol[:,i] , label = 'J' + str(i+1) + ' (t)')
        ax2[1].plot(Time_step / Period, np.cos(sol[:, i + dof]), label='$cos \phi$ ' + str(i + 1) + " (t)")
        ax2[2].plot(Time_step / Period, np.sin(sol[:, i + dof]), label='$sin \phi$ ' + str(i + 1) + " (t)")

    ax2[0].legend(loc='best')
    ax2[1].legend(loc='best')
    ax2[2].legend(loc='best')
    ax2[0].set_xlabel('t/T')
    ax2[1].set_xlabel('t/T')
    ax2[2].set_xlabel('t/T')

    ax2[0].set_yscale('log')

    fig3, ax3 = plt.subplots(nrows=3,ncols=1)

    action_t_list = [sol[:, i] for i in range(dof)]
    angle_t_list = [sol[:, i + dof] for i in range(dof)]

    # data in the form [Time_step, dof]
    action_t_list = np.transpose(action_t_list)
    angle_t_list = np.transpose(angle_t_list)

    angle_velocity_list = []
    action_velocity_list = []
    Len = len(Time_step)
    for i in range(Len):
        action_t = action_t_list[i]
        angle_t = angle_t_list[i]
        angle_velocity = SCCL2_Realistic_Hamiltonian_angle_velocity(action_t,angle_t,frequency,Coefficient,nquanta_list)
        action_velocity = SCCL2_Realistic_Hamiltonian_action_velocity(action_t,angle_t,frequency,Coefficient,nquanta_list)

        angle_velocity_list.append(angle_velocity)
        action_velocity_list.append(action_velocity)

    angle_velocity_list = np.transpose(angle_velocity_list)
    action_velocity_list = np.transpose(action_velocity_list)

    for i in range(dof):
        ax3[0].plot(Time_step/Period , sol[:,i] , label = 'J' + str(i+1) + ' (t)')
        ax3[1].plot(Time_step / Period, action_velocity_list[i] * cf, label ='action velocity '+ str(i+1) )
        ax3[2].plot(Time_step / Period, angle_velocity_list[i] * cf, label = 'angle velocity  ' +str(i+1))

    ax3[0].legend(loc='best')
    ax3[1].legend(loc='best')
    ax3[2].legend(loc='best')
    ax3[0].set_xlabel('t/T')
    ax3[1].set_xlabel('t/T')
    ax3[2].set_xlabel('t/T')

    sol_transpose = np.transpose(sol)
    for i in range(dof):
        print('action ' + str(i+1))
        print(sol_transpose[i][-20:-10])

    for i in range(dof):
        print('angle  ' + str(i+1))
        print(sol_transpose[i + dof][-20:-10])

    plt.show()


def Sample_SCCL2_scaling_angular_velocity():
    # This parameter tune the chaos
    V0 = 300

    scaling_parameter = 0.2

    frequency = [1149, 508, 291, 474, 843, 333]
    # frequency = np.array([1003.1, 1003.5, 1002.9, 1002.4, 1003.8, 1001.1])

    dof = 6

    f0 = 500

    nquanta_list = Generate_n_quanta_list_for_SCCL2(dof)

    Iteration_number = 1000

    min_max_abs_angle_velocity_in_all_sample = 100000
    optimal_angle = []
    optimal_action = []
    optimal_angle_velocity = []

    for i in range(Iteration_number):
        angle = np.random.random(6) * 2 * np.pi
        action = np.random.random(6) * 6

        angle_velocity = SCCL2_angle_velocity(action,angle,V0,scaling_parameter,frequency,f0,nquanta_list)
        # action_velocity = SCCL2_Realistic_Hamiltonian_action_velocity(action,angle,frequency,Coefficient, nquanta_list)

        max_abs_angle_velocity = np.mean(np.abs(angle_velocity))

        if (max_abs_angle_velocity < min_max_abs_angle_velocity_in_all_sample):
            min_max_abs_angle_velocity_in_all_sample = max_abs_angle_velocity
            optimal_angle = angle
            optimal_action = action
            optimal_angle_velocity = angle_velocity

    print('optimal angle velocity  ' + str(optimal_angle_velocity))
    print('optimal action:  ' + str(optimal_action))
    print('optimal angle:  ' + str(optimal_angle))


def Sample_SCCL2_Realistic_Hamiltonian_angular_velocity():
    frequency, Coefficient, nquanta_list = Read_Realistic_SCCL2()

    Coefficient_sum = np.sum( np.abs(Coefficient) )
    print('Coefficient_sum    ' + str(Coefficient_sum) )

    dof = 6

    Iteration_number = 100

    min_angle_velocity_combination_all = 100000
    Sin_coefficient_for_optimal_result = 0
    Coefficient_for_optimal_result = 0
    nquanta_for_combination_optimal = 0
    max_Effect = 0

    optimal_angle = []
    optimal_action = []
    optimal_angle_velocity = []

    for i in range(Iteration_number):
        angle = np.random.random(6) * 2 * np.pi
        action = np.random.random(6) * 6

        angle_velocity = SCCL2_Realistic_Hamiltonian_angle_velocity(action, angle, frequency, Coefficient, nquanta_list)
        # action_velocity = SCCL2_Realistic_Hamiltonian_action_velocity(action,angle,frequency,Coefficient, nquanta_list)

        min_angle_velocity_combination , Coefficient_for_combination, nquanta_for_combination = compute_min_abs_angle_velocity_combination(angle_velocity,nquanta_list,Coefficient)

        angle_constant = np.sum(np.array(angle) * np.array(nquanta_for_combination))
        Sin_angle_constant = np.sin(angle_constant)

        Effect = Coefficient_for_combination / min_angle_velocity_combination * Sin_angle_constant

        if(Effect > max_Effect):
            min_angle_velocity_combination_all = min_angle_velocity_combination
            Coefficient_for_optimal_result = Coefficient_for_combination
            Sin_coefficient_for_optimal_result = Sin_angle_constant
            max_Effect = Effect
            optimal_angle_velocity = angle_velocity
            optimal_angle = angle
            optimal_action = action
            nquanta_for_combination_optimal = nquanta_for_combination

    print('min_angle_velocity_combination:  ' +  str(min_angle_velocity_combination_all) )
    print('Sin coefficient : ' + str(Sin_coefficient_for_optimal_result))
    print('Coefficient for optimal result: ' +str(Coefficient_for_optimal_result))
    print('optimal angle velocity  ' +  str(optimal_angle_velocity) )
    print('optimal action:  ' )
    print(optimal_action)
    print('optimal angle:  ')
    print(optimal_angle)
    print('nquanta for combination   ' + str(nquanta_for_combination_optimal))

def compute_min_abs_angle_velocity_combination(angle_velocity, nquanta_list,Coefficient):
    dof = len(angle_velocity)

    angle_velocity_combine_list_all = []
    Coefficient_for_combination_list = []
    nquanta_for_combination_list = []
    for i in range(len(nquanta_list)):
        if(abs(Coefficient[i]) < 10):
            continue

        nquanta = nquanta_list[i]
        index = 0
        nquanta_combine = [nquanta]
        for j in range(dof):
            if nquanta[j] == 0:
                continue
            else:
                nquanta_combine_new = []
                for old_quanta in nquanta_combine:
                    new_quanta = np.copy(old_quanta)
                    new_quanta[j] = - new_quanta[j]
                    nquanta_combine_new.append(new_quanta)

                # concate two list
                nquanta_combine = nquanta_combine + nquanta_combine_new

        angle_velocity_combine_list = []
        for quanta_element in nquanta_combine:
            angle_velocity_combine = np.sum(np.array(quanta_element) * np.array(angle_velocity))
            angle_velocity_combine_list.append(angle_velocity_combine)
            Coefficient_for_combination_list.append(Coefficient[i] )
            nquanta_for_combination_list.append(quanta_element)

        angle_velocity_combine_list_all = angle_velocity_combine_list_all + angle_velocity_combine_list

    angle_velocity_combine_list_all = np.abs(np.array(angle_velocity_combine_list_all))

    min_angle_velocity_combination_index = np.argmin(angle_velocity_combine_list_all)
    min_angle_velocity_combination = angle_velocity_combine_list_all[min_angle_velocity_combination_index]
    Coefficient_for_combination = Coefficient_for_combination_list[ min_angle_velocity_combination_index  ]
    nquanta_for_combination = nquanta_for_combination_list[min_angle_velocity_combination_index]

    return min_angle_velocity_combination , Coefficient_for_combination, nquanta_for_combination




