import numpy as np
from scipy.integrate import odeint
from Potential import compute_angle_velocity, compute_action_velocity
from Potential import prepare_Tuple_list
import matplotlib.pyplot as plt
from Evolve_dynamics import Evolve_dynamics

cf = 2 * np.pi * 0.0299792458  # conversion from cm^{-1} to ps^{-1}

def Analyze_Sensitivity_number_operator():
    '''
    compute |\n_{i}(t) / \phi_{j}|^{2} matrix at early time
    :return:
    '''

    D = 32924

    # This term tune the chaos
    V_phi = 251.2

    dof = 6

    Tuple_list = prepare_Tuple_list(dof)

    frequency = [1003.1, 1003.5, 1002.9, 1002.4, 1003.8, 1001.1]  # in unit of cm^{-1}

    final_time = 0.019

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

    fig, ax = plt.subplots(nrows=1,ncols=1)
    for i in range(dof):
        ax.plot(np.array(Time_step) / Period ,Singular_value_list[i], label='singular value '+str(i))

    ax.legend(loc= 'best')
    ax.set_xlabel('t/T')

    # ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()

def Analyze_Stability_Matrix_change_action():
    D = 32924

    # This term tune the chaos
    V_phi = 251.2

    dof = 6

    Tuple_list = prepare_Tuple_list(dof)

    frequency = [1003.1, 1003.5, 1002.9, 1002.4, 1003.8, 1001.1]  # in unit of cm^{-1}

    final_time = 0.0015

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




def Analyze_Stability_Matrix_for_xp():
    D = 32924

    # This term tune the chaos
    V_phi = 251.5

    dof = 6

    Tuple_list = prepare_Tuple_list(dof)

    frequency = [1003.1, 1003.5, 1002.9, 1002.4, 1003.8, 1001.1]  # in unit of cm^{-1}

    final_time = 0.005

    Time_step = np.linspace(0, final_time, 500)

    Initial_action = [2, 2, 3, 3, 3, 2]
    Initial_angle = [np.random.random() * np.pi * 2 for i in range(dof)]

    print('initial action')
    print(Initial_action)
    print('initial angle:')
    print(Initial_angle)

    Initial_position = Initial_action + Initial_angle

    sol = Evolve_dynamics(Initial_position, Time_step, frequency, V_phi, D, Tuple_list)

    Period = 0.03

    Sol_change_list = []   # list of trajectory after impose a phase or action jitter
    action_jitter = 0.001

    for i in range(dof):
        action_change = np.zeros(dof)
        action_change[i] = action_jitter

        Initial_action1 = np.array(Initial_action) + np.array(action_change)
        Initial_action1 = Initial_action1.tolist()

        Initial_position = Initial_action1 + Initial_angle

        sol1 = Evolve_dynamics(Initial_position, Time_step, frequency, V_phi, D, Tuple_list)

        Sol_change_list.append(sol1)

    phase_jitter = 0.001
    for i in range(dof):
        phase_change = np.zeros(dof)
        phase_change[i] = phase_jitter

        Initial_angle1 = np.array(Initial_angle) + np.array(phase_change)
        Initial_angle1 = Initial_angle1.tolist()

        Initial_position = Initial_action + Initial_angle1

        sol1 = Evolve_dynamics(Initial_position, Time_step, frequency, V_phi, D, Tuple_list)

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

            Q = np.sqrt(J) * np.cos(phi)
            P = np.sqrt(J) * np.sin(phi)

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

                Q = np.sqrt(J) * np.cos(phi)
                P = np.sqrt(J) * np.sin(phi)

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
                Stability_Matrix[i][j] = (Diff_XP_matrix_list[j][i][t] / action_jitter * 2 * np.sqrt(Initial_action[j]) * np.sin(Initial_angle[j]) +
                                          Diff_XP_matrix_list[j+dof][i][t]/phase_jitter * np.cos(Initial_angle[j]) / np.sqrt(Initial_action[j])  )   # {, Q_{j} }
                # {Q_{i}(t) , P_{j}}
                Stability_Matrix[i][j+dof] =  (Diff_XP_matrix_list[j][i][t] / action_jitter * 2 * np.sqrt(Initial_action[j]) * np.cos(Initial_angle[j]) -
                                          Diff_XP_matrix_list[j+dof][i][t]/phase_jitter * np.sin(Initial_angle[j]) / np.sqrt(Initial_action[j]) )      # { , P_{j}}

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

    # plot result
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for i in range(dof):
        ax.plot(np.array(Time_step) / Period, Eigen_value_list[i], label='Eigenvalue ' + str(i)+" for $M^{2}$ ")

    ax.legend(loc='best')
    ax.set_xlabel('t/T')

    # ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()
