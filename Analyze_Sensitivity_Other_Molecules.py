import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Evolve_dynamics import Evolve_dynamics_Other_Molecules , Evolve_dynamics_SCCL2_Realistic_Hamiltonian
from SCCL2_potential import Generate_n_quanta_list_for_SCCL2, Other_molecule_angle_velocity, Other_molecule_action_velocity, SCCL2_Realistic_Hamiltonian_action_velocity, SCCL2_Realistic_Hamiltonian_angle_velocity
from SCCL2_potential import Read_Realistic_SCCL2
from Evolve_back_in_time import Evolve_dynamics_SCCL2_Realistic_Hamiltonian_back_in_time
from Evolve_dynamics_Using_Bulirsch import Evolve_dynamics_Other_Molecule_BS_method
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()

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

    nquanta_list_trans = np.transpose(nquanta_list)

    sol = Evolve_dynamics_Other_Molecules(Initial_position,Time_step,V0,scaling_parameter,frequency,f0,nquanta_list , nquanta_list_trans)

    Period = 0.03

    Sol_diff_list = []
    phase_jitter = 0.001

    nquanta_list_trans = np.transpose(nquanta_list)

    for i in range(dof):
        phase_change = np.zeros(dof)
        phase_change[i] = phase_jitter

        Initial_angle2 = np.array(Initial_angle1) + np.array(phase_change)
        Initial_angle2 = Initial_angle2.tolist()

        Initial_position = Initial_action + Initial_angle2

        sol1 = Evolve_dynamics_Other_Molecules(Initial_position,Time_step,V0,scaling_parameter,frequency,f0,nquanta_list, nquanta_list_trans)

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


def Other_molecules_Analyze_Stability_Matrix_for_xp(folder_path):
    matplotlib.rcParams.update({'font.size': 14})

    # This term tune the chaos
    V0 = 3050

    scaling_parameter = 0.2

    # Cyclopentane
    # frequency = [3062, 2914, 1445, 1349, 1275, 1203, 1043, 905, 695]
    # dof = 9
    # Initial_action = [2, 2, 2, 3, 2, 2, 2, 2,2]
    # Initial_angle_for_largest_eigenvalue = np.zeros([dof]).tolist()

    # Cyclopentaone
    frequency = [2210 , 2222 , 2966 , 2945 , 2130 , 2126 , 2880 , 2880]
    dof = 8
    Initial_action = [2 ,2 ,3 ,2 ,2 ,2 ,2 ,2]
    Initial_angle_for_largest_eigenvalue = np.zeros([dof]).tolist()

    f0 = 500

    nquanta_list = Generate_n_quanta_list_for_SCCL2(dof)

    final_time = 0.005

    Time_step_len = 100

    Time_step = np.linspace(0, final_time, Time_step_len)

    Iterate_number = 20

    Largest_Eigenvalue_List = []
    Largest_Singularvalue_List = []
    Period = 0.03

    Eigenvalue_List_in_all_simulation = []
    Singularvalue_List_in_all_simulation = []
    random_angle_list = []

    Iteration_number_per_core = int(Iterate_number / num_proc)
    Iterate_number = Iteration_number_per_core * num_proc

    nquanta_list_trans = np.transpose(nquanta_list)


    for iter_index in range(Iteration_number_per_core):
        Initial_angle = [np.random.random() * np.pi * 2 for i in range(dof)]
        # Initial_angle = [4.87371406040547, 6.20316183022866, 4.093689463478892, 3.8541398433268963, 0.23028078649982597, 0.8565145819947704, 5.365277076121005, 2.9484079410970825]

        random_angle_list.append(Initial_angle)
       # print('initial action')
        # print(Initial_action)
        # print('initial angle:')
        # print(Initial_angle)

        Initial_position = Initial_action + Initial_angle

        # sol = Evolve_dynamics_Other_Molecules(Initial_position,Time_step,V0,scaling_parameter,frequency,f0,nquanta_list ,nquanta_list_trans)
        _, sol, finish_simulation = Evolve_dynamics_Other_Molecule_BS_method(Initial_position, Time_step, V0, scaling_parameter,
                                                             frequency, f0, nquanta_list, nquanta_list_trans)

        if(finish_simulation == False):
            print("Simulation failed with angle:  " + str(Initial_angle) +" and iter index :  " + str(iter_index) +"  for original dynamics" )
        sol_len = len(sol)
        if(sol_len < Time_step_len):
            zero_array = np.zeros( (Time_step_len - sol_len , 2 * dof) )
            sol = np.concatenate((sol, zero_array), axis = 0 )

        Sol_change_list = []   # list of trajectory after impose a phase or action jitter
        action_jitter = 0.001

        for i in range(dof):
            action_change = np.zeros(dof)
            action_change[i] = action_jitter

            Initial_action1 = np.array(Initial_action) + np.array(action_change)
            Initial_action1 = Initial_action1.tolist()

            Initial_position = Initial_action1 + Initial_angle

            # sol1 = Evolve_dynamics_Other_Molecules(Initial_position, Time_step, V0, scaling_parameter, frequency, f0,
            #                                       nquanta_list, nquanta_list_trans)

            _, sol1 , finish_simulation = Evolve_dynamics_Other_Molecule_BS_method(Initial_position, Time_step, V0, scaling_parameter,
                                                              frequency, f0, nquanta_list, nquanta_list_trans)
            if (finish_simulation == False):
                print("Simulation failed with angle:  " + str(Initial_angle) + " and iter index :  " + str(
                    iter_index) + "  for action perturbation for mode "+ str(i))

            sol_len = len(sol1)
            if (sol_len < Time_step_len):
                zero_array = np.zeros((Time_step_len - sol_len, 2 * dof))
                sol1 = np.concatenate((sol1, zero_array), axis=0)

            Sol_change_list.append(sol1)

        phase_jitter = 0.001
        for i in range(dof):
            phase_change = np.zeros(dof)
            phase_change[i] = phase_jitter

            Initial_angle1 = np.array(Initial_angle) + np.array(phase_change)
            Initial_angle1 = Initial_angle1.tolist()

            Initial_position = Initial_action + Initial_angle1

            # sol1 = Evolve_dynamics_Other_Molecules(Initial_position, Time_step, V0, scaling_parameter, frequency, f0,
            #                                       nquanta_list, nquanta_list_trans)

            _, sol1, finish_simulation = Evolve_dynamics_Other_Molecule_BS_method(Initial_position, Time_step, V0, scaling_parameter,
                                                              frequency, f0, nquanta_list, nquanta_list_trans)
            if (finish_simulation == False):
                print("Simulation failed with angle:  " + str(Initial_angle) + " and iter index :  " + str(
                    iter_index) + "  for angle perturbation for mode "+ str(i))

            sol_len = len(sol1)
            if (sol_len < Time_step_len):
                zero_array = np.zeros((Time_step_len - sol_len, 2 * dof))
                sol1 = np.concatenate((sol1, zero_array), axis=0)

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
                if(J >= 0):
                    Q = np.sqrt(2*J) * np.cos(phi)
                    P = np.sqrt(2*J) * np.sin(phi)
                else:
                    Q = 0
                    P = 0

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

                    if (J >= 0):
                        Q = np.sqrt(2 * J) * np.cos(phi)
                        P = np.sqrt(2 * J) * np.sin(phi)
                    else:
                        Q = 0
                        P = 0

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
                    if(Initial_action[j] != 0):
                    # {Q_{i}(t) ,Q_{j}}
                        Stability_Matrix[i][j] = (Diff_XP_matrix_list[j][i][t] / action_jitter * np.sqrt(2 * Initial_action[j]) * np.sin(Initial_angle[j]) +
                                                  Diff_XP_matrix_list[j+dof][i][t]/phase_jitter * np.cos(Initial_angle[j]) / np.sqrt(2 * Initial_action[j])  )   # {, Q_{j} }
                        # {Q_{i}(t) , P_{j}}
                        Stability_Matrix[i][j+dof] =  (Diff_XP_matrix_list[j][i][t] / action_jitter *  np.sqrt(2*Initial_action[j]) * np.cos(Initial_angle[j]) -
                                                  Diff_XP_matrix_list[j+dof][i][t]/phase_jitter * np.sin(Initial_angle[j]) / np.sqrt(2*Initial_action[j]) )      # { , P_{j}}

                    else:
                        Stability_Matrix[i][j] = 0
                        Stability_Matrix[i][j+dof] = 0

            Stability_Matrix_list.append(Stability_Matrix)

        Singular_value_list = []
        Eigen_value_list = []
        for t in range(Time_step_len):
            u, s, vh = np.linalg.svd(Stability_Matrix_list[t])
            s = -np.sort(-s)
            # do an upper cutoff as pow(10,6)
            for j in range(2 * dof):
                    if s[j] > pow(10,6):
                        s[j] = pow(10, 6)

            Singular_value_list.append(s)
            eigenvalue = np.power(s,2)
            Eigen_value_list.append(eigenvalue)

        # Now we have Singular value at different time
        Singular_value_list = np.transpose(Singular_value_list)
        Eigen_value_list = np.transpose(Eigen_value_list)

        Largest_Lyapunov_exponent = np.log(Eigen_value_list[0][-1]) / (2*Time_step[-1])

        Eigenvalue_List_in_all_simulation.append(Eigen_value_list)
        Singularvalue_List_in_all_simulation.append(Singular_value_list)

    # combine results in different process together:
    Eigenvalue_List_in_all_simulation = np.real(Eigenvalue_List_in_all_simulation)
    Singularvalue_List_in_all_simulation = np.real(Singularvalue_List_in_all_simulation)
    random_angle_list = np.real(random_angle_list)

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
    if (rank == 0):
        recv_angle_list = np.empty([num_proc, Iteration_number_per_core, dof], dtype=np.float64)

    comm.Gather(random_angle_list, recv_angle_list, 0)


    if rank == 0 :
        # convert recved data to original format
        # Now shape [iterate_number , 2 * dof,  Time_step_len ]
        recv_Eigenvalue_list_shape = recv_Eigenvalue_list_in_all_simulation.shape
        Eigenvalue_List_in_all_simulation = np.reshape(recv_Eigenvalue_list_in_all_simulation,
                                                       ( recv_Eigenvalue_list_shape[0] * recv_Eigenvalue_list_shape[1],
                                                         recv_Eigenvalue_list_shape[2], recv_Eigenvalue_list_shape[3])  )

        recv_Singular_value_shape = recv_Singular_value_List_in_all_simulation.shape
        Singularvalue_List_in_all_simulation = np.reshape(recv_Singular_value_List_in_all_simulation,
                                                          (recv_Singular_value_shape[0] * recv_Eigenvalue_list_shape[1] ,
                                                           recv_Singular_value_shape[2] , recv_Eigenvalue_list_shape[3])
                                                          )
        # Now shape : [Iterate_number , dof ]
        recv_angle_list_shape = recv_angle_list.shape
        random_angle_list = np.reshape(recv_angle_list,
                                       (recv_angle_list_shape[0] * recv_angle_list_shape[1] , recv_angle_list_shape[2])
                                       )

        # -----Now we have to cut the array because we fill in 0 for time where results below up. --------
        min_nonzero_len = 100000
        for i in range(Iterate_number):
            nonzero_len = len ( [ i for i in  Eigenvalue_List_in_all_simulation[i][0] if i!=0 ])
            if(min_nonzero_len > nonzero_len):
                min_nonzero_len = nonzero_len

        Eigenvalue_List_in_all_simulation = Eigenvalue_List_in_all_simulation[:,:,:min_nonzero_len]
        Singularvalue_List_in_all_simulation = Singularvalue_List_in_all_simulation[:, : ,:min_nonzero_len ]

        Time_step = Time_step [:min_nonzero_len]

        # -----------------------------------------------------------

        # Now compute Largest Singular_value_list and Largest_eigenvalue_list and Largest Lyapunov_exponent_list and their initial angles.
        Largest_Lypunov_exponent_in_all_simulation = 0
        for i in range(Iterate_number):
            Largest_Lyapunov_exponent = np.log(Eigenvalue_List_in_all_simulation[i][0][-1]) / (2 * Time_step[-1])
            if (Largest_Lyapunov_exponent > Largest_Lypunov_exponent_in_all_simulation):
                Largest_Lypunov_exponent_in_all_simulation = Largest_Lyapunov_exponent
                Initial_angle_for_largest_eigenvalue = random_angle_list[i]
                Largest_Eigenvalue_List = Eigenvalue_List_in_all_simulation[i]
                Largest_Singularvalue_List = Singularvalue_List_in_all_simulation[i]

        Singular_value_list = Largest_Singularvalue_List
        Eigen_value_list = Largest_Eigenvalue_List

        print('initial action')
        print(Initial_action)
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
            ax1.plot(np.array(Time_step)/Period, Singular_value_list[i],label = 'Singular value '+ str(i+1) + ' for M' )
        ax1.plot(np.array(Time_step) / Period, Singular_value_list[0], label='Largest Singular value for M')

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

        # for i in range( 2*dof):
        #     ax3.plot(np.array(Time_step) / Period, Average_Eigenvalue_list[i], label='Average Eigenvalue ' + str(i+1)+" for $M^{2}$ ")

        ax3.plot(np.array(Time_step) / Period, Average_Eigenvalue_list[0],
                 label='Largest Average Eigenvalue  for $M^{2}$ ')

        ax3.set_xlabel('t/T')
        ax3.set_ylabel('Average Eigenvalue')

        ax3.set_title('Average Eigenvalue')

        # ax3.legend(loc = 'best')
        ax3.set_yscale('log')

        # save the result
        file_path = os.path.join(folder_path, "Average_Eigenvalue_for_chaotic_regime.txt")
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

        plt.show()


def Analyze_OTOC_for_xp_for_Realistic_SCCL2_Hamiltonian(folder_path):
    matplotlib.rcParams.update({'font.size': 14})
    frequency, Coefficient, nquanta_list = Read_Realistic_SCCL2()

    Coefficient = np.array(Coefficient) * 1

    dof = 6
    final_time = 0.05

    Period = 0.03

    Time_step_len = 10

    Time_step = np.linspace(0, final_time, Time_step_len)

    Iterate_number = 1

    # Initial_action =  [6.2187, 5.5134, 1.0357, 3.2284, 4.9875, 2.896]

    Initial_action = [6 ,5 ,1 ,3 , 5 ,3 ]

    Eigenvalue_List_in_all_simulation = []
    Singularvalue_List_in_all_simulation = []
    Largest_Lypunov_exponent_in_all_simulation = 0

    Initial_angle = [31.94, 6.098, 8.846, 16.529, 20.217,
                     14.1]

    Initial_angle_for_largest_eigenvalue = []
    Largest_Eigenvalue_List= []
    Largest_Singularvalue_List = []
    random_angle_list = []

    Iteration_number_per_core = int(Iterate_number / num_proc)
    Iterate_number = Iteration_number_per_core * num_proc

    for l in range(Iteration_number_per_core):

        if(l != 0 ):
            # Initial_angle_new = [ ((np.random.random() -0.5) * 2 * 0.1 + 1) * angle for angle in Initial_angle ]
            Initial_angle_new = [ 2 * np.pi * np.random.random() for i in range(dof) ]
            Initial_angle = Initial_angle_new

        random_angle_list.append(Initial_angle)

        Initial_position = Initial_action + Initial_angle

        sol = Evolve_dynamics_SCCL2_Realistic_Hamiltonian(Initial_position,Time_step,frequency,Coefficient,nquanta_list)

        Sol_change_list = []  # list of trajectory after impose a phase or action jitter
        action_jitter = 0.001

        for i in range(dof):
            action_change = np.zeros(dof)
            action_change[i] = action_jitter

            Initial_action1 = np.array(Initial_action) + np.array(action_change)
            Initial_action1 = Initial_action1.tolist()

            Initial_position = Initial_action1 + Initial_angle

            sol1 = Evolve_dynamics_SCCL2_Realistic_Hamiltonian(Initial_position,Time_step,frequency,Coefficient,nquanta_list)
            Sol_change_list.append(sol1)

        phase_jitter = 0.001
        for i in range(dof):
            phase_change = np.zeros(dof)
            phase_change[i] = phase_jitter

            Initial_angle1 = np.array(Initial_angle) + np.array(phase_change)
            Initial_angle1 = Initial_angle1.tolist()

            Initial_position = Initial_action + Initial_angle1

            sol1 =  Evolve_dynamics_SCCL2_Realistic_Hamiltonian(Initial_position,Time_step,frequency,Coefficient,nquanta_list)

            Sol_change_list.append(sol1)

        # we first compute \partial Q_{i} / \partial J_{j} or \partial Q_{i} / \partial theta_{j}

        # for original dynamics
        XP_matrix = []
        for i in range(2 * dof):
            XP_matrix.append([])

        for t in range(Time_step_len):
            for i in range(dof):
                J = sol[t][i]
                phi = sol[t][i + dof]

                Q = np.sqrt(2 * J) * np.cos(phi)
                P = np.sqrt(2 * J) * np.sin(phi)

                XP_matrix[i].append(Q)
                XP_matrix[i + dof].append(P)
        XP_matrix = np.array(XP_matrix)

        Diff_XP_matrix_list = []  # change of XP matrix after we change action/ angle  :  Delta Q_{i}. or Delta P_{i}
        for i in range(2 * dof):
            XP_matrix_new = []
            for j in range(2 * dof):
                XP_matrix_new.append([])

            Sol_after_change = Sol_change_list[i]

            for t in range(Time_step_len):
                for j in range(dof):
                    J = Sol_after_change[t][j]
                    phi = Sol_after_change[t][j + dof]

                    Q = np.sqrt(2 * J) * np.cos(phi)
                    P = np.sqrt(2 * J) * np.sin(phi)

                    XP_matrix_new[j].append(Q)
                    XP_matrix_new[j + dof].append(P)

            XP_matrix_new = np.array(XP_matrix_new)
            Diff_XP_matrix = XP_matrix_new - XP_matrix

            Diff_XP_matrix_list.append(Diff_XP_matrix)

        #  Diff_XP_matrix_list: size [2*dof, 2*dof, Time_len]

        Stability_Matrix_list = []  # Stability_Matrix at different time step

        for t in range(Time_step_len):
            Stability_Matrix = np.zeros([2 * dof, 2 * dof])

            for i in range(2 * dof):
                for j in range(dof):
                    # {Q_{i}(t) ,Q_{j}}
                    Stability_Matrix[i][j] = (
                                Diff_XP_matrix_list[j][i][t] / action_jitter * np.sqrt(2 * Initial_action[j]) * np.sin(
                            Initial_angle[j]) +
                                Diff_XP_matrix_list[j + dof][i][t] / phase_jitter * np.cos(Initial_angle[j]) / np.sqrt(
                            2 * Initial_action[j]))  # {, Q_{j} }
                    # {Q_{i}(t) , P_{j}}
                    Stability_Matrix[i][j + dof] = (
                                Diff_XP_matrix_list[j][i][t] / action_jitter * np.sqrt(2 * Initial_action[j]) * np.cos(
                            Initial_angle[j]) -
                                Diff_XP_matrix_list[j + dof][i][t] / phase_jitter * np.sin(Initial_angle[j]) / np.sqrt(
                            2 * Initial_action[j]))  # { , P_{j}}

            Stability_Matrix_list.append(Stability_Matrix)

        Singular_value_list = []
        Eigen_value_list = []
        for t in range(Time_step_len):
            u, s, vh = np.linalg.svd(Stability_Matrix_list[t])
            s = -np.sort(-s)
            Singular_value_list.append(s)
            eigenvalue = np.power(s, 2)
            Eigen_value_list.append(eigenvalue)

        # Now we have Singular value at different time
        Singular_value_list = np.transpose(Singular_value_list)
        Eigen_value_list = np.transpose(Eigen_value_list)

        Eigenvalue_List_in_all_simulation.append(Eigen_value_list)
        Singularvalue_List_in_all_simulation.append(Singular_value_list)



    # combine results in different process together:
    Eigenvalue_List_in_all_simulation = np.real(Eigenvalue_List_in_all_simulation)
    Singularvalue_List_in_all_simulation = np.real(Singularvalue_List_in_all_simulation)
    random_angle_list = np.real(random_angle_list)

    # size: [num_proc, iteration_number_per_core, 2*dof, Time_step_len]
    recv_Eigenvalue_list_in_all_simulation = []
    recv_Singular_value_List_in_all_simulation = []
    if(rank == 0):
        recv_Eigenvalue_list_in_all_simulation = np.empty([num_proc, Iteration_number_per_core, 2 * dof, Time_step_len] , dtype = np.float64)
        recv_Singular_value_List_in_all_simulation = np.empty([num_proc, Iteration_number_per_core, 2*dof, Time_step_len] , dtype = np.float64)

    comm.Gather(Eigenvalue_List_in_all_simulation, recv_Eigenvalue_list_in_all_simulation, 0)
    comm.Gather(Singularvalue_List_in_all_simulation , recv_Singular_value_List_in_all_simulation, 0)

    # angle_list : size : [num_proc, iteration_number_per_core, dof]
    recv_angle_list = []
    if(rank == 0):
        recv_angle_list = np.empty([num_proc, Iteration_number_per_core , dof] ,dtype = np.float64)

    comm.Gather(random_angle_list, recv_angle_list , 0)

    if (rank == 0):
        # convert recved data to original format
        # Now shape [iterate_number , 2 * dof,  Time_step_len ]
        recv_Eigenvalue_list_shape = recv_Eigenvalue_list_in_all_simulation.shape
        Eigenvalue_List_in_all_simulation = np.reshape(recv_Eigenvalue_list_in_all_simulation,
                                                       ( recv_Eigenvalue_list_shape[0] * recv_Eigenvalue_list_shape[1],
                                                         recv_Eigenvalue_list_shape[2], recv_Eigenvalue_list_shape[3])  )

        recv_Singular_value_shape = recv_Singular_value_List_in_all_simulation.shape
        Singularvalue_List_in_all_simulation = np.reshape(recv_Singular_value_List_in_all_simulation,
                                                          (recv_Singular_value_shape[0] * recv_Eigenvalue_list_shape[1] ,
                                                           recv_Singular_value_shape[2] , recv_Eigenvalue_list_shape[3])
                                                          )
        # Now shape : [Iterate_number , dof ]
        recv_angle_list_shape = recv_angle_list.shape
        random_angle_list = np.reshape(recv_angle_list,
                                       (recv_angle_list_shape[0] * recv_angle_list_shape[1] , recv_angle_list_shape[2])
                                       )

        # Now compute Largest Singular_value_list and Largest_eigenvalue_list and Largest Lyapunov_exponent_list and their initial angles.
        index_for_largest_lypunov = 0
        Largest_Lypunov_exponent_in_all_simulation = 0
        for i in range(Iterate_number):
            Largest_Lyapunov_exponent = np.log(Eigenvalue_List_in_all_simulation[i][0][-1]) / (2 * Time_step[-1])
            if(Largest_Lyapunov_exponent > Largest_Lypunov_exponent_in_all_simulation):
                Largest_Lypunov_exponent_in_all_simulation = Largest_Lyapunov_exponent
                Initial_angle_for_largest_eigenvalue = random_angle_list[i]
                Largest_Eigenvalue_List = Eigenvalue_List_in_all_simulation[i]
                Largest_Singularvalue_List = Singularvalue_List_in_all_simulation[i]

        Singular_value_list = Largest_Singularvalue_List
        Eigen_value_list = Largest_Eigenvalue_List

        print('initial action')
        print(Initial_action)
        Initial_angle = Initial_angle_for_largest_eigenvalue
        print('initial angle:')
        print(Initial_angle)

        # plot largest eigenvalue
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i in range(2 * dof):
            ax.plot(np.array(Time_step) / Period, Eigen_value_list[i],
                    label='Eigenvalue ' + str(i + 1) + " for $M^{2}$ ")

        ax.legend(loc='best')
        ax.set_xlabel('t/T')

        # ax.set_xscale('log')
        ax.set_yscale('log')

        # plot largest singlar value
        fig1, ax1 = plt.subplots(nrows=1, ncols=1)
        for i in range(2 * dof):
            ax1.plot(np.array(Time_step) / Period, Singular_value_list[i],
                     label='Singular value ' + str(i + 1) + ' for M')
        ax1.legend(loc='best')
        ax1.set_xlabel('t/T')

        # ax1.set_xscale('log')
        ax1.set_yscale('log')

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



        # save the result
        file_path = os.path.join(folder_path, "Average_Eigenvalue_for_chaotic_regime.txt")
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



def Plot_Trajectory_Other_Molecules():
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
    # Initial_action = [2, 2, 2, 3, 2, 2, 2, 2, 2]
    Initial_action = [0, 0, 0, 1, 0, 0, 1, 0, 0]

    Initial_angle = [2.4442179158763073, 1.0017940119372089, 4.250017300134825, 2.5048770218889747, 0.5626141486498147,
                     3.158573153008315, 0.7225434637992191, 6.038088800623476, 4.530226217933312]

    Initial_position = Initial_action + Initial_angle

    nquanta_list_trans = np.transpose(nquanta_list)
    # solve dynamics
    sol = Evolve_dynamics_Other_Molecules(Initial_position, Time_step, V0, scaling_parameter, frequency, f0, nquanta_list, nquanta_list_trans)

    Period = 0.03

    fig2, ax2 = plt.subplots(nrows=1, ncols=1)

    for i in range(dof):
        ax2.plot(Time_step / Period, sol[:, i] , label=' J' + str(i + 1) + ' (t)')

    ax2.legend(loc='best')
    ax2.set_xlabel('t/T')
    ax2.set_yscale('log')
    # ax2[1].set_yscale('log')

    # compute angle velocity at given time
    # fig3 , ax3 = plt.subplots(nrows=2,ncols=1)
    # action_t_list = [ sol[:, i ] for i in range(dof) ]
    # angle_t_list = [sol[:,i+dof] for i in range(dof)]
    #
    # # data in the form [Time_step, dof]
    # action_t_list = np.transpose(action_t_list)
    # angle_t_list = np.transpose(angle_t_list)
    #
    # angle_velocity_list = []
    # action_velocity_list = []
    # Len = len(Time_step)
    # for i in range(Len):
    #     action_t = action_t_list[i]
    #     angle_t = angle_t_list[i]
    #     angle_velocity = Other_molecule_angle_velocity(action_t,angle_t,V0,scaling_parameter,frequency,f0,nquanta_list , nquanta_list_trans)
    #     action_velocity = Other_molecule_action_velocity(action_t,angle_t, V0, scaling_parameter, frequency, f0, nquanta_list, nquanta_list_trans)
    #
    #     angle_velocity_list.append(angle_velocity)
    #     action_velocity_list.append(action_velocity)
    #
    # angle_velocity_list = np.transpose(angle_velocity_list)
    # action_velocity_list = np.transpose(action_velocity_list)
    #
    # for i in range(dof):
    #     ax3[0].plot(Time_step / Period, action_velocity_list[i] * cf , label = 'action velocity ' + str(i+1))
    #     ax3[1].plot(Time_step/ Period, angle_velocity_list[i] * cf, label = 'angle velocity  ' + str(i+1) )
    #
    # ax3[0].legend(loc='best')
    # ax3[1].legend(loc='best')
    # ax3[0].set_xlabel('t/T')
    # ax3[1].set_xlabel('t/T')
    #
    # ax3[0].set_ylim([-400,400])
    # ax3[1].set_ylim([-400,400])



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

    Initial_action = [6.2187, 5.5134, 1.0357, 3.2284, 4.9875, 2.896]

    # Initial_action1 = [2, 2, 3, 3, 3 + action_jitter, 2]

    max_action_in_all_simulation = 0
    Index = -1 # Index for max action angle.
    max_sol = []
    max_Sol_diff = []
    Initial_angle_list = []


    for i in range(Iteration_number):
        Initial_angle = [np.random.random() * np.pi * 2 for i in range(dof)]

        # Initial_angle = [30.93947782165516, 6.097124720812282, 8.845719425253785, 16.528919283844296, 20.21703311486376, 14.099748439492283]

        Initial_angle = [31.93947782165516, 6.097124720812282, 8.845719425253785, 16.528919283844296, 20.21703311486376, 14.099748439492283]
        # for j in range(dof):
        #     Initial_angle[j] = (0.02 * (np.random.random()-0.5) * 2 +1) * Initial_angle[j]

        Initial_angle_list.append(Initial_angle)
        print(Initial_angle)
        # print('initial angle:' + str(Initial_angle1))

        Initial_position = Initial_action + Initial_angle

        # solve dynamics
        sol = Evolve_dynamics_SCCL2_Realistic_Hamiltonian(Initial_position,Time_step,frequency,Coefficient,nquanta_list)

        # sol = Evolve_dynamics_SCCL2_Realistic_Hamiltonian_back_in_time(Initial_position,Time_step,frequency, Coefficient, nquanta_list)

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
        print(sol_transpose[i][-50:-10])

    for i in range(dof):
        print('angle  ' + str(i+1))
        print(sol_transpose[i + dof][-50:-10])

    # Energy_list = [  np.sum(np.array(sol[i][:dof] ) * np.array(frequency)) for i in range(len(sol)) ]
    # print('Energy:  ')
    # print(Energy_list)
    # print(min(Energy_list))

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

    nquanta_list_tras = np.transpose(nquanta_list)

    Iteration_number = 1000

    min_max_abs_angle_velocity_in_all_sample = 100000
    optimal_angle = []
    optimal_action = []
    optimal_angle_velocity = []

    for i in range(Iteration_number):
        angle = np.random.random(6) * 2 * np.pi
        action = np.random.random(6) * 6

        angle_velocity = Other_molecule_angle_velocity(action,angle,V0,scaling_parameter,frequency,f0,nquanta_list , nquanta_list_tras)
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




