# this code use Metropolis Hasting algorithm to find most chaotic region in phase space.
# We transform fast Lyapunov indicator into potential, and use MCMC to find most chaotic region.
from Fast_Lyapunov_indicator import Compute_fast_Lyapunov_indicator
from SCCL2_potential import Read_Realistic_SCCL2
from Evolve_dynamics import SCCL2_Realistic_Hamiltonian
from Evolve_dynamics_Using_Bulirsch import Evolve_dynamics_Realistic_SCCL2_BS_method , Evolve_dynamics_Realistic_SCCL2_BS_method_back_in_time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

def Compute_Potential_for_MCMC(initial_position, simulation_argument):

    time_step, Dynamical_function, dynamical_argument = simulation_argument
    frequency, Coefficient, nquanta_list = dynamical_argument

    dof = len(initial_position)

    # initial_tangent_vector = np.ones( dof) / np.sqrt( dof)
    #
    # _, _ , FLI = Compute_fast_Lyapunov_indicator(initial_position, Dynamical_function, time_step, dynamical_argument, initial_tangent_vector)

    nearby_trajectory_displs_strength = 0.01
    _, sol, _ = Evolve_dynamics_Realistic_SCCL2_BS_method( initial_position , time_step, frequency, Coefficient,
                                                          nquanta_list)
    trajectory = sol[-1]

    Norm_ratio_list = []
    repeat_num = 3
    for j in range(repeat_num):
        nearby_trajectory_displs = (np.random.random(dof) - 0.5) * 2 * nearby_trajectory_displs_strength
        old_Norm = np.linalg.norm(nearby_trajectory_displs)

        nearby_position = initial_position + nearby_trajectory_displs
        _, sol1, _ = Evolve_dynamics_Realistic_SCCL2_BS_method(nearby_position, time_step, frequency,
                                                               Coefficient, nquanta_list)
        nearby_trajectory = sol1[-1]

        new_Norm = np.linalg.norm(nearby_trajectory - trajectory)

        Norm_ratio = new_Norm / old_Norm
        Norm_ratio_list.append(Norm_ratio)

    Norm_ratio = np.max(Norm_ratio_list)
    potential = - Norm_ratio

    return potential , sol

def simulate_single_MCMC_trajectory(initial_position,  center_action , region_range ,
                                    simulation_argument, max_step_number, trial_time_max,
                                    jump_distance , temperature , change_action_bool):
    '''

    :param initial_position: initial position is the point we start jumping.
    :param center_action: center for MCMC, we will have distance cutoff given by region_range. Points outside region_range will not be sampled
    :param: region range: range to sample points for MCMC
    :param simulation_argument: (time_step, Dynamical_function, dynamical_argument) . used to compute FLI (or see it as - potential)
    :param max_step_number: max_step_number allowed in simulation
    :param trial_time_max : maximum time for failure to jump in MCMC
    :param jump_distance: jump distance control the distance we jump in phase space.
    :param temperature: temperature control the MCMC. if temperature is too high, we will go randomly in phase space. if temperature is too low, we will stuck in local minima
    :param: change action: if we change action during MCMC
    :return:
    '''
    matplotlib.rcParams.update({'font.size': 20})

    position = initial_position
    potential , trajectory = Compute_Potential_for_MCMC(position, simulation_argument)

    potential_list = [potential]
    position_list = [position]
    dof = int(len(initial_position) / 2)
    # the trial position is sampled from \eta(r,r') = exp(- (r-r')^2 / 2 \sigma^2 )
    step_number = 0
    trial_time = 0

    time_step, _, dynamical_argument = simulation_argument
    frequency, Coefficient, nquanta_list = dynamical_argument
    time_step_len = len(time_step)

    while step_number < max_step_number:

        if(change_action_bool):
            # change action and angle at same time
            jump_step = np.random.normal(0, jump_distance, 2 * dof)
            new_position = position + jump_step

            new_action = new_position[:dof]
            distance_from_center = np.max(np.abs(new_action - center_action))
            if(distance_from_center > region_range):
                continue

        else:
            # only change angle
            jump_step = np.random.normal(0, jump_distance, dof)
            action = position[:dof]
            angle = position[dof:]
            new_angle = angle + jump_step
            new_position = np.concatenate( (action, new_angle) )

        #shooting algorithm. choose middle point to perturb.
        # middle_time_step = int(np.random.random() * time_step_len)
        # middle_time = time_step[middle_time_step]
        # middle_point = trajectory[middle_time_step]
        # jump_step = np.random.normal(0, jump_distance, 2 * dof)
        # new_middle_point = middle_point + jump_step
        # time_step_back = np.linspace(0, middle_time, middle_time_step)
        # _,sol_back,_ = Evolve_dynamics_Realistic_SCCL2_BS_method_back_in_time(new_middle_point,time_step_back, frequency,
        #                                                                       Coefficient, nquanta_list)
        # new_position = sol_back[-1]

        for j in range(dof, 2 * dof):
            if (new_position[j] < 0):
                new_position[j] = new_position[j] + 2 * np.pi
            if (new_position[j] > 2 * np.pi):
                new_position[j] = new_position[j] - 2 * np.pi

        new_potential , trajectory1 = Compute_Potential_for_MCMC(new_position, simulation_argument)

        acceptance_prob = np.exp(-(new_potential - potential) /temperature )
        if(acceptance_prob >= 1):
            acceptance_prob = 1

        random_number = np.random.random()
        if(random_number < acceptance_prob):
            step_number = step_number + 1
            trial_time = 0
            position = np.copy(new_position)
            potential = new_potential
            trajectory = trajectory1

            position_list.append(position)
            potential_list.append(potential)
        else:
            trial_time = trial_time + 1

        if(trial_time == trial_time_max):
            print("try  " + str(trial_time) +" all fails. stop MCMC. check temperature or we arrive minimum" )
            break

    # find position with minimum potential
    potential_list = np.array(potential_list)
    position_list = np.array(position_list)
    potential_list_sort_index = np.argsort(potential_list)

    potential_list_sort = potential_list[potential_list_sort_index]
    position_list_sort = position_list[potential_list_sort_index]

    # print information about maximum lyapunov exponent found
    maximum_Lyapunov_exponent = - potential_list_sort[0]
    maximum_position = position_list_sort[0]

    # print('maximum lyapunov exponent found:  ' + str(maximum_Lyapunov_exponent))
    # print('action found: ' + str(maximum_position[:dof]))
    # print('angle found: ' + str(maximum_position[ dof : ]))

    # plot how potential change with step number to see if MCMC is successful
    # step = range(step_number + 1 )
    # fig,ax = plt.subplots(nrows=1, ncols=1)
    # ax.plot(step , potential_list )
    # ax.set_xlabel('step')
    # ax.set_ylabel('potential E')
    #

    position_list_trans = np.transpose(position_list)

    # fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    # for i in range(dof):
    #     ax1.plot(step, position_list_trans[i] , label = 'action  ' + str(i+1))
    # for i in range(dof):
    #     ax1.plot(step, position_list_trans[i + dof] , label = 'angle  ' + str(i+1))
    #
    # ax1.set_xlabel('step')
    # ax1.set_ylabel('trajecotry')
    # ax1.legend(loc = 'best')

    trajectory_length = step_number + 1

    return maximum_position, maximum_Lyapunov_exponent , trajectory_length , potential_list, position_list_trans

def MCMC_SCCL2(folder_path , Initial_action ):

    frequency, Coefficient, nquanta_list = Read_Realistic_SCCL2()
    dynamic_argument = (frequency, Coefficient, nquanta_list)
    Dynamical_function = SCCL2_Realistic_Hamiltonian

    dof = 6
    final_time = 0.02

    Time_step_len = 10
    Time_step = np.linspace(0, final_time, Time_step_len)

    simulation_argument = [ Time_step, Dynamical_function, dynamic_argument ]


    center_action = np.array(Initial_action)
    region_range = 1

    # Initial_angle = [5.64267210628619 , 15.599748227027739 , 5.735664470146787 , -3.0438284437721053 , 2.9316823714284084 , 8.920742399199833]
    # Initial_angle = [6.0252938 , 3.91642767 ,4.68152349, 3.77998479 ,2.9231359,  3.47953764]

    Initial_angle = np.random.random(dof) * 2 * np.pi
    Initial_angle = Initial_angle.tolist()

    max_step_number = 50
    trial_time_max = 20
    jump_distance = 0.1
    temperature = 0.15

    iteration_number = 10

    trajectory_length_list = []
    trajectory_potential_list = []
    maximum_position_list = []
    maximum_Lyapunov_exponent_list = []

    for iter_index in range(iteration_number):
        if(iter_index == 0):
            pass
        else:
            Initial_angle = np.random.random(dof) * 2 * np.pi
            Initial_angle = Initial_angle.tolist()

        Initial_position = Initial_action + Initial_angle
        maximum_position, maximum_Lyapunov_exponent ,  trajectory_length , potential_list, _ = simulate_single_MCMC_trajectory(Initial_position, center_action, region_range, simulation_argument,
                                                                                      max_step_number, trial_time_max, jump_distance, temperature,
                                                                                      change_action_bool= False  )
        maximum_Lyapunov_exponent_list.append(maximum_Lyapunov_exponent)
        maximum_position_list.append(maximum_position)
        trajectory_length_list.append(trajectory_length)
        trajectory_potential_list.append(potential_list)

    sort_index = np.argsort(-np.array(maximum_Lyapunov_exponent_list))
    print("best result is for trajectory: " + str(sort_index[0] + 1))
    print("best result: ")
    print(maximum_Lyapunov_exponent_list[sort_index[0]])
    print('action : ' + str(maximum_position_list[sort_index[0]][:dof] ))
    print ('angle : ' + str(maximum_position_list[sort_index[0]][dof:]) )

    # save result to the file
    best_result_output_file = os.path.join(folder_path , 'best_action_angle_lyapunov.txt')
    with open(best_result_output_file , "w") as f:
        f.write('best result : \n')
        f.write("Lyapunov exponent:  " + str(maximum_Lyapunov_exponent_list[sort_index[0]]) + " \n")
        f.write('action:  ')
        for i in range(dof):
            f.write(str(maximum_position_list[sort_index[0]][i] ) + " , ")
        f.write('\n')
        f.write("angle: ")
        for i in range(dof):
            f.write(str( maximum_position_list[sort_index[0]][i + dof ] ) + " , ")
        f.write("\n")

        f.write("\n")

    all_result_file = os.path.join(folder_path, "all_action_angle_lyapunov.txt")
    with open(all_result_file, "w") as f:
        f.write(str(iteration_number) + " \n ")
        for i in range(iteration_number):
            # Lyapunov exponent
            f.write(str(maximum_Lyapunov_exponent_list[i]) + " \n")
            # action and angle
            for j in range(dof):
                f.write(str(maximum_position_list[i][j]) + "  ")
            for j in range(dof):
                f.write(str( maximum_position_list[i][j+ dof] ) + "  ")
            f.write("\n")
            f.write("\n")



    potential_of_trajectory_output = os.path.join(folder_path , "potential_trajector.txt")
    with open(potential_of_trajectory_output, "w") as f:
        f.write( str(iteration_number) + "\n" )
        for i in range(iteration_number):
            for j in range(trajectory_length_list[i]):
                f.write(str(  np.around(trajectory_potential_list[i][j],3)  ) + " ")
            f.write("\n")

    # plot result
    fig, ax = plt.subplots(nrows=1, ncols = 1)
    for i in range(iteration_number):
        Len = trajectory_length_list[i]
        step = range(Len)
        ax.plot(step, trajectory_potential_list[i] , label = 'trajectory ' + str(i+1))
    ax.set_xlabel('step')
    ax.set_ylabel('potential')
    # ax.legend(loc = 'best')


    # get statistics for final result
    maximum_position_list_trans = np.transpose(maximum_position_list)
    angle_1_list = maximum_position_list_trans[dof]
    angle_2_list = maximum_position_list_trans[dof + 1]
    fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.scatter(angle_1_list, angle_2_list)
    ax1.set_xlabel('$\phi_{1}$')
    ax1.set_ylabel('$\phi_{2}$')

    plt.show()

    return maximum_Lyapunov_exponent_list, maximum_position_list

folder_path = "/home/phyzch/PycharmProjects/OTOC_classical simulation/result/Lyapunov weighted path/try/"

#Initial_action = [6, 5, 1, 3, 5, 3]
Initial_action = [3,3,3,2,2,2]

MCMC_SCCL2(folder_path, Initial_action)
