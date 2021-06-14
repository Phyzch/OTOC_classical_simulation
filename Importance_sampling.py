# This code use importance sampling to sample around small region that have largest Lyapunov exponent.
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from Fast_Lyapunov_indicator import Compute_fast_Lyapunov_indicator
from SCCL2_potential import Read_Realistic_SCCL2
from Evolve_dynamics import SCCL2_Realistic_Hamiltonian

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()


def Read_chaotic_center_action_and_angle(folder_path):
    file_name = "all_action_angle_lyapunov.txt"
    file_path = os.path.join(folder_path, file_name)

    with open(file_path) as f:
        data = f.read().splitlines()

        line = data[0].strip('\n')
        line = re.split(' ', line)
        center_number = int(line[0])

        line_index = 1
        lyapunov_exponent_list = []
        position_list = []
        for i in range(center_number):
            line = data[line_index].strip('\n')
            line = re.split(' ' , line)
            line = [i for i in line if i!='']
            lyapunov_exponent = float(line[0])
            lyapunov_exponent_list.append(lyapunov_exponent)

            line_index = line_index + 1
            line = data[line_index].strip('\n')
            line = re.split(' ' , line)
            line = [i for i in line if i!='']
            position = [float(i) for i in line]
            position_list.append(position)

            line_index = line_index + 1

        lyapunov_exponent_list = np.array(lyapunov_exponent_list)
        position_list = np.array(position_list)

        maximum_lyapunov_exponent = np.max(lyapunov_exponent_list)
        sort_index = np.argsort(lyapunov_exponent_list)
        lyapunov_exponent_list_sorted = lyapunov_exponent_list[sort_index]
        position_list_sorted = position_list[sort_index]

        start_index = 0
        for i in range(len(lyapunov_exponent_list_sorted)):
            if(lyapunov_exponent_list_sorted[i] < maximum_lyapunov_exponent / 3):
                start_index = start_index + 1
            else:
                break

        lyapunov_exponent_list_sorted = lyapunov_exponent_list_sorted[start_index : ]
        position_list_sorted = position_list_sorted[start_index : ]

        center_number = len(lyapunov_exponent_list_sorted)

        return center_number , lyapunov_exponent_list_sorted, position_list_sorted

def compute_average_FLI( initial_position_list  ,Dynamical_function, dynamical_argument, Time_step):
    number = len(initial_position_list)
    dof = len(initial_position_list[0])
    initial_tangent_vector = np.ones(dof) / np.sqrt(dof)
    FLI_list = []
    for i in range(number):
        _, _, FLI = Compute_fast_Lyapunov_indicator(initial_position_list[i] , Dynamical_function, Time_step, dynamical_argument, initial_tangent_vector)
        FLI_list.append(FLI)

    average_FLI = np.mean(FLI_list)

    return average_FLI , FLI_list

def Compute_chaotic_center_radius( center_position,  criteria_max  , change_action_bool ,
                                   Dynamic_function, dynamic_argument , Time_step):
    '''
    compute radius for center position.
    :param center_position:
    :param lyapunov_exponent:
    :param criteria_max : criteria for radius cutoff
    :param change_action_bool:
    :return:
    '''
    dof = int( len(center_position) / 2 )
    radius = 2 * np.pi
    sample_number = 10

    initial_action = center_position[:dof]
    initial_angle = center_position[dof:]
    initial_tangent_vector = np.ones(dof * 2 ) / np.sqrt(dof * 2 )
    _, _, center_position_FLI = Compute_fast_Lyapunov_indicator(center_position, Dynamic_function, Time_step, dynamic_argument, initial_tangent_vector )

    while(1):
        position_list = []
        for i in range(sample_number):
            if(change_action_bool):
                random_number = (np.random.random(2 * dof)  - 0.5) * 2 * radius
                new_position = center_position + random_number
            else:
                random_number = (np.random.random(dof)  - 0.5) * 2 * radius
                new_angle = initial_angle + random_number
                new_position = np.concatenate( (initial_action, new_angle) )

            position_list.append(new_position)

        average_FLI , FLI_list= compute_average_FLI(position_list, Dynamic_function, dynamic_argument, Time_step)
        if(average_FLI < center_position_FLI * criteria_max ):
            radius = radius / 2
        else:
            print("center FLI : " + str(center_position_FLI))
            print("average FLI:  " + str(average_FLI))
            print("FLI list:  " + str(FLI_list))
            break

    return radius

def compute_center_position_and_radius_SCCL2(folder_path , change_action_bool ):
    center_number, lyapunov_exponent_list , position_list = Read_chaotic_center_action_and_angle(folder_path)

    frequency, Coefficient, nquanta_list = Read_Realistic_SCCL2()
    dynamic_argument = (frequency, Coefficient, nquanta_list)
    Dynamical_function = SCCL2_Realistic_Hamiltonian

    dof = 6
    final_time = 0.02
    Time_step_len = 100
    Time_step = np.linspace(0, final_time, Time_step_len)

    criteria_max = 0.5

    # code to compute radius  set radius to 0.5 according to simulation.

    # radius_list = []
    # for i in range(center_number):
    #     radius = Compute_chaotic_center_radius(position_list[i], criteria_max, change_action_bool, Dynamical_function,
    #                                            dynamic_argument, Time_step)
    #
    #     radius_list.append(radius)

    radius_list = np.ones(center_number) * 0.25

    print("radius for chaotic region:  ")
    print(radius_list)

    return center_number, lyapunov_exponent_list , position_list , radius_list

def importance_sampling_compute_Lyapunov_spectrum(folder_path):
    change_action_bool = False
    initial_action = [6 , 5 ,1, 3, 5 ,3 ]

    center_number, center_lyapunov_exponent_list , center_position_list , radius_list = compute_center_position_and_radius_SCCL2(folder_path, change_action_bool)

    dof = int ( len(center_position_list[0]) / 2 )
    Iteration_number = 20
    Iteration_number_per_core = Iteration_number / num_proc
    Iteration_number = Iteration_number_per_core * num_proc

    initial_position_list = []
    probability_list = []  # probability for corresponding new distribution.
    for i in range(Iteration_number_per_core):
        random_number = np.random.random()
        if(random_number < 1/2):
            even_sampling_in_phase_space = True
        else:
            even_sampling_in_phase_space = False

        if(even_sampling_in_phase_space):
            distance_bool = True
            while distance_bool == True :
                distance_bool = False
                # sample evenly in state space. fixme: Here we do not change action
                random_angle = np.random.random( dof ) * 2 * np.pi

                for index in range(center_number):
                    center_angle = np.array(center_position_list[index][dof : ])
                    angle_distance = np.abs ( random_angle - center_angle )
                    Bool_list = [angle_distance[i] > 3 * radius_list[index] ]
                    Bool = np.all(Bool_list)
                    # this angle is within region of chaotic center . we want 3 * radius cutoff for each center as indicated by gaussian error function
                    if(Bool == False):
                        distance_bool = True
                        break

            initial_position = np.concatenate( (initial_action , random_angle) )


        else:
            random_number = random_number - 0.5
            # random_number / ( 1 / (2 * center_number)). 1 / (2 * center_number) is prob for eawch chaotic region
            center_index = int ( random_number * 2 * center_number )
            if(center_index == center_number):
                center_index = center_number - 1

            center_position = center_position_list[center_index]
            radius = radius_list[center_index]
            center_angle = center_position[dof:]
            random_angle = []
            for j in range(dof):
                angle = np.random.normal(center_angle[j], radius, 1 )[0]
                random_angle.append(angle)

            initial_position = np.concatenate((initial_action, random_angle))
            # 1/ (2 * center_number)

        initial_position_list.append(initial_position)
        # probability_list.append(probability)
    


folder_path = "/home/phyzch/PycharmProjects/OTOC_classical simulation/result/Test FLI/SCCL2 651353/"
change_action_bool = False
# compute_center_position_and_radius_SCCL2(folder_path)
