# This code computing fast Lyapunov indicator of trajectories given function for velocity of action J and angle \theta
# See https://link.springer.com/chapter/10.1007/978-3-662-48410-4_2 for introduction of Fast Lyapunov indicator.

import numpy as np
import math
import matplotlib.pyplot as plt
from Bulirsch_Stoer_module import bulStoer
from Evolve_dynamics import SCCL2_Realistic_Hamiltonian
from SCCL2_potential import Read_Realistic_SCCL2
import matplotlib.pyplot

def Dynamical_function_gradient( Dynamical_function, position, time, dynamical_argument):
    # get gradient of dynamical function by small perturbation.
    dof = len(position)
    perturbation = 0.001

    F = Dynamical_function(position, time, *dynamical_argument)
    # F_perturb = F(x+\delta x , t) - F(x,t)   F_perturb_list[i][j] = \partial F_{j} / \partial x_{i}
    F_perturb_list = []
    for i in range(dof):
        position_perturb = np.copy(position)
        position_perturb[i] = position_perturb[i] + perturbation
        F_perturb = Dynamical_function(position_perturb, time, *dynamical_argument)
        F_perturb = np.array(F_perturb - F) / perturbation
        F_perturb_list.append(F_perturb)

    F_perturb_list = np.transpose(F_perturb_list)

    # now  F_perturb_list[i][j] = \partial F_{i} / \partial x_{j}
    return F_perturb_list

def Compute_fast_Lyapunov_indicator(initial_position , Dynamic_function, Time_step, dynamic_argument , initial_tangent_vector):
    '''

    :param initial_position: Initial position in Arnold web
    :param Dynamic_function:  function to evolve Arnold web
    :param Time_step:  Time step to output Arnold web result
    :param dynamic_argument:  argument for Dynamic function
    :return:
    '''
    t0 = 0
    time_step_size = Time_step[1]
    final_time = Time_step[-1]
    initial_position = np.array(initial_position)
    tol = 1e-6

    time, position, _ = bulStoer(Dynamic_function, t0, initial_position, final_time, time_step_size, args = dynamic_argument , tol = tol )
    position = np.array(position)
    time = np.array(time)
    # position : [Time, dof]

    # we know dv/dt = \partial F /\partial x  * v
    # F(x) is velocity of field x, thus is given by Dynamic_function(position[time] , time, *dynamic_argument)
    # F_list is F(x) we should compute \partial F / \partial x for evolution of velocity
    time_len = len(time)

    tangent_vector_list = []
    tangent_vector = np.copy( np.array(initial_tangent_vector) )
    tangent_vector_list.append(tangent_vector)
    for i in range(time_len - 1):
        F_gradient = Dynamical_function_gradient(Dynamic_function, position[i], time[i], dynamic_argument)  # F_gradient [i][j] = \partial F_{i} / \partial x_{j}
        tangent_vector_velocity = np.sum(F_gradient * tangent_vector , 1)  # dv/dt =  \partial_x F * v
        delt = time[i+1] - time[i]
        tangent_vector_change =  delt * tangent_vector_velocity
        tangent_vector = tangent_vector + tangent_vector_change
        tangent_vector_copy = np.copy(tangent_vector)
        tangent_vector_list.append(tangent_vector_copy)

    FLI_candidate = [np.linalg.norm(tangent_vector) for tangent_vector in tangent_vector_list]
    FLI = np.max(FLI_candidate)

    return time, FLI_candidate , FLI

def Test_faast_Lyapunov_indicator():
    matplotlib.rcParams.update({'font.size': 20})
    dof = 6
    # more chaotic case:
    initial_action1 = [6, 5, 1, 3, 5, 3]
    initial_angle1 = [6.0252938,  3.91642767, 4.68152349, 3.77998479, 2.9231359,  3.47953764]
    # initial_angle1 = [ 2.03389453,  5.86846771 ,-0.57982631,  0.16723806,  2.08499875, -1.69735188]
    initial_position1 = initial_action1 + initial_angle1

    # less chaotic case:
    initial_action2 = [6 , 5 ,1, 3, 5 ,3 ]
    initial_angle2 = [5.55370451, 0.02425863, 5.04619891, 0.35696992, 1.91216829, 1.78723057]
    initial_position2 = initial_action2 + initial_angle2

    Dynamic_function = SCCL2_Realistic_Hamiltonian

    final_time = 0.02
    Time_step_len = 100
    Time_step = np.linspace(0, final_time, Time_step_len)

    frequency, Coefficient, nquanta_list = Read_Realistic_SCCL2()
    dynamic_argument = (frequency, Coefficient, nquanta_list)

    initial_tangent_vector = np.ones(2*dof) / np.sqrt(2*dof)

    time1, FLI_candidate1 , FLI1 = Compute_fast_Lyapunov_indicator(initial_position1, Dynamic_function, Time_step, dynamic_argument, initial_tangent_vector)

    time2 , FLI_candidate2  , FLI2 = Compute_fast_Lyapunov_indicator(initial_position2, Dynamic_function, Time_step, dynamic_argument, initial_tangent_vector)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(time1, FLI_candidate1 , label = 'more chaotic case')
    ax.plot(time2, FLI_candidate2 , label = 'less chaotic case')

    ax.legend(loc = 'best')
    ax.set_xlabel('t')
    ax.set_ylabel('FLI (t)')

    print('FLI for less chaotic case: ' + str(FLI2))
    print('FLI for more chaotic case: ' + str(FLI1))
    plt.show()

# Test_faast_Lyapunov_indicator()