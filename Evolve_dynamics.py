import numpy as np
from scipy.integrate import odeint
from Potential import compute_angle_velocity, compute_action_velocity
from Potential import prepare_Tuple_list
import matplotlib.pyplot as plt
from SCCL2_potential import SCCL2_angle_velocity, SCCL2_action_velocity
from SCCL2_potential import SCCL2_Realistic_Hamiltonian_action_velocity, SCCL2_Realistic_Hamiltonian_angle_velocity

cf = 2 * np.pi * 0.0299792458  # conversion from cm^{-1} to ps^{-1}

def anharmonic_oscillator(y,t,frequency,V_phi,D,Tuple_list):
    '''

    :param y:  [action, angle] 2*dof array
    :param frequency: frequency omega of system
    :param V_phi:  strength of anharmonic coupling
    :param D:  Dissociation energy
    :param Tuple_list = [ [1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,1], [6,1,2] ] for dof. impose nearest neighbor coupling
    :return:
    '''
    dof = int (len(y) / 2)

    action = y[0:dof]
    angle = y[dof:2*dof]

    angle_velocity = compute_angle_velocity(frequency,action,angle,D,V_phi,Tuple_list)
    action_velocity = compute_action_velocity(V_phi,action,angle,Tuple_list)

    angle_velocity = angle_velocity.tolist()
    action_velocity = action_velocity.tolist()

    dydt = action_velocity + angle_velocity  # append two list together

    dydt = cf * np.array(dydt)

    return dydt


def Other_Molecules(y,t, V0, scaling_parameter ,frequency,f0, nquanta_list):
    '''

    :param y: [action, angle] 2*dof array
    :param t:  time t
    :param V0: coupling strength
    :param scaling_parameter: scaling parameter in coupling strength
    :param frequency: frequency of oscillator
    :param f0: mean frequency
    :param nquanta_list:  quanta_list used to generate angle and action velocity
    :return:
    '''
    dof = int(len(y) / 2)

    action = y[0:dof]
    angle = y[dof:dof*2]

    angle_velocity = SCCL2_angle_velocity(action,angle,V0,scaling_parameter,frequency,f0,nquanta_list)
    action_velocity = SCCL2_action_velocity(action,angle,V0,scaling_parameter,frequency,f0,nquanta_list)

    angle_velocity = angle_velocity.tolist()
    action_velocity = action_velocity.tolist()

    dydt = action_velocity + angle_velocity

    dydt = cf * np.array(dydt)

    return dydt

def SCCL2_Realistic_Hamiltonian(y,t,frequency, Coefficient, nquanta_list):
    dof = int(len(y) / 2)

    action = y[0:dof]
    angle = y[dof:dof * 2]

    angle_velocity = SCCL2_Realistic_Hamiltonian_angle_velocity(action,angle,frequency,Coefficient,nquanta_list)
    action_velocity = SCCL2_Realistic_Hamiltonian_action_velocity(action,angle,frequency,Coefficient,nquanta_list)

    angle_velocity = angle_velocity.tolist()
    action_velocity = action_velocity.tolist()

    dydt = action_velocity + angle_velocity

    dydt = cf * np.array(dydt)

    return dydt

def Evolve_dynamics(Initial_position, Time_step, frequency, V_phi, D, Tuple_list):
    '''
    :return: sol: [len(t), len(y0)], [:,0]: action 0. [:,1] : action1.  [:,6] : angle 0. ...
    '''
    sol = odeint(anharmonic_oscillator,Initial_position,Time_step, args=(frequency,V_phi,D,Tuple_list) )

    return sol

def Evolve_dynamics_Other_Molecules(Initial_position, Time_step,V0, scaling_parameter ,frequency,f0, nquanta_list ):

    sol = odeint(Other_Molecules,Initial_position,Time_step, args = (V0, scaling_parameter ,frequency,f0, nquanta_list) )

    return sol

def Evolve_dynamics_SCCL2_Realistic_Hamiltonian(Initial_position, Time_step, frequency, Coefficient, nquanta_list):

    sol = odeint(SCCL2_Realistic_Hamiltonian, Initial_position,Time_step, args = (frequency, Coefficient, nquanta_list) )

    return sol

def Plot_Trajectory():
    D = 32924

    # This term tune the chaos
    V_phi = 7
    # V_phi = 0

    dof = 6

    Tuple_list = prepare_Tuple_list(dof)

    frequency = [1003.1, 1003.5, 1002.9, 1002.4, 1003.8, 1001.1]  # in unit of cm^{-1}

    final_time = 0.03

    Time_step = np.linspace(0,final_time,500)

    Initial_action = [2,2,3,3,3,2]
    # Initial_angle1 = [np.random.random() * np.pi * 2 for i in range(dof)]
    Initial_angle1 = [3.7268590833121285, 1.5376804777399058, 1.0841995955875696, 5.032864515693573, 2.302900424402824, 2.820636714460905]

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

    plt.show()


