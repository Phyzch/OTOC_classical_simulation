import numpy as np
from scipy.integrate import odeint
from Potential import compute_angle_velocity, compute_action_velocity
from Potential import prepare_Tuple_list
import matplotlib.pyplot as plt

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

def Evolve_dynamics(Initial_position, Time_step, frequency, V_phi, D, Tuple_list):
    '''
    :return: sol: [len(t), len(y0)], [:,0]: action 0. [:,1] : action1.  [:,6] : angle 0. ...
    '''
    sol = odeint(anharmonic_oscillator,Initial_position,Time_step, args=(frequency,V_phi,D,Tuple_list) )

    return sol


def Compute_sensitivity():
    D = 32924

    # This term tune the chaos
    V_phi = 251.2

    dof = 6

    Tuple_list = prepare_Tuple_list(dof)

    frequency = [1003.1, 1003.5, 1002.9, 1002.4, 1003.8, 1001.1]  # in unit of cm^{-1}

    final_time = 0.01

    Time_step = np.linspace(0,final_time,500)

    Initial_action = [2,2,3,3,3,2]
    Initial_angle1 = [np.random.random() * np.pi * 2 for i in range(dof)]

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

    Initial_action = [2,2,3,3,3,2]

    Initial_angle2 = np.array(Initial_angle1) + np.array([0,0,phase_jitter,0,0,0])
    Initial_angle2 = Initial_angle2.tolist()

    Initial_position = Initial_action + Initial_angle2

    sol1 = Evolve_dynamics(Initial_position,Time_step,frequency,V_phi,D,Tuple_list)

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
    Sol_diff = np.array(sol1) - np.array(sol)

    fig2, ax2 = plt.subplots(nrows=2,ncols=1)

    for i in range(dof):
        ax2[0].plot(Time_step/Period , Sol_diff[:,i] , label = ' $\Delta$ J' + str(i) + ' (t)' )
        ax2[1].plot(Time_step / Period, Sol_diff[:, i + dof], label='$\Delta \phi$ ' + str(i) + " (t)")

    ax2[0].legend(loc='best')
    ax2[1].legend(loc='best')
    ax2[0].set_xlabel('t/T')
    ax2[1].set_xlabel('t/T')

    ax2[0].set_yscale('log')
    ax2[1].set_yscale('log')

    # ax2[0].set_xscale('log')
    # ax2[1].set_xscale('log')

    plt.show()