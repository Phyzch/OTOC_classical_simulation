import numpy as np
from scipy.integrate import odeint
from Potential import compute_angle_velocity, compute_action_velocity
from Potential import prepare_Tuple_list
import matplotlib.pyplot as plt
from SCCL2_potential import Other_molecule_angle_velocity, Other_molecule_action_velocity
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

    action_velocity = compute_action_velocity(V_phi,action,angle,Tuple_list)
    angle_velocity = compute_angle_velocity(frequency,action,angle,D,V_phi,Tuple_list)

    angle_velocity = angle_velocity.tolist()
    action_velocity = action_velocity.tolist()

    dydt = action_velocity + angle_velocity  # append two list together

    dydt = cf * np.array(dydt)

    return dydt


def Other_Molecules(y,t, V0, scaling_parameter ,frequency,f0, nquanta_list , nquanta_list_trans):
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

    angle_velocity = Other_molecule_angle_velocity(action,angle,V0,scaling_parameter,frequency,f0,nquanta_list, nquanta_list_trans)
    action_velocity = Other_molecule_action_velocity(action,angle,V0,scaling_parameter,frequency,f0,nquanta_list , nquanta_list_trans)

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

def Evolve_dynamics_Other_Molecules(Initial_position, Time_step,V0, scaling_parameter ,frequency,f0, nquanta_list , nquanta_list_trans ):

    sol = odeint(Other_Molecules,Initial_position,Time_step, args = (V0, scaling_parameter ,frequency,f0, nquanta_list , nquanta_list_trans) )

    return sol

def Evolve_dynamics_SCCL2_Realistic_Hamiltonian(Initial_position, Time_step, frequency, Coefficient, nquanta_list):

    sol = odeint(SCCL2_Realistic_Hamiltonian, Initial_position,Time_step, args = (frequency, Coefficient, nquanta_list) )

    return sol



