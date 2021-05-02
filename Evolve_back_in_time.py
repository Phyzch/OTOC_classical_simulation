import numpy as np
from scipy.integrate import odeint
from Potential import compute_angle_velocity, compute_action_velocity
from Potential import prepare_Tuple_list
import matplotlib.pyplot as plt
from SCCL2_potential import Other_molecule_angle_velocity, Other_molecule_action_velocity
from SCCL2_potential import SCCL2_Realistic_Hamiltonian_action_velocity, SCCL2_Realistic_Hamiltonian_angle_velocity

cf = 2 * np.pi * 0.0299792458

def SCCL2_Realistic_Hamiltonian_back_in_time(y,t,frequency, Coefficient, nquanta_list):
    dof = int(len(y) / 2)

    action = y[0:dof]
    angle = y[dof:dof * 2]

    angle_velocity =  - SCCL2_Realistic_Hamiltonian_angle_velocity(action, angle, frequency, Coefficient, nquanta_list)
    action_velocity = - SCCL2_Realistic_Hamiltonian_action_velocity(action, angle, frequency, Coefficient,
                                                                  nquanta_list)

    angle_velocity = angle_velocity.tolist()
    action_velocity = action_velocity.tolist()

    dydt = action_velocity + angle_velocity

    dydt = cf * np.array(dydt)

    return dydt

def Evolve_dynamics_SCCL2_Realistic_Hamiltonian_back_in_time(Initial_position, Time_step, frequency, Coefficient, nquanta_list):

    sol = odeint(SCCL2_Realistic_Hamiltonian_back_in_time, Initial_position,Time_step, args = (frequency, Coefficient, nquanta_list) )

    return sol