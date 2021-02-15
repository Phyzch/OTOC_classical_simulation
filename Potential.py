import numpy as np
import scipy

def Diagonal_Hamiltonian(D,frequency,action):
    '''
    H0 = omega * J - (omega * J)^2 /(4*D)
    :param D: Dissociation energy D
    :param frequency:  frequency omega
    :param action: action variable J
    :return:  diagonal energy
    '''
    dof = len(action)
    Energy = 0
    for i in range(dof):
        Energy = Energy + frequency[i] * action[i] - pow(frequency[i] * action[i] , 2)/ (4 * D)

    return Energy

def Offdiagonal_coupling(V_phi,action,theta):
    '''
    V = V_phi * \sum_{nearest neighbor} (2* J * cos(theta) )
    :param V_phi: strength of coupling , phi in paper: https://doi.org/10.1063/1.471937
    :param action: J, action variable
    :param theta:  angle variable
    :return: offdiagonal energy
    '''

    dof = len(action)

    V = 0

    for center_index in range(dof):
        if(center_index ==0):
            left_index = dof - 1
        else:
            left_index = center_index -1

        if(center_index == dof-1):
            right_index = 0
        else:
            right_index = center_index

        index_list = [left_index,center_index,right_index]

        coupling_term = V_phi
        for index in index_list:
            coupling_term = coupling_term * 2 * np.sqrt(action[index]) * np.cos(theta[index])

        V = V + coupling_term

    return V

def prepare_Tuple_list(dof):
    Tuple_list = []
    for center_index in range(dof):
        if (center_index == 0):
            left_index = dof - 1
        else:
            left_index = center_index - 1

        if (center_index == dof - 1):
            right_index = 0
        else:
            right_index = center_index + 1

        Tuple = [left_index, center_index, right_index]

        Tuple_list.append(Tuple)

    return Tuple_list

def compute_angle_velocity(frequency,action, angle, D, V_phi , Tuple_list):
    '''

    :param frequency:
    :param action:
    :param angle:
    :param D: Dissociation energy
    :param V_phi: coupling strength for off-diagonal term (generate chaos)
    :param: Tuple_list = [ [1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,1], [6,1,2] ]
    :return: array of size [dof]. d (theta) / dt.  (np.array)
    '''
    dof = len(action)

    # integrable part
    Omega_list = []
    for i in range(dof):
        Omega = frequency[i] - pow(frequency[i],2) / (2*D)  * action[i]
        Omega_list.append(Omega)

    # contribution from V:

    Partial_V_partial_J_list = []
    for i in range(dof):
        if(action[i] != 0):
            term1 = np.cos(angle[i]) * V_phi / pow(action[i],0.5)
        else:
            term1 = 0

        partial_V_partial_J = 0

        for tuple in Tuple_list:
            if i in tuple:
                coupling = term1
                for index in tuple:
                    if(index!= i):
                        coupling = coupling * 2 * pow(action[index],0.5) * np.cos(angle[index])

                partial_V_partial_J = partial_V_partial_J + coupling

        Partial_V_partial_J_list.append(partial_V_partial_J)

    Omega_list = np.array(Omega_list)
    Partial_V_partial_J_list = np.array(Partial_V_partial_J_list)

    angle_velocity = Omega_list + Partial_V_partial_J_list

    return angle_velocity

def compute_action_velocity(V_phi,action,angle,Tuple_list):
    '''

    :param V_phi:
    :param action:
    :param angle:
    :param Tuple_list:
    :return: velocity of action variable . np.array
    '''
    dof = len(action)

    action_velocity_list = []

    for i in range(dof):
        partial_V_partial_theta = 0

        term1 = V_phi * 2 * pow(action[i],0.5) * (-np.sin(angle[i]))

        for tuple in Tuple_list:
            if(i in tuple):
                coupling = term1
                for index in tuple:
                    if(index!=i):
                        coupling = coupling * 2 * pow(action[index],0.5) * np.cos(angle[index])

                partial_V_partial_theta = partial_V_partial_theta + coupling

        action_velocity = - partial_V_partial_theta

        action_velocity_list.append(action_velocity)

    action_velocity_list = np.array(action_velocity_list)

    return action_velocity_list

