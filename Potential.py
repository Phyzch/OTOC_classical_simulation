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

    Tuple_list_len = len(Tuple_list)

    # Generate new tuple list for vectorization when computing angular and action velocity
    New_Tuple_list = []
    for i in range(Tuple_list_len):
        Tuple = Tuple_list[i]
        A = np.zeros([dof])
        for j in Tuple:
            A[j] = 1

        New_Tuple_list.append(A)
    New_Tuple_list = np.array(New_Tuple_list)

    return New_Tuple_list

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
        # Omega = frequency[i] - pow(frequency[i],2) / (2*D)  * action[i]
        Omega = frequency[i]
        Omega_list.append(Omega)

    # contribution from V:
    # Vectorize this part: V * 1/(2J_{i}) * [\prod ( 2 * sqrt(J_{j}) cos(\theta_{j} ) * Tuple_{i}]

    # rearrange Tuple list from [0,2,4] to [1,0,1,0,1, ... ] :
    Tuple_list_trans = np.transpose(Tuple_list)

    coupling_scaling = 2 * np.sqrt(action) * np.cos(angle)
    # coupling_scaling : [dof] .  New_tuple_list : [list_len, dof]
    Coupling = coupling_scaling ** Tuple_list
    # Now Coupling_prod have size [list_len]
    Coupling_prod = np.prod(Coupling , axis = 1)
    # Now * n_{i}
    Coupling_prod = Coupling_prod * Tuple_list_trans

    Coupling_prod_sum = np.sum(Coupling_prod , axis = 1)

    Coupling_prod_sum = Coupling_prod_sum * V_phi

    Partial_V_partial_J_list = Coupling_prod_sum / ( 2* np.array(action))


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


    Tuple_list_trans = np.transpose(Tuple_list)

    coupling_scaling = 2 * np.sqrt(action) * np.cos(angle)
    # coupling_scaling : [dof] .  New_tuple_list : [list_len, dof]
    Coupling = coupling_scaling ** Tuple_list
    # Now Coupling_prod have size [list_len]
    Coupling_prod = np.prod(Coupling , axis = 1)
    # Now * n_{i}
    Coupling_prod = Coupling_prod * Tuple_list_trans

    Coupling_prod_sum = np.sum(Coupling_prod , axis = 1)
    Coupling_prod_sum = Coupling_prod_sum * V_phi

    partial_V_partial_theta = Coupling_prod_sum * (-np.tan(angle))

    action_velocity_list = - partial_V_partial_theta

    return action_velocity_list

