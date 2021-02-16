import numpy as np
import re

def Generate_n_quanta_list_for_SCCL2(dof):
    Delta_n = 3
    nquanta_list1 = Generate_n_quanta_list(Delta_n, dof)

    # Delta_n = 4
    # nquanta_list2 = Generate_n_quanta_list(Delta_n, dof)

    nquanta_list = nquanta_list1

    return nquanta_list

def Generate_n_quanta_list(Delta_n , dof):
    # Delta_n = 3 or 4
    n_quanta = np.zeros([dof])
    n_quanta_list = []
    mode_index = 0
    for quanta in range(Delta_n,-1,-1):
        Quanta_remain = Delta_n - quanta
        n_quanta[mode_index] = quanta
        if(Quanta_remain == 0):
            for j in range(mode_index + 1, dof):
                n_quanta[j] = 0
            n_quanta1 = np.copy(n_quanta)

            n_quanta_list.append(n_quanta1)

        else:
            Generate_n_quanta(mode_index + 1, Quanta_remain, dof, n_quanta, n_quanta_list)

    return n_quanta_list


def Generate_n_quanta( mode_index, Delta_n, dof, n_quanta, n_quanta_list ):
    if mode_index < dof - 1:
        for quanta in range(Delta_n,-1,-1):
            Quanta_remain = Delta_n - quanta
            n_quanta[mode_index] = quanta
            if(Quanta_remain == 0):
                for j in range(mode_index + 1, dof):
                    n_quanta[j] = 0
                n_quanta1 = np.copy(n_quanta)

                n_quanta_list.append(n_quanta1)


            else:
                Generate_n_quanta(mode_index + 1, Quanta_remain, dof, n_quanta, n_quanta_list)

    else:
        n_quanta[mode_index] = Delta_n
        n_quanta1 = np.copy(n_quanta)

        n_quanta_list.append(n_quanta1)


    return

def SCCL2_Diagonal_Hamiltonian(frequency,action):
    dof = len(action)
    Energy = 0
    for i in range(dof):
        Energy = Energy + frequency[i] * action[i]

    return Energy

def SCCL2_Offdiagonal_coupling(action, angle, V0,  scaling_parameter, frequency, f0, nquanta_list):
    V = 0
    for nquanta in nquanta_list:
        coupling = V0

        for i in range(len(nquanta)):
            ni = nquanta[i]
            if ni!=0:
                coupling = coupling * pow(-1, ni) * pow( pow(frequency[i]/f0,1/2) * scaling_parameter, ni ) \
                * pow(2 * np.sqrt(action[i]) * np.cos(angle[i]) , ni )

        V = V + coupling

    return V



def SCCL2_angle_velocity(action, angle, V0,  scaling_parameter, frequency, f0, nquanta_list):
    dof = len(action)

    # add anharmonicity
    Omega_list = []
    D = 10000
    for i in range(dof):
        Omega = frequency[i] - pow(frequency[i],2) / (2*D)  * action[i]
        Omega_list.append(Omega)


    partial_V_partial_J_list = []

    for i in range(dof):
        partial_V_partial_J = 0
        if(action[i] == 0):
            partial_V_partial_J_list.append(0)
            continue

        for nquanta in nquanta_list:
            coupling = V0
            if(nquanta[i] == 0):
                continue
            for j in range(dof):
                if(nquanta[j] == 0):
                    continue
                nj = nquanta[j]

                coupling = coupling * pow(-1,nj) * np.power(scaling_parameter * pow(frequency[j] / f0 , 0.5) , nj ) *\
                np.power(2 * pow(action[j] , 0.5) * np.cos(angle[j]) , nj )

            coupling = coupling * (nquanta[i]/2) / action[i]

            partial_V_partial_J = partial_V_partial_J + coupling

        partial_V_partial_J_list.append(partial_V_partial_J)

    angle_velocity = np.array(partial_V_partial_J_list) + np.array(Omega_list)

    return angle_velocity

def  SCCL2_action_velocity(action, angle, V0,  scaling_parameter, frequency, f0, nquanta_list):
    dof = len(action)

    partial_V_partial_theta_list = []

    for i in range(dof):
        if(action[i] == 0):
            partial_V_partial_theta_list.append(0)
            continue

        partial_V_partial_theta = 0

        for nquanta in nquanta_list:
            coupling = V0
            if(nquanta[i] == 0):
                continue

            for j in range(dof):
                if(nquanta[j] == 0):
                    continue
                nj = nquanta[j]
                coupling = coupling * pow(-1,nj) * np.power( scaling_parameter * pow(  frequency[j] / f0 ,0.5) , nj ) * \
                           pow(2 * pow(action[j] ,0.5) * np.cos(angle[j]) , nj)

            coupling = coupling * nquanta[i] * (-np.tan(angle[i]))

            partial_V_partial_theta = partial_V_partial_theta + coupling

        partial_V_partial_theta_list.append(partial_V_partial_theta)

    action_velocity = - np.array(partial_V_partial_theta_list)

    return action_velocity

def SCCL2_Realistic_Hamiltonian_angle_velocity(action, angle, frequency, Coefficient, nquanta_list):
    '''

    :param action:
    :param angle:
    :param frequency:
    :param Coefficient: coefficient for A q1^{2} q2 etc.
    :param nquanta_list: [1,0,2,0,0,0] stands for q1 * q3^2 , here q = \sqrt(2J) * cos(\theta)
    :return:
    '''
    dof = len(action)
    partial_V_partial_J_list = []

    Omega_list = []
    D = 10000
    for i in range(dof):
        Omega = frequency[i] - pow(frequency[i],2) / (2*D)  * action[i]
        Omega_list.append(Omega)

    for i in range(dof):
        partial_V_partial_J = 0
        if (action[i] == 0):
            partial_V_partial_J_list.append(0)
            continue
        Len = len(Coefficient)
        for j in range(Len):
            nquanta = nquanta_list[j]
            if (nquanta[i] == 0):
                continue

            coupling = Coefficient[j]
            for k in range(dof):
                nk = nquanta[k]
                if(nquanta[k] == 0):
                    continue
                coupling = coupling * pow(np.sqrt(2 * action[k]) * np.cos(angle[k]) , nk)

            coupling = coupling * (nquanta[i] /2 ) / action[i]
            partial_V_partial_J = partial_V_partial_J + coupling

        partial_V_partial_J_list.append(partial_V_partial_J)

    angle_velocity = np.array(partial_V_partial_J_list) + np.array(frequency)
    # angle_velocity = np.array(partial_V_partial_J_list) + np.array(Omega_list)

    return angle_velocity

def SCCL2_Realistic_Hamiltonian_action_velocity(action, angle, frequency, Coefficient, nquanta_list):
    '''

    :param action:
    :param angle:
    :param frequency:
    :param Coefficient: coefficient for A q1^{2} q2 etc.
    :param nquanta_list: [1,0,2,0,0,0] stands for q1 * q3^2 , here q = \sqrt(2J) * cos(\theta)
    :return:
    '''
    dof = len(action)
    partial_V_partial_theta_list = []

    for i in range(dof):
        partial_V_partial_theta = 0

        if(action[i] == 0):
            partial_V_partial_theta_list.append(0)
            continue

        Len = len(Coefficient)
        for j in range(Len):
            nquanta = nquanta_list[j]
            if nquanta[i] == 0:
                continue

            coupling = Coefficient[j]

            for k in range(dof):
                nk = nquanta[k]
                if (nquanta[k] == 0):
                    continue
                coupling = coupling * pow(np.sqrt(2 * action[k]) * np.cos(angle[k]), nk)

            coupling = coupling * nquanta[i] * (-np.tan(angle[i]))
            partial_V_partial_theta = partial_V_partial_theta + coupling

        partial_V_partial_theta_list.append(partial_V_partial_theta)

    action_velocity = - np.array(partial_V_partial_theta_list)

    return action_velocity


def Read_Realistic_SCCL2():
    file_path = "/home/phyzch/PycharmProjects/OTOC_classical simulation/Hamiltonian/pot.dat"
    dof = 6
    frequency = []
    Coefficient = []
    nquanta_list = []
    with open(file_path) as f:
        data = f.read().splitlines()
        linenumber = 1
        for i in range(dof):
            line = data[linenumber].strip('\n')
            line = re.split(' ',line)
            line = [i for i in line if i!='']
            frequency.append(float(line[0]))
            linenumber = linenumber + 1

        linenumber = linenumber + 6
        Len = len(data)
        while(linenumber < Len ):
            line = data[linenumber].strip('\n')
            line = re.split(' ', line)
            line = [i for i in line if i!='']

            Coefficient.append( round(float(line[0]) , 4) )

            nquanta = [ int(line[i]) for i in range(1, 1+dof) ]
            nquanta_list .append(nquanta)

            linenumber = linenumber + 1

        return frequency, Coefficient, nquanta_list



