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
        coupling = 3050

        for i in range(len(nquanta)):
            ni = nquanta[i]
            if ni!=0:
                # coupling = coupling * pow(-1, ni) * pow( pow(frequency[i]/f0,1/2) * scaling_parameter, ni ) \
                # * pow(2 * np.sqrt(action[i]) * np.cos(angle[i]) , ni )

                coupling = coupling * pow(-1 , ni) *pow( pow(frequency[i] , 1/2) / 270 , ni ) * pow( 2 * np.sqrt(action[i]) * np.cos(angle[i]) , ni )

        V = V + coupling

    return V



def Other_molecule_angle_velocity(action, angle, V0,  scaling_parameter, frequency, f0, nquanta_list, nquanta_list_trans):
    dof = len(action)

    # add anharmonicity
    Omega_list = []
    D = 10000
    for i in range(dof):
        Omega = frequency[i]
        Omega_list.append(Omega)



    # for vectorization: coupling_scaling = - freq[i] * (2* sqrt{J_{i}} * cos(theta(j)) / 270
    Coupling_scaling = []
    for j in range(dof):
        if(action[j] > 0):
            A = - np.sqrt(frequency[j]) * 2* np.sqrt(action[j]) * np.cos(angle[j])/270
        else:
            A = 0
        Coupling_scaling.append(A)
    Coupling_scaling  = np.array(Coupling_scaling)

    # After Vectorization
    # fixme: nquanta_list have size : [List_len,  dof]
    # fixme : Coupling_scaling have size: [dof]
    nquanta_list = np.array(nquanta_list)
    coupling_list = Coupling_scaling ** nquanta_list

    coupling_list_prod = np.prod(coupling_list, axis = 1)
    coupling_list_prod = coupling_list_prod * 3050

    # now coupling_list_prod == 3050 * \prod (- freq[i] * (2* sqrt{J_{i}} * cos(theta(j)) / 270) ^{n_{j}}
    # Now * nquanta[i]  here i is dof.
    # nquanta_list_trans : [2 * dof, nquanta_list_len] .  coupling_list_prod : [nquanta_list_len ]
    coupling_list_prod_time_nquanta = np.sum(coupling_list_prod * nquanta_list_trans  , 1)
    # Now /( 2* J_{i})
    partial_V_partial_J_list = np.zeros(dof)
    for i in range(dof):
        if(action[i] != 0):
            partial_V_partial_J_list[i] = coupling_list_prod_time_nquanta[i] / (2 * action[i])
        else:
            partial_V_partial_J_list[i] = 0


    angle_velocity = np.array(partial_V_partial_J_list) + np.array(Omega_list)

    return angle_velocity

def  Other_molecule_action_velocity(action, angle, V0,  scaling_parameter, frequency, f0, nquanta_list, nquanta_list_trans):
    dof = len(action)

    partial_V_partial_theta_list = []

    # After Vectorization:
    Coupling_scaling = []
    for j in range(dof):
        if(action[j] >= 0):
            A = - np.sqrt(frequency[j]) * 2* np.sqrt(action[j]) * np.cos(angle[j])/270
        else:
            A = 0

        Coupling_scaling.append(A)
    Coupling_scaling  = np.array(Coupling_scaling)

    nquanta_list = np.array(nquanta_list)
    coupling_list = Coupling_scaling ** nquanta_list

    coupling_list_prod = np.prod(coupling_list, axis=1)
    coupling_list_prod = coupling_list_prod * 3050

    # now coupling_list_prod == 3050 * \prod (- freq[i] * (2* sqrt{J_{i}} * cos(theta(j)) / 270) ^{n_{j}}
    # Now * nquanta[i]  here i is dof.
    # nquanta_list_trans : [2 * dof, nquanta_list_len] .  coupling_list_prod : [nquanta_list_len ]
    coupling_list_prod_time_nquanta = np.sum(coupling_list_prod * nquanta_list_trans, 1)
    partial_V_partial_theta_list = coupling_list_prod_time_nquanta * (- np.tan(angle) )


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



