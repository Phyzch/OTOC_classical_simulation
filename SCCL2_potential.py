import numpy as np

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

            # we only add 1,1,1 quanta in Hamiltoniam
            # bool1 = True
            # for j in range(dof):
            #     if(n_quanta1[j] > 1):
            #         bool1 = False
            # if bool1 == True:
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
                # we only add 111000 like coupling
                # bool1 = True
                # for j in range(dof):
                #     if (n_quanta1[j] > 1):
                #         bool1 = False
                # if bool1 == True:
                n_quanta_list.append(n_quanta1)


            else:
                Generate_n_quanta(mode_index + 1, Quanta_remain, dof, n_quanta, n_quanta_list)

    else:
        n_quanta[mode_index] = Delta_n
        n_quanta1 = np.copy(n_quanta)

        # we only add 111000 like coupling
        # bool1 = True
        # for j in range(dof):
        #     if (n_quanta1[j] > 1):
        #         bool1 = False
        # if bool1 == True:
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
    angle_velocity = np.zeros([dof])

    for i in range(dof):
        angle_velocity[i] = frequency[i]

        for nquanta in nquanta_list:
            coupling = V0

            if nquanta[i] == 0:
                continue

            for j in range(len(nquanta)):
                if (j == i):
                    ni = nquanta[i]
                    if(action[i] < 0.001):
                        coupling = 0
                    else:
                        coupling =  coupling *  pow( pow(frequency[i]/f0,1/2) * scaling_parameter, ni ) \
                    * pow(2 * np.cos(angle[i]) , ni ) * ni/2 * pow(action[i] , ni/2 -1 )

                    coupling = pow(-1, ni) * coupling
                else:
                    nj = nquanta[j]
                    if nj!=0:
                        if(action[j] < 0.001):
                            coupling = 0
                        else:
                            coupling =  coupling *  pow(pow(frequency[j] / f0, 1 / 2) * scaling_parameter, nj) \
                               * pow(2 * np.power(action[j] , 0.5) * np.cos(angle[j]), nj)
                        coupling =  pow(-1, nj) * coupling

            angle_velocity[i] = angle_velocity[i] + coupling

    return angle_velocity

def SCCL2_action_velocity(action, angle, V0,  scaling_parameter, frequency, f0, nquanta_list):
    dof = len(action)
    action_velocity = np.zeros([dof])

    for i in range(dof):
        partialJpartialtheta = 0
        for nquanta in nquanta_list:
            coupling = V0

            if nquanta[i] == 0:
                continue

            for j in range(len(nquanta)):
                if(j==i):
                    ni = nquanta[i]
                    if(action[i] < 0.001):
                        coupling = 0
                    else:
                        coupling =  coupling * pow( pow(frequency[i]/f0,1/2) * scaling_parameter, ni ) \
                    * pow(2 * np.sqrt(action[i]) , ni ) * ni * pow(np.cos(angle[i]), ni - 1) * (-np.sin(angle[i]))

                    coupling = pow(-1, ni) *  coupling
                else:
                    nj = nquanta[j]
                    if nj!=0 :
                        if(action[j] < 0.001):
                            coupling = 0
                        else:
                            coupling = coupling *  pow( pow(frequency[j]/f0,1/2) * scaling_parameter, nj ) \
                    * pow(2 * np.sqrt(action[j]) * np.cos(angle[j]) , nj )

                        coupling =  pow(-1, nj) * coupling

            partialJpartialtheta = partialJpartialtheta +  coupling

        action_velocity[i] = -partialJpartialtheta

    return action_velocity

