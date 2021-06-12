import numpy as np
import matplotlib.pyplot as plt
from SCCL2_potential import Read_Realistic_SCCL2
from Evolve_dynamics import SCCL2_Realistic_Hamiltonian
from Evolve_dynamics_Using_Bulirsch import Evolve_dynamics_Realistic_SCCL2_BS_method
import os

class Clone:
    def __init__(self, dof , trajectory, nearby_trajectory ):
        self.dof = dof
        self.trajectory = np.copy(trajectory)
        self.nearby_trajectory = np.copy(nearby_trajectory)

        self.diff_vec = np.array(nearby_trajectory) - np.array(trajectory)
        self.Lyapunov_vec_len = 0


def Lyapunov_weighted_path( initial_num, max_num, iter_time , initial_action , output_folder_path):
    '''

    :param initial_num: initial number of clone
    :param max_num:  maximum number of clone allowed
    :param iter_time: maximum iteration number
    :param initial_action:  action for region of interest.
    :return:
    '''
    print("begin computing Lyapunov weighted path ")
    dof = len(initial_action)

    # alpha is used for clone
    alpha = 3
    # rate is used to control how quick the new element is added. This parameter need to be tuned
    rate = 1

    # we monitor clone_num to see if it exceed max_num
    clone_num = initial_num

    # perturbation strength
    perturb_strength = 0.1

    # nearby trajectory displacement
    nearby_trajectory_displs_strength = 0.01

    # parameter for evolving dynamics
    Time_step_len = 2
    final_time = 0.001
    Time_step = np.linspace(0, final_time, Time_step_len)

    frequency, Coefficient, nquanta_list = Read_Realistic_SCCL2()

    # initialize clone trajectory
    Clone_list = []
    for i in range(initial_num):
        initial_angle = np.random.random(dof) * 2 * np.pi
        initial_position = np.concatenate( (initial_action, initial_angle) )

        nearby_trajectory_displs = (np.random.random(2 * dof ) - 0.5) * 2 * nearby_trajectory_displs_strength

        nearby_position = initial_position + nearby_trajectory_displs

        clone_instance = Clone(dof,initial_position, nearby_position )
        Clone_list.append(clone_instance)

    step = 0
    Target_clone_num = initial_num
    while( step < iter_time and clone_num < max_num  ):
        # Evolve dynamics
        i = 0
        new_clone_num = clone_num
        rate =  float(Target_clone_num)/( clone_num )
        min_Norm_ratio = 100
        max_Norm_ratio = 0
        while(i<clone_num):
            clone_instance = Clone_list[i]
            old_Norm = np.linalg.norm(clone_instance.diff_vec)

            # add random noise to trajectory and nearby trajectory
            noise = np.random.normal(0,perturb_strength,2 * dof )
            clone_instance.trajectory = clone_instance.trajectory + noise
            clone_instance.nearby_trajectory = clone_instance.nearby_trajectory + noise

            _,sol,_ = Evolve_dynamics_Realistic_SCCL2_BS_method(clone_instance.trajectory, Time_step, frequency, Coefficient, nquanta_list)
            # final time of sol as new point.
            action_drift = sol[-1][:dof] - clone_instance.trajectory[:dof]
            new_trajectory = sol[-1]

            _,sol1,_ = Evolve_dynamics_Realistic_SCCL2_BS_method(clone_instance.nearby_trajectory, Time_step, frequency, Coefficient, nquanta_list)
            new_nearby_trajectory = sol1[-1]


            # check action within range of [-0.5 + initial_action, 0.5 + initial_action]
            # for j in range(dof):
            #     if(clone_instance.trajectory[j] > initial_action[j] + 0.5):
            #         change = - int(clone_instance.trajectory[j] - (initial_action[j] - 0.5))
            #         clone_instance.trajectory[j] = clone_instance.trajectory[j]  + change
            #         clone_instance.nearby_trajectory[j] = clone_instance.nearby_trajectory[j] + change
            #     if(clone_instance.trajectory[j] < initial_action[j] - 0.5):
            #         change = + int(initial_action[j] + 0.5 - clone_instance.trajectory[j])
            #         clone_instance.trajectory[j] = clone_instance.trajectory[j]  + change
            #         clone_instance.nearby_trajectory[j] = clone_instance.nearby_trajectory[j] + change
            #
            # for j in range(dof, 2*dof):
            #     if(clone_instance.trajectory[j] < 0 ):
            #         change = ( int( -clone_instance.trajectory[j] / (2 * np.pi) ) + 1 ) * (2 * np.pi )
            #         clone_instance.trajectory[j] = clone_instance.trajectory[j] + change
            #         clone_instance.nearby_trajectory[j] = clone_instance.nearby_trajectory[j] + change
            #     if(clone_instance.trajectory[j] > 2 * np.pi ):
            #         change = - int(clone_instance.trajectory[j] / (2 * np.pi ) ) * (2 * np.pi )
            #         clone_instance.trajectory[j] = clone_instance.trajectory[j]  + change
            #         clone_instance.nearby_trajectory[j] = clone_instance.nearby_trajectory[j] + change

            # compute Norm and rescale the size of Lyapunov vector.
            diff_vec = new_nearby_trajectory - new_trajectory
            Norm = np.linalg.norm(diff_vec)
            Norm_ratio = Norm / old_Norm

            if(Norm_ratio > max_Norm_ratio):
                max_Norm_ratio = Norm_ratio
            if(Norm_ratio < min_Norm_ratio):
                min_Norm_ratio = Norm_ratio

            clone_instance.Lyapunov_vec_len = Norm_ratio

            ran = np.random.random(1)[0]
            nk = int( np.floor( pow( Norm_ratio , alpha) * rate + ran  ) )

            # nk == 0 : clone is killed.  nk == 1, retain this clone.  nk == 2 , add more clone
            if(nk == 0):
                # clone is killed
                Clone_list.pop(i)
                clone_num = clone_num - 1
                new_clone_num = new_clone_num - 1
            else:
                for j in range(nk - 1):
                    new_clone_instance = Clone(dof, clone_instance.trajectory, clone_instance.nearby_trajectory)
                    Clone_list.append(new_clone_instance)
                    new_clone_num = new_clone_num + 1

                i = i + 1

        clone_num = new_clone_num
        print("clone number: " + str(clone_num) + "  step :  " + str(step))
        print("rate : " + str(rate))
        print("min Norm ratio:  " + str(min_Norm_ratio) + "  max Norm ratio: " + str(max_Norm_ratio))
        step = step + 1

        if(clone_num >= max_num):
            print("reach maximum number defined:  " + str(max_num))
            print("check value of rate and alpha in algorithm")

    # sort result according to their Lyapunov exponent.
    Lyapunov_vec_len  = np.array([ clone_instance.Lyapunov_vec_len for clone_instance in Clone_list])
    trajectory_list = np.array( [clone_instance.trajectory for clone_instance in Clone_list] )
    sort_index = np.argsort(-Lyapunov_vec_len)
    trajectory_list = trajectory_list[sort_index]

    # write results to file.
    output_file = os.path.join(output_folder_path, "LWP.txt")
    with open(output_file, "w") as f:
        f.write(str(clone_num) + "\n")
        for i in range(clone_num):
            trajectory = trajectory_list[i]
            # action
            for j in range(dof):
                f.write( str(trajectory[j]) + " , " )
            f.write("\n")
            # angle
            for j in range(dof, 2*dof):
                f.write(str(trajectory[j]) + " , ")
            f.write("\n")
            f.write("\n")

    # plot result in 2d diagram.
    fig,ax = plt.subplots(nrows=1, ncols=1)
    X_list = []
    Y_list = []
    for i in range(clone_num):
        clone_instance = Clone_list[i]
        trajectory = clone_instance.trajectory

        x = trajectory[0]
        y = trajectory[1]
        X_list.append(x)
        Y_list.append(y)

    ax.scatter(X_list, Y_list)

    plt.show()

initial_num = 200
max_num = 5000
iter_num = 50
initial_action = [6, 5, 1, 3, 5 ,3 ]
output_file_path = "/home/phyzch/PycharmProjects/OTOC_classical simulation/result/Lyapunov weighted path/"
# Lyapunov_weighted_path(initial_num,max_num,iter_num, initial_action, output_file_path)