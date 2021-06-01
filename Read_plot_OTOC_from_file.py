import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import re

def Read_all_Eigenvalue_from_file(folder_path):
    file_path = os.path.join(folder_path, "All_trajectory_eigenvalue.txt")
    with open (file_path) as f:
        data = f.read().splitlines()

        line = data[0].strip("\n")
        line = re.split(" ", line )
        iterate_number = int(line[0])

        line = data[1].strip("\n")
        line = re.split(" ",line)
        Time = np.array([float(i) for i in line if i!='' ])

        line_index = 2
        Largest_eigenvalue_all_trajectory = []
        for _ in range(iterate_number):
            line  = data[line_index].strip("\n")
            line = re.split(" ", line)
            Largest_eigenvalue = [float(i) for i in line if i!='']
            Largest_eigenvalue_all_trajectory.append(Largest_eigenvalue)
            line_index = line_index + 1

        Largest_eigenvalue_all_trajectory = np.array(Largest_eigenvalue_all_trajectory)
        return iterate_number, Time, Largest_eigenvalue_all_trajectory

def Plot_all_Eigenvalue_for_diff_traj_from_file(folder_path):
    iterate_number, Time, Largest_eigenvalue_all_trajectory = Read_all_Eigenvalue_from_file(folder_path)

    Lyapunov_exponent_final = np.array([np.log(Largest_eigenvalue_all_trajectory[i][-1]) / Time[-1] for i in range(iterate_number)])
    sort_index = np.argsort(- Lyapunov_exponent_final)

    sorted_eigenvalue_all_trajectory = Largest_eigenvalue_all_trajectory[sort_index]

    fig, ax = plt.subplots(nrows=1,ncols=1)
    Num = min(iterate_number, 20)
    start_index = 0
    for i in range(start_index, Num):
        ax.plot(Time, sorted_eigenvalue_all_trajectory[i] ,  label = 'state ' + str(sort_index[i]))


    ax.set_xlabel('Time')
    ax.set_ylabel('Lyapunov spectrum eigenvalue ')
    ax.set_yscale('log')

    real_time = np.array(Time) * 0.03
    fig1,ax1 = plt.subplots(nrows=1,ncols=1)
    for i in range(start_index, Num):
        lyapunov_exponent = np.log(sorted_eigenvalue_all_trajectory[i][1:] ) / (2 * real_time[1:])
        ax1.plot(real_time[1:] , lyapunov_exponent , label = 'state ' + str(sort_index[i]))
    ax1.set_xlabel('time')
    ax1.set_ylabel('lyapunov exponent')

    eigenvalue_all_trajectory_sifted = sorted_eigenvalue_all_trajectory[start_index:]
    Average_eigenvalue_sifted = np.mean(eigenvalue_all_trajectory_sifted, 0)

    file = os.path.join(folder_path, "Average eigenvalue sifted.txt")
    Len = len(Time)
    with open(file,"w") as f:
        for i in range(Len):
            f.write(str(Time[i]) + " ")
        f.write("\n")
        for i in range(Len):
            f.write(str(Average_eigenvalue_sifted[i]) + " ")
        f.write("\n")

    file = os.path.join(folder_path, "Largest eigenvalue sifted.txt")
    with open(file,"w") as f:
        for i in range(Len):
            f.write(str(Time[i]) + " ")
        f.write("\n")
        for i in range(Len):
            f.write(str(sorted_eigenvalue_all_trajectory[start_index][i]) + " ")
        f.write("\n")

    plt.show()

