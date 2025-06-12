import numpy as np
import matplotlib.pyplot as plt



with open("examples/com.npy", 'rb') as f:
    com = np.load(f)
    RR = np.load(f)
    RL = np.load(f)

# plot com position in xy plane
def plot_com_xy():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # simulation step of 1 ms
    time = np.arange(com.shape[0])
    
    # plot com position in xy plane
    ax.plot(com[:, 0], com[:, 1], label='COM Position', color='blue')
    ax.plot(RR[:,0], RR[:,1], label="RR foot pos", color='red')
    ax.plot(RL[:,0], RL[:,1], label="RL foot pos")

    # add grid and labels
    ax.grid(True)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Center of Mass (COM) Position in XY Plane')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("examples/results/com_position.png")
    plt.show()

# 3D plot of com position
def plot_com_3d():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot com position in 3D
    ax.plot(com[:, 0], com[:, 1], com[:, 2], ls='dashdot', label='COM Position', color='blue')
    ax.plot(RR[:,0], RR[:,1], RR[:,2], ls='--', label="RR foot pos", color='red')
    ax.plot(RL[:,0], RL[:,1], RL[:,2], ls='--', label="RL foot pos")

    # add start and end points for COM
    ax.scatter(com[0, 0], com[0, 1], com[0, 2], color='green', s=80, marker='o', label='COM Start')
    ax.scatter(com[-1, 0], com[-1, 1], com[-1, 2], color='black', s=80, marker='x', label='COM End')
    
    # add start and end points for feet
    ax.scatter(RR[0, 0], RR[0, 1], RR[0, 2], color='orange', s=80, marker='o', label='RR Start')
    ax.scatter(RR[-1, 0], RR[-1, 1], RR[-1, 2], color='red', s=80, marker='x', label='RR End')
    ax.scatter(RL[0, 0], RL[0, 1], RL[0, 2], color='purple', s=80, marker='o', label='RL Start')
    ax.scatter(RL[-1, 0], RL[-1, 1], RL[-1, 2], color='magenta', s=80, marker='x', label='RL End')

    # add points every 10 steps
    for i in range(0, com.shape[0], 10):
        ax.scatter(com[i, 0], com[i, 1], com[i, 2], color='blue', s=20, marker='o')
        ax.scatter(RR[i, 0], RR[i, 1], RR[i, 2], color='red', s=20, marker='o')
        ax.scatter(RL[i, 0], RL[i, 1], RL[i, 2], color='purple', s=20, marker='o')
    
    # add grid and labels
    ax.grid(True)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Center of Mass (COM) Position in 3D')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("examples/results/com_position_3d.png")
    plt.show()

plot_com_3d()