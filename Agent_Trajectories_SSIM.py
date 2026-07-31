import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import os

from SSIM import run_simulation, n, num_steps, record_every 

target_seed = 975569

print(f"Running agent trajectory simulation for seed {target_seed}...")

run_data = run_simulation(seed=target_seed) 

print("Simulation complete! Processing micro-level agent data...")

opinions = np.array(run_data['ind_opinions'])
identities = np.array(run_data['ind_identities'])
selfesteems = np.array(run_data['ind_selfesteem'])
groups = np.array(run_data['agent_groups'])

num_recorded_points = opinions.shape[0]
time_steps = np.linspace(0, num_steps, num_recorded_points)

group_A_idx = np.where(groups == 'A')[0]
group_B_idx = np.where(groups == 'B')[0]

plt.rcParams.update({'font.size': 14}) 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

def plot_trajectories(ax, data, title, ylabel=""):
    ax.plot(time_steps, data[:, group_B_idx], color='darkred', alpha=0.05, linewidth=1)
    ax.plot(time_steps, data[:, group_A_idx], color='navy', alpha=0.2, linewidth=1)
    
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time Steps")
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.grid(True, linestyle='--', alpha=0.5)

plot_trajectories(ax1, opinions, "Individual Opinions over Time", "Variable Value (0 to 1)")
plot_trajectories(ax2, identities, "Individual Identities over Time")
plot_trajectories(ax3, selfesteems, "Individual Self-Esteem over Time")

ax1.plot([], [], color='navy', lw=3, label='Group 1')
ax1.plot([], [], color='darkred', lw=3, label='Group 2')
ax1.legend(loc='best')

fig.suptitle(f"Micro-Level Agent Trajectories (Seed: {target_seed})", fontweight='bold', fontsize=16)
plt.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))
plot_path = os.path.join(script_dir, f"exp_trajectories_seed_{target_seed}.png")
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to {plot_path}")

plt.show()
