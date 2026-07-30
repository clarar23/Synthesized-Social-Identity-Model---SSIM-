import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import os

from SSIM import run_simulation, n, num_steps, record_every

num_seeds = 100
all_runs = []

print(f"Starting IPT Ablation sweep for {num_seeds} random seeds...")
print("Condition: Identity Theory DISABLED (gamma=0)")

for i in range(num_seeds):
    random_seed = np.random.randint(0, 1000000)
    run_data = run_simulation(seed=random_seed, gamma =0) 
    all_runs.append({'seed': random_seed, 'data': run_data})
    
    if (i + 1) % 10 == 0:
        print(f"Completed {i+1}/{num_seeds} runs...")

print("\nFormatting data for CSV export...")
csv_rows = []
for s_idx, run_dict in enumerate(all_runs):
    run = run_dict['data']
    seed_val = run_dict['seed']
    for step_idx in range(len(run['op_A'])):
        csv_rows.append({
            'run_index': s_idx,
            'seed': seed_val,
            'interaction_step': step_idx * record_every,
            'opinion_A': run['op_A'][step_idx],
            'opinion_B': run['op_B'][step_idx],
            'identity_A': run['id_A'][step_idx],
            'identity_B': run['id_B'][step_idx],
            'selfesteem_A': run['se_A'][step_idx],
            'selfesteem_B': run['se_B'][step_idx],
            'opinion_spread_A': run['sd_op_A'][step_idx],
            'opinion_spread_B': run['sd_op_B'][step_idx]
        })

df = pd.DataFrame(csv_rows)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Save to a new filename specific to this ablation study
save_path = os.path.join(script_dir, "ablation_no_IPT_results.csv")
df.to_csv(save_path, index=False, sep=";", decimal=",")

# Calculate stats
final_ops_A = [run_dict['data']['op_A'][-1] for run_dict in all_runs]
final_ops_B = [run_dict['data']['op_B'][-1] for run_dict in all_runs]

mean_A, sd_A = np.mean(final_ops_A), np.std(final_ops_A)
mean_B, sd_B = np.mean(final_ops_B), np.std(final_ops_B)

initial_spread_A = np.mean([run_dict['data']['sd_op_A'][0] for run_dict in all_runs])
initial_spread_B = np.mean([run_dict['data']['sd_op_B'][0] for run_dict in all_runs])
final_spread_A = np.mean([run_dict['data']['sd_op_A'][-1] for run_dict in all_runs])
final_spread_B = np.mean([run_dict['data']['sd_op_B'][-1] for run_dict in all_runs])

stats_df = pd.DataFrame({
    'Group': ['A (Activists)', 'B (Mainstream)'],
    'Initial_Spread': [initial_spread_A, initial_spread_B],
    'Final_Mean_Opinion': [mean_A, mean_B],
    'Macro_Robustness_SD': [sd_A, sd_B],
    'Final_Internal_Spread': [final_spread_A, final_spread_B] 
})

# Name it dynamically based on the experiment!
stats_path = os.path.join(script_dir, "ablation_no_IPT_summary_stats.csv")
stats_df.to_csv(stats_path, index=False, sep=";", decimal=",")

print("\n======================================================")
print(f"--> Main Data saved to: {save_path}")
print(f"Group A -> Final Mean: {mean_A:.4f} (Robustness SD: {sd_A:.4f})")
print(f"           Initial Spread: {initial_spread_A:.4f}  -->  Final Spread: {final_spread_A:.4f}")
print(f"Group B -> Final Mean: {mean_B:.4f} (Robustness SD: {sd_B:.4f})")
print(f"           Initial Spread: {initial_spread_B:.4f}  -->  Final Spread: {final_spread_B:.4f}")
print("======================================================")

print("\nDrawing graphs...")
plt.rcParams.update({'font.size': 14}) 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5)) 

x_axis = np.linspace(0, num_steps / n, len(all_runs[0]['data']['op_A']))
window_size = int(5000 / record_every)

for run_dict in all_runs:
    run = run_dict['data']
    ax1.plot(x_axis, run['op_A'], color='steelblue', alpha=0.1)
    ax1.plot(x_axis, run['op_B'], color='indianred', alpha=0.1)
    
    ax2.plot(x_axis, run['id_A'], color='steelblue', alpha=0.1)
    ax2.plot(x_axis, run['id_B'], color='indianred', alpha=0.1)
    
    se_A_roll = pd.Series(run['se_A']).rolling(window=window_size, min_periods=1).mean()
    se_B_roll = pd.Series(run['se_B']).rolling(window=window_size, min_periods=1).mean()
    ax3.plot(x_axis, se_A_roll, color='steelblue', alpha=0.1)
    ax3.plot(x_axis, se_B_roll, color='indianred', alpha=0.1)

mean_op_A = np.mean([run_dict['data']['op_A'] for run_dict in all_runs], axis=0)
mean_op_B = np.mean([run_dict['data']['op_B'] for run_dict in all_runs], axis=0)
mean_id_A = np.mean([run_dict['data']['id_A'] for run_dict in all_runs], axis=0)
mean_id_B = np.mean([run_dict['data']['id_B'] for run_dict in all_runs], axis=0)

mean_se_A = pd.Series(np.mean([run_dict['data']['se_A'] for run_dict in all_runs], axis=0)).rolling(window=window_size, min_periods=1).mean()
mean_se_B = pd.Series(np.mean([run_dict['data']['se_B'] for run_dict in all_runs], axis=0)).rolling(window=window_size, min_periods=1).mean()

ax1.plot(x_axis, mean_op_A, color='navy', linewidth=3, label="Group A Mean")
ax1.plot(x_axis, mean_op_B, color='darkred', linewidth=3, label="Group B Mean")
ax2.plot(x_axis, mean_id_A, color='navy', linewidth=3)
ax2.plot(x_axis, mean_id_B, color='darkred', linewidth=3)
ax3.plot(x_axis, mean_se_A, color='navy', linewidth=3)
ax3.plot(x_axis, mean_se_B, color='darkred', linewidth=3)

fig.suptitle(f"Ablation Study: Identity Process Theory DISABLED (100 Seeds)", fontsize=16, fontweight='bold')

ax1.set_title("Mean Opinion", fontsize=14, pad=10)
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlabel("Interactions per Agent")

ax2.set_title("Mean Identity Standard (Individualized)", fontsize=14, pad=10)
ax2.set_ylim(-0.05, 1.05)
ax2.set_xlabel("Interactions per Agent")

ax3.set_title(f"Mean Self-Esteem", fontsize=14, pad=10)
ax3.set_ylim(-0.05, 1.05)
ax3.set_xlabel("Interactions per Agent")

ax1.legend()
plt.tight_layout()
plt.show()
