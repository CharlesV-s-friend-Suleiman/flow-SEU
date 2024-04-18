import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = '../rl_env/data/progress.csv'
OUTPUT_PATH = './Figur'
EXP_TAG = "fixed_time_ppo"

df = pd.read_csv(FILE_PATH)
max_r_cav, min_r_cav, r_cav = df["policy_reward_max/cav"], df["policy_reward_min/cav"], df["policy_reward_mean/cav"]
max_r_tl, min_r_tl, r_tl = df["policy_reward_max/tl"], df["policy_reward_min/tl"], df["policy_reward_mean/tl"]

iteration = df["training_iteration"]

fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=(10, 8))
ax1.scatter(iteration, max_r_cav, label="max-reward per iteration", c='y', s=15)
ax1.scatter(iteration, r_cav, label="mean-reward per iteration", c='b', s=15)
ax1.scatter(iteration, min_r_cav, label="min-reward per iteration", c='r', s=15)
ax1.fill_between(iteration, min_r_cav, max_r_cav, where=(max_r_cav > min_r_cav), color='y', alpha=0.3)
ax1.set_title("reward of CAV")
ax1.set_xlabel('number of iterations')
ax1.set_ylabel("reward of CAV")
ax1.grid(color='y', )
ax1.legend(loc="upper right")


ax2.scatter(iteration, max_r_tl, label="max-reward per iteration", c='y', s=15)
ax2.scatter(iteration, r_tl, label="mean-reward per iteration", c='b', s=15)
ax2.scatter(iteration, min_r_tl, label="min-reward per iteration", c='r', s=15)
ax2.fill_between(iteration, min_r_tl, max_r_tl, where=(max_r_tl > min_r_tl), color='y', alpha=0.3)
ax2.set_title("reward of Traffic Light")
ax2.set_xlabel('number of iterations')
ax2.set_ylabel("reward of Traffic Light")
ax2.grid(color='y', )
ax2.legend(loc="upper right")

fig.savefig(OUTPUT_PATH + '/' + EXP_TAG + '.png')
