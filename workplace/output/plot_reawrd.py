import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = '../rl_env/data/progress.csv'
OUTPUT_PATH = './Figur'
EXP_TAG = "fixed_time_ppo"

df = pd.read_csv(FILE_PATH)
max_r, min_r,r = df["episode_reward_max"], df["episode_reward_min"], df["episode_reward_mean"]
iteration = df["training_iteration"]



fig = plt.figure(figsize=(10, 8))
plt.scatter(iteration, max_r, label="max-reward per iteration", c='y',s=15)
plt.scatter(iteration, r, label="mean-reward per iteration", c='b',s=15)
plt.scatter(iteration, min_r, label="min-reward per iteration", c='r',s=15)
plt.fill_between(iteration, min_r, max_r, where=(max_r>min_r),color='y',alpha=0.3)

plt.xlabel('number of iterations')
plt.ylabel("reward")
plt.grid(color='y',)
plt.legend(loc="upper right")

fig.savefig(OUTPUT_PATH + '/' + EXP_TAG + '.png')


