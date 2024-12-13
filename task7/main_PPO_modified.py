from stable_baselines3 import PPO #PPO
from typing import Callable
import os
from carenv import CarEnv
import time



print('This is the start of training script')

print('setting folders for logs and models')
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

print('connecting to env..')

env = CarEnv()

env.reset(seed=None)
print('Env has been reset as part of launch')
model = PPO(
    'CnnPolicy',  
    env,
    verbose=1,
    learning_rate=1e-4,  
    gamma=0.99, 
    n_steps=2048, 
    gae_lambda=0.95,  
    ent_coef=0.01, 
    vf_coef=0.5,  
    tensorboard_log=logdir
)
TIMESTEPS = 500_000 # how long is each training iteration - individual steps

for iters in range(8):
	print('Iteration ', iters,' is to commence...')
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO" )
	print('Iteration ', iters,' has been trained')
	model.save(f"{models_dir}/{TIMESTEPS*iters}")