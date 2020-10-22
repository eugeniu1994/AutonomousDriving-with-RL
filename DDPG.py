import sys
import numpy as np
import math
import random
import matplotlib.pyplot as plt

import gym
import gym_race
from rl import DDPG

def plot_rewards(rewards):
    plt.figure(2)
    plt.clf()
    rewards_t = rewards
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    plt.grid(True)
    plt.plot(rewards_t)
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = np.mean(rewards_t)
        plt.plot(means)

    plt.pause(0.001)  # pause a bit so that plots are updated

MAX_EPISODES = 2001
MAX_EP_STEPS = 1000
ON_TRAIN = False
ON_TRAIN = True

# set env
env = gym.make("Pyrace-v0")
s_dim = 5
a_dim = 1
a_bound = [-5., 5.]

print('s_dim:{}, a_dim:{},a_bound:{}'.format(s_dim,a_dim,a_bound))
rl = DDPG(a_dim, s_dim, a_bound)
steps = []
def train():
    cumulative_rewards = []
    for i in range(1,MAX_EPISODES+1):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            a = rl.choose_action(np.array(s))
            #print('a ',a)
            s_, r, done, _ = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()
                #print('Learn')

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break

            if i>200:
                env.render()

        cumulative_rewards.append(ep_r)
        if i % 100 == 0:
            plot_rewards(cumulative_rewards)

    rl.save()


if ON_TRAIN:
    train()



print('Done')