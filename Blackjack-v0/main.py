from collections import defaultdict
import numpy as np
import pprint
import gym

env = gym.make('Blackjack-v0')
print('Blackjack-v0')
print('Action Space:', env.action_space)
print('Observation Space:', env.observation_space)
print()

Q = defaultdict(int)
gamma = 1
alpha = 0.1
N = 100000
M = 50000

for _ in range(N):
    state = env.reset()
    # print('Initial Observation: {}\n'.format(obs))

    while True:
        # Choose action using epsilon greedy policy.
        if np.random.random() > 0.25:
            action = Q[(state, 1)] > Q[(state, 0)]
        else:
            action = np.random.randint(0, 2)
        nstate, reward, done, info = env.step(action)
        # print('Action: {}\nNext State: {}\nReward: {}\n'.format(action, nstate, reward))

        # Update q-state using exponential moving average.
        sample = reward + gamma*max(Q[(nstate, 0)], Q[(nstate, 1)]) 
        Q[(state, action)] = (1-alpha)*Q[(state, action)] + alpha*sample
        state = nstate

        # Terminate after game is finished.
        if done:
            # print("Episode terminated. {}".format('win' if reward > 0 else 'lose'))
            break

# pprint.pprint(Q)

wins, draws, losses = 0, 0, 0
for _ in range(M):
    state = env.reset()
    while True:
        action = Q[(state, 1)] > Q[(state, 0)]
        nstate, reward, done, info = env.step(action)
        state = nstate
        if done:
            wins += reward > 0
            draws += reward == 0
            losses += reward < 0
            break
print('W/D/L Percentage:', wins/M, draws/M, losses/M)

env.close()
