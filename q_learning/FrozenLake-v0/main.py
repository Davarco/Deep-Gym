from collections import defaultdict
import numpy as np
import pprint
import gym


def test_parameters(env, params):
    Q = defaultdict(int)
    gamma = params['gamma']
    alpha = params['alpha']
    epsilon = params['epsilon']
    decay_rate = params['decay_rate']
    decay_freq = params['decay_freq']
    min_epsilon = params['min_epsilon']
    N = params['N']
    M = params['M']

    for n in range(N):
        state = env.reset()
        # print('Initial Observation: {}\n'.format(state))
        if n % decay_freq == 0:
            epsilon = max(min_epsilon, epsilon*decay_rate)

        # while True:
        for _ in range(100):
            # Choose action using epsilon greedy policy.
            if np.random.random() > epsilon:
                Q_temp = [Q[(state, i)] for i in range(4)]
                action = Q_temp.index(max(Q_temp))
            else:
                action = env.action_space.sample()
            nstate, reward, done, info = env.step(action)
            # print('Action: {}\nNext State: {}\nReward: {}\n'.format(action, nstate, reward))

            # Update q-state using exponential moving average.
            sample = reward + gamma*max([Q[(nstate, i)] for i in range(4)]) 
            Q[(state, action)] = (1-alpha)*Q[(state, action)] + alpha*sample
            state = nstate

            # Terminate after game is finished.
            if done:
                # print("Episode terminated. {}".format('win' if reward > 0 else 'lose'))
                break

    wins = 0
    for _ in range(M):
        state = env.reset()
        while True:
            Q_temp = [Q[(state, i)] for i in range(4)]
            action = Q_temp.index(max(Q_temp))
            nstate, reward, done, info = env.step(action)
            state = nstate
            if done:
                wins += reward
                break
    print(wins/M)
    return wins/M

def search_parameters(env):
    # G = np.linspace(0.25, 0.99, 15)
    # A = np.linspace(0.85, 0.99, 5)
    G = [0.99]
    A = [0.10]
    T = 1
    results = []
    for gamma in G:
        for alpha in A:
            total = 0
            params = {
                'gamma': gamma,
                'alpha': alpha, 
                'epsilon': 1, 
                'decay_rate': 0.95,
                'decay_freq': 100,
                'min_epsilon': 0.1,
                'N': 10000,
                'M': 1000
            }
            for _ in range(T):
                total += test_parameters(env, params)
            total /= T
            results.append((total, params))
            print('Gamma: {}\nAlpha: {}\nScore: {}\n'.format(gamma, alpha, total))

    hp = max(results)[1]
    print(hp)
    return hp

def main():
    env = gym.make('FrozenLake-v0')
    print('FrozenLake-v0')
    print('Action Space:', env.action_space)
    print('Observation Space:', env.observation_space)
    print()
    search_parameters(env)
    env.close()

if __name__ == '__main__':
    main()
