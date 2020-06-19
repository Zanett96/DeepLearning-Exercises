import random
import scipy.special as sp
import dill
import numpy as np

from matplotlib import colors
from matplotlib import pyplot as plt

import agent
import environment


# method for plotting the environment
def plot_map(x, y, state, goal, pit, labyrinth, walls, index):
    map = np.ones((x, y))
    cmap = colors.ListedColormap(['red', 'green', 'blue', 'black', 'white'])

    # color the pit
    if pit:
        for i in range(0, 6):
            map[i][4] = 0.75

            # color the labyrinth
    if labyrinth:
        for i in walls:
            y = i % 10
            x = i // 10
            map[x][y] = 0.75

    map[goal[0]][goal[1]] = 0.5  # color the goal
    map[state[0]][state[1]] = 0.25  # color the agent

    plt.imshow(map, cmap=cmap, interpolation='nearest')
    #plt.savefig('img' + str(index) + '.png')
    plt.plot()
    plt.pause(0.25)


# method to check recursively if the agent spawn on a wall
def start_walls(initial, walls, x, y):
    # get a random position
    initial = [np.random.randint(0, x), np.random.randint(0, y)]
    # if initial collide with a wall, call the method recursively
    for i in walls:
        yy = i % 10
        xx = i // 10
        if initial[0] == xx and initial[1] == yy:
            initial = start_walls(initial, walls, x, y)

    return initial

# Plot the rewards over the number of episodes
def rew_plot(rewards_1, rewards_2, rewards_3):

    plt.figure(figsize=(12,6))
    plt.plot(rewards_1, label='Q-Learning')
    plt.plot(rewards_2, label='SARSA ε-greedy')
    plt.plot(rewards_3, label='SARSA softmax')
    plt.xlabel('number of steps (1/50)')
    plt.ylabel('reward')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


# method for testing the agent
def test_agent(agent, x_start, y_start, epsilon, goal, pit, labyrinth, plots):
    # initialize and plot the environment
    state = [x_start, y_start]
    env = environment.Environment(x, y, state, goal, pit, labyrinth)
    if plots: plot_map(x, y, state, goal, pit, labyrinth, walls, 0)
    reward = 0
    # run episodes
    for step in range(1, 30):

        # find state index
        state_index = state[0] * y + state[1]
        # choose an action
        action = agent.select_action(state_index, epsilon)

        # the agent moves in the environment
        result = env.move(action)

        # update state
        state = result[0]
        reward += result[1]

        # plot the environment in the current state
        if plots: plot_map(x, y, state, goal, pit, labyrinth, walls, step)

        if (state[0] * y == goal[0]) and (state[1] == goal[1]):
            print('The agent reached the goal starting from x:', x_start, ' y:', y_start, 'in ', step, ' steps')
            break


## perform the validation
def validation_agent(agent, epsilon, episode_length, goal, pit, labyrinth, walls, x, y):
    avg_val = []

    for index in range(0, 100):

        # start from a random state not on the walls if inside the labyrinth
        if labyrinth:
            initial = [np.random.randint(0, x), np.random.randint(0, y)]
            for i in walls:
                yy = i % 10
                xx = i // 10
                if initial[0] == xx and initial[1] == yy:
                    initial = start_walls(initial, walls, x, y)
        else:
            # start from a random state
            initial = [np.random.randint(0, x), np.random.randint(0, y)]

        # initialize environment
        state = initial
        env = environment.Environment(x, y, state, goal, pit, labyrinth)
        val_reward = 0

        # run episode
        for step in range(0, episode_length):
            # find state index
            state_index = state[0] * y + state[1]

            # choose an action
            action = agent.select_action(state_index, epsilon)

            # the agent moves in the environment
            result = env.move(action)

            # update state and reward
            val_reward += result[1]
            state = result[0]

        val_reward /= episode_length
        avg_val.append(val_reward)

        if (index + 1) % 50 == 0:
            print('Episode ', index + 1, ': the agent has obtained an average reward of ', val_reward,
                  ' starting from position ',
                  initial)
    return np.mean(avg_val)


# method for training and validating the agent
def train_val_agent(learner, epsilon, alpha, episodes, episode_length, goal, pit, labyrinth, walls, x, y, validation):
    cumulative = 0
    tot_reward = []
    # perform the training
    for index in range(0, episodes):

        # start from a random state not on the walls
        if labyrinth:
            initial = [np.random.randint(0, x), np.random.randint(0, y)]
            for i in walls:
                yy = i % 10
                xx = i // 10
                if initial[0] == xx and initial[1] == yy:
                    initial = start_walls(initial, walls, x, y)
        else:
            # start from a random state
            initial = [np.random.randint(0, x), np.random.randint(0, y)]

        # initialize environment
        state = initial
        env = environment.Environment(x, y, state, goal, pit, labyrinth)
        reward = 0

        # run episode
        for step in range(0, episode_length):
            # find state index
            state_index = state[0] * y + state[1]

            # choose an action
            action = learner.select_action(state_index, epsilon[index])

            # the agent moves in the environment
            result = env.move(action)

            # Q-learning update
            next_index = result[0][0] * y + result[0][1]
            learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])

            # update state and reward
            reward += result[1]
            state = result[0]

        reward /= episode_length
        cumulative += reward
        tot_reward.append(cumulative)

    # Save the agent
    with open('agent.obj', 'wb') as agent_file:
        dill.dump(learner, agent_file)

    if validation:
        val_reward = validation_agent(learner, epsilon[episodes - 1], episode_length, goal, pit, labyrinth, walls, x, y)
        print('validation reward:', val_reward)
        return val_reward
    else:
        return tot_reward


# numbers of combinations of hyperparameters to try
num_iters = 100


# Random search find the best set of hyperparameters given x random combinations trough train-validation
def random_search(num_iters, verbose, goal, pit, labyrinth, walls):
    best = 0
    for i in range(num_iters):

        # Set random hyperparameters
        rand_lr = random.randint(1, 9) * (10 ** (-random.randint(0, 3)))
        rand_episode = (random.randint(10000, 20000))
        lr = np.ones(rand_episode) * rand_lr
        rand_discount = (random.randint(0, 10) * 10 ** (-1))
        rand_lenght = (random.randint(20, 50))
        rand_eps = np.linspace(random.randint(5, 10) * 10 ** (-1), random.randint(1, 5) * 10 ** (-random.randint(1, 3)),
                               rand_episode)

        # Initialize the agent
        learner = agent.Agent((10 * 10), 5, rand_discount, max_reward=1, softmax=False, sarsa=False)

        if verbose:
            print("iteration: ", i, " lr: ", rand_lr, "#episode: ", rand_episode, 'episode length: ', rand_lenght,
                  ' discount: ', rand_discount, 'epsilon :', rand_eps)

        # Compute the training
        reward = train_val_agent(learner, rand_eps, lr, rand_episode, rand_lenght, goal, pit, labyrinth, walls, 10, 10,
                                 True)

        if verbose:
            print("The average validation reward was ", reward)

        ## Update the best parameters
        if (i == 0):
            best = reward
        elif (reward > best):
            best_lr = rand_lr
            best_episode = rand_episode
            best_discount = rand_discount
            best_epsilon = rand_eps
            best = reward
            print("The best reward so far is: ", best, " with lr: ", rand_lr, "#episode: ", rand_episode,
                  'episode length: ', rand_lenght, ' discount: ', rand_discount, 'epsilon :', rand_eps)

    return best, best_lr, best_episode, best_discount, best_epsilon, rand_lenght


episodes = 20000   # number of training episodes
episode_length = 50  # maximum episode length
x = 10  # horizontal size of the box
y = 10  # vertical size of the box
goal = [0, 3]  # objective point
discount = 0.8  # exponential discount factor
softmax = True  # set to true to use Softmax policy
sarsa = True  # set to true to use the Sarsa algorithm
labyrinth = True # set to true to use the labyrinth environment
pit = False #set to true to add the pit into the environment

walls = [80, 1, 11, 21, 41, 51, 61, 81, 22, 42, 62, 43, 63, 73, 83, 93, 4, 14, 24, 44,
         25, 65, 75, 95, 16, 26, 46, 56, 66, 76, 96, 57, 77, 97, 8, 18, 28, 48, 58, 78]

## Random search - Long time to process, excute only when researching hyperparameters!!!
#best, best_lr, best_episode, best_discount, best_epsilon, rand_lenght = random_search(100, True, goal, pit, labyrinth, walls)

# learning rate
alpha = np.ones(episodes) * 0.2

# epsilon (ε) determine the balance between exploration and exploitation
# epsilon should start higher (prefer exploring) then gradually decrease (exploit knowledge of the environment)
epsilon = np.linspace(1, 0.002, episodes)

# initialize the agent
#learner = Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)

#reward_1 = train_val_agent(learner, epsilon, alpha, episodes, episode_length, goal, pit, labyrinth, walls, x, y, False)

#sarsa = True
#agent = Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)

#reward_2 = train_val_agent(agent, epsilon, alpha, episodes, episode_length, goal, pit, labyrinth, walls, x, y, False)

#softmax = True
#model = Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)

#reward_3 = train_val_agent

#rew_plot(reward_1, reward_2, reward_3)

with open('agent.obj', 'rb') as agent_file:
    agent = dill.load(agent_file)

test_agent(agent, 9, 8, epsilon[episodes-1], [0,3], False, True, True)