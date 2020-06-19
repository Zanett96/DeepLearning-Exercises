import numpy as np


# Environment on which the agent can act (a square X by Y)
class Environment:
    state = []
    goal = []
    boundary = []


    # movements of the agents
    actions = {
        0: [0, 0],  # stand still
        1: [0, 1],  # go up
        2: [0, -1],  # go down
        3: [1, 0],  # go left
        4: [-1, 0],  # go right
    }

    def __init__(self, x, y, initial, goal, isPit, isLaby):
        self.boundary = np.asarray([x, y])
        self.state = np.asarray(initial)
        self.goal = goal
        self.pit = isPit
        self.laby = isLaby

    # the agent makes an action
    def move(self, action):
        walls = [80, 1, 11, 21, 41, 51, 61, 81, 22, 42, 62, 43, 63, 73, 83, 93, 4, 14, 24, 44,
                 25, 65, 75, 95, 16, 26, 46, 56, 66, 76, 96, 57, 77, 97, 8, 18, 28, 48, 58, 78]
        reward = 0
        # select the action
        movement = self.actions[action]

        # if the agent reach the goal, stand still and get reward
        if action == 0 and (self.state == self.goal).all():
            reward = 1
        next_state = self.state + np.asarray(movement)

        # check if the agent went over the walls of the labyrinth
        if self.laby and self.check_laby(next_state, walls):
            reward = -1
        # check if the agent went over the pit in the map
        elif self.pit and self.check_pit(next_state):
            reward = -1
        # check if the agent went over the boundaries of the map
        elif self.check_boundaries(next_state):
            reward = -1
        else:
            self.state = next_state

        return [self.state, reward]

    # method for checking if the agent went over the boundaries of the map
    def check_boundaries(self, state):
        out = len([num for num in state if num < 0])
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        return out > 0

    # method for checking if the agent went into the pit
    def check_pit(self, state):
        fallen = False
        for i in range(0, 6):
            if state[0] == i and state[1] == 4:
                fallen = True
        return fallen

    # method for checking if the agent went into the labyrinth walls
    def check_laby(self, state, walls):
        trespassing = False
        for i in walls:
            y = i % 10
            x = i // 10
            if state[0] == x and state[1] == y:
                trespassing = True
        return trespassing
