from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
import copy
from random import choice, uniform

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from tqdm import tqdm


def random_coordinates(boundary: int) -> tuple:
    """Generates random x, y coordinates given a bounding box (-boundary, +boundary)."""
    x = choice([i for i in range(-boundary, boundary+1)])
    y = choice([i for i in range(-boundary, boundary+1)])
    return x, y


def generate_random_obstacle_map(boundary: int, num_obstacles: int) -> set:
    return {random_coordinates(boundary) for _ in range(num_obstacles)}


class Action(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class State:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def copy(self):
        return copy.deepcopy(self)

    def distance_from(self, other: State) -> float:
        return (
            self.x - other.x
        )**2 + (
            self.y - other.y
        )**2

    def __str__(self):
        return str((self.x, self.y))

    def __eq__(self, another):
        return self.x == another.x and self.y == another.y

    def __hash__(self):
        return hash((self.x, self.y))

    @staticmethod
    def random(boundary):
        x, y = random_coordinates(boundary)
        return State(x, y)


class Car:
    def __init__(self, state: State) -> None:
        self.state = state

    def take_action(self, action: Action) -> State:
        """Returns the potential next state of the car if it takes action in state."""
        if action == Action.UP:
            next_state = Car.move_forward(self.state)
        elif action == Action.DOWN:
            next_state = Car.move_backwards(self.state)
        elif action == Action.LEFT:
            next_state = Car.move_left(self.state)
        elif action == Action.RIGHT:
            next_state = Car.move_right(self.state)
        else:
            raise Exception('Action Not Allowed!')
        return next_state

    @staticmethod
    def move_forward(state: State) -> State:
        return State(
            x=state.x,
            y=state.y + 1
        )

    @staticmethod
    def move_backwards(state: State) -> State:
        return State(
            x=state.x,
            y=state.y - 1
        )

    @staticmethod
    def move_right(state: State) -> State:
        return State(
            x=state.x + 1,
            y=state.y
        )

    @staticmethod
    def move_left(state: State) -> State:
        return State(
            x=state.x - 1,
            y=state.y
        )


class Game:
    def __init__(
            self,
            boundary: int,
            start_state: State,
            end_state: State,
            obstacles: list,
            max_steps_timeout: int) -> None:
        self.boundary = boundary
        self.start_state = start_state
        self.end_state = end_state
        self.obstacles = obstacles
        self.max_steps_timeout = max_steps_timeout
        self.n_steps = 0
        self.n_boundary_hits = 0
        self.n_obstacles_hits = 0
        # make sure the starting state is wthin boundary
        assert self.is_state_allowed(
            start_state) and self.is_state_allowed(end_state), \
            Exception(
                f'Bad starting state! start_state: {start_state} end_state:{end_state}')
        # put the car at the starting state
        self.car = Car(state=start_state)
        self.state = self.car.state
        # TODO(zakariae): verify that the state of the game is the same as
        # the state of the car at all times

    def next_state(self, action: Action):
        """Moves the game to the next state according to a certain action."""
        self.n_steps += 1
        # pass the action to the car and see where it might move
        potential_next_state = self.car.take_action(
            action=action
        )
        # measure the reward based on the action taken
        reward = self.get_reward(
            curr_state=self.state,
            next_state=potential_next_state
        )
        # only change state if the move is allowed inside the game
        if self.is_state_allowed(potential_next_state):
            # update the state of the game
            self.car.state = potential_next_state
            self.state = self.car.state

        return reward

    def is_state_allowed(self, state: State) -> bool:
        """Returns True if state is allowed, otherwise False."""
        return self.is_within_boundary(state) and (not self.is_on_obstacles(state))

    def is_within_boundary(self, state: State) -> bool:
        """Checks if the state is within the boundaries of the game."""
        return abs(state.x) <= self.boundary and \
            abs(state.y) <= self.boundary

    def is_on_obstacles(self, state: State) -> bool:
        """Checks if we are on any obstacles."""
        return (state.x, state.y) in self.obstacles

    def is_end_game(self):
        """Checks if the game is in the end state."""
        return (self.state == self.end_state) or (self.n_steps >= self.max_steps_timeout)

    def restart_game(self, start_state: State):
        """Restarts the game."""
        # move the car back to the starting state
        self.n_steps = 0
        self.n_boundary_hits = 0
        self.n_obstacles_hits = 0
        self.car.state = start_state
        self.state = start_state
        self.start_state = start_state

    def get_reward(self, curr_state: State, next_state: State) -> float:
        """Measures the reward for moving from one state to the next."""
        distance_curr_state_end_state = curr_state.distance_from(
            self.end_state)
        distance_next_state_end_state = next_state.distance_from(
            self.end_state)
        # if you hit a boundary
        if not self.is_within_boundary(next_state):
            self.n_boundary_hits += 1
            reward = -3
        # if you hit an obstacle
        elif self.is_on_obstacles(next_state):
            self.n_obstacles_hits += 1
            reward = -3
        # if we moved closer to target
        elif distance_next_state_end_state < distance_curr_state_end_state:
            reward = 1
        else:
            reward = -1
        return reward

    def get_vis_values(self):
        """Returns matrix used to visualize the current state of the game."""
        side_size = 2 * self.boundary + 1
        vis_values = np.zeros((side_size, side_size))
        # add start state
        vis_values[self.start_state.y + self.boundary, self.start_state.x + self.boundary] = Colors.YELLOW.value
        # add end state
        vis_values[self.end_state.y + self.boundary, self.end_state.x + self.boundary] = Colors.GREEN.value
        # add obstacles
        for x, y in self.obstacles:
            vis_values[y + self.boundary, x + self.boundary] = Colors.RED.value
        # add current position
        vis_values[self.state.y + self.boundary, self.state.x + self.boundary] = Colors.BLUE.value
        return vis_values


class ExploreStrategy(ABC):

    @abstractmethod
    def get_action(self, Q: dict, curr_state: State):
        pass


class EpsilonGreedy(ExploreStrategy):
    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def get_action(self, Q: dict, curr_state: State) -> Action:
        """
        Generates a random action with epsilon probability, otherwise it
        chooses the action with the highest Q value.
        """
        l_actions = list(Action)
        # generate random numner
        if uniform(0, 1) < self.epsilon:
            # pick a random action
            next_action = choice(l_actions)
        else:
            # choose action with the highest q-value
            index_best_action = np.argmax(
                [
                    Q[(curr_state, a)] for a in l_actions
                ]
            )
            next_action = l_actions[index_best_action]
        return next_action

class Colors(Enum):
    WHITE = 0
    YELLOW = 2
    BLUE = 4
    RED = 6
    GREEN = 8



class Learner:
    def __init__(self,
                 game: Game,
                 explore_strategy: ExploreStrategy,
                 learning_rate: float,
                 discount_rate: float
                 ):
        self.game = game
        self.Q_values = defaultdict(lambda: 0)
        self.game_no = 1
        self.explore_strategy = explore_strategy
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        # visualization
        self.im = None
        self.vis_values = None

    def learn(self, n_iters: int, visualize: bool):
        if visualize:
            # learn without viz
            fig = plt.figure()
            self.vis_values = self.game.get_vis_values()
            from matplotlib.colors import ListedColormap, BoundaryNorm
            # cmap = Colormap('viridis', 4)
            cmap = ListedColormap([i.name for i in list(Colors)])
            self.im = plt.imshow(
                self.vis_values, cmap=cmap, origin='lower')
            anim = animation.FuncAnimation(
                fig,
                self.run_iter,
                frames=4,
                interval=50)
            plt.show()
        else:
            # learn with visualization
            for _ in tqdm(range(n_iters)):
                self.run_iter()

    def run_iter(self, i):
        # if car in the last state, re-start the game
        if self.game.is_end_game():
            print(f'END of game number: {self.game_no}')
            print(
                {
                    'number of steps': self.game.n_steps,
                    'number of boundary hits': self.game.n_boundary_hits,
                    'number of obstacles hits': self.game.n_obstacles_hits
                }
            )
            # generate random start state
            start_state = State.random(boundary=self.game.boundary)
            # restart the game
            self.game.restart_game(start_state=start_state)
            self.game_no += 1
        # else, take a step
        # store what's the current state of the game
        curr_state = self.game.state.copy()
        # decide which action to take based on the explore strategy in the Learner
        action = self.explore_strategy.get_action(
            Q=self.Q_values,
            curr_state=curr_state)
        # apply action
        reward = self.game.next_state(action=action)
        # get the new visualization values in the new state
        if self.vis_values is not None:
            self.vis_values = self.game.get_vis_values()
            self.im.set_data(self.vis_values)
            plt.title(f'Game number: {self.game_no}')
        # update the Q function
        self.update_Q_values(
            prev_state=curr_state,
            action=action,
            new_state=self.game.state,
            reward=reward)
        return self.im

    def update_Q_values(self, prev_state: State, action: Action, new_state: State, reward: float) -> None:
        """
        Updates the Q values.

        Given we were in state S1 and took action A, and moved to state S2 and got reward R, we update
        as follows:
        Q[S1, A] = (1 - lr) * Q[S1, A] + lr * (R + max([Q[S2, i] for i in l_actions]))
        """
        # get old values
        old_values = self.Q_values[(prev_state, action)]
        # fresh values are the last reward
        max_value_next_state = max(
            [
                self.Q_values[(new_state, a)] for a in list(Action)
            ]
        )
        fresh_values = reward + self.discount_rate * max_value_next_state
        self.Q_values[(prev_state, action)] = (1 - self.learning_rate) * \
            old_values + self.learning_rate * fresh_values
        # print(
        #     {
        #         'prev_state': str(curr_state),
        #         'new_state': str(self.game.state),
        #         'car state': str(self.game.car.state),
        #         'action': action,
        #         'reward': reward
        #     }
        # )


if __name__ == "__main__":
    # learning variables
    N_ITERS = 10000
    LEARNING_RATE = 0.3
    DISCOUNT_RATE = 0.9
    # strategy variables
    EPSILON = 0.1
    # game variables
    BOUNDARY = 10
    NUM_OBSTACLES = 20
    MAX_STEPS_TIMEOUT = 1000
    # create the game you want to play
    obstacles = generate_random_obstacle_map(
        boundary=BOUNDARY, num_obstacles=NUM_OBSTACLES)
    game = Game(
        boundary=BOUNDARY,
        start_state=State.random(BOUNDARY),
        end_state=State.random(BOUNDARY),
        obstacles=obstacles,
        max_steps_timeout=MAX_STEPS_TIMEOUT
        )
    # choose explore strategy
    explore_strategy = EpsilonGreedy(epsilon=EPSILON)
    # create learner
    learner = Learner(
        game,
        explore_strategy=explore_strategy,
        learning_rate=LEARNING_RATE,
        discount_rate=DISCOUNT_RATE
    )
    # start training
    learner.learn(n_iters=N_ITERS, visualize=True)
