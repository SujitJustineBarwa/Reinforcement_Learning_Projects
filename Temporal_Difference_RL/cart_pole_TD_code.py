import gymnasium as gym
from itertools import product
import pandas as pd
import numpy as np
import time
import os

class state_space_discretizer:
    def __init__(self):
        num_of_bins_for_cart_position = 5
        num_of_bins_for_cart_velocity = 5
        num_of_bins_for_pole_angle = 5
        num_of_bins_for_pole_velocity = 5

        self.bins = {
            'cart_position': np.linspace(-4.8,4.8,num_of_bins_for_cart_position),
            'cart_velocity': np.linspace(-3,3,num_of_bins_for_cart_velocity),
            'pole_angle': np.linspace(-0.418,0.418,num_of_bins_for_pole_angle),
            'pole_velocity': np.linspace(-3.5,3.5,num_of_bins_for_pole_velocity)
        }

        self.all_states = list(product(list(range(1,num_of_bins_for_cart_position+1)),
                                        list(range(0,num_of_bins_for_cart_velocity+1)),
                                        list(range(1,num_of_bins_for_pole_angle+1)),
                                        list(range(0,num_of_bins_for_pole_velocity+1))))

    def discretize(self, state):

        discretized_state = (
            np.digitize(state[0], self.bins['cart_position']),
            np.digitize(state[1], self.bins['cart_velocity']),
            np.digitize(state[2], self.bins['pole_angle']),
            np.digitize(state[3], self.bins['pole_velocity'])
        )
        return discretized_state

class action_value_function:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = np.zeros((len(state_space), len(action_space)))
        self.state_idx = {state: idx for idx, state in enumerate(state_space)}

    def get_q_value(self, state, action):
        return self.q_table[self.state_idx[state]][action]

    def set_q_value(self, state, action, value):
        self.q_table[self.state_idx[state]][action] = value

    def get_action_with_max_q(self, state):
        return np.argmax(self.q_table[self.state_idx[state]])

    def save_q_table(self, filename="TD_q_table.csv"):
        """Save Q-table to a CSV file"""
        df = pd.DataFrame(self.q_table)
        df.to_csv(filename, index=False)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename="TD_q_table.csv"):
        """Load Q-table from a CSV file"""
        df = pd.read_csv(filename)
        self.q_table = df.values
        print(f"Q-table loaded from {filename}")

    def show(self):
            """Print the Q-table neatly"""
            print("Q-Table:")
            for s_idx, state in enumerate(self.state_space):
                row_str = f"State {state}: "
                for a_idx, action in enumerate(self.action_space):
                    row_str += f"A{action}={self.q_table[s_idx][a_idx]:.2f}  "
                print(row_str)

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("CartPole-v1", render_mode="human")
discretizer = state_space_discretizer()
q_func = action_value_function(discretizer.all_states, [0,1])
alpha = 0.1
eps = 0.1
discount_factor = 0.99
if os.path.exists("TD_q_table.csv"):
    q_func.load_q_table()

for ep_num in range(1000):

    # Reset environment to start a new episode
    observation, info = env.reset()
    episode_over = False
    t = 0             # timestep
    records = []
    np.random.seed(ep_num+1)
    current_state = discretizer.discretize(observation)

    while not episode_over:
        if np.random.rand() < eps:
            action = env.action_space.sample()  # action: 0 = push cart left, 1 = push cart right
        else:
            action = q_func.get_action_with_max_q(current_state)

        observation, reward, terminated, truncated, info = env.step(action)
        new_state = discretizer.discretize(observation)

        current_state_q_value = q_func.get_q_value(current_state, action)
        action_with_max_q = q_func.get_action_with_max_q(new_state)
        new_state_q_value = q_func.get_q_value(new_state, action_with_max_q)
        updated_q_value = current_state_q_value + alpha * (reward + discount_factor * new_state_q_value - current_state_q_value)
        q_func.set_q_value(current_state, action, updated_q_value)

        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)

        t += 1
        current_state = new_state

        if terminated or truncated:
            episode_over = True

    os.system('clear')
    print(f"Episode {ep_num+1} finished.")
    print(f"Total time step : {t}")
    #q_func.show()
    q_func.save_q_table()
    #time.sleep(0.5)

state_space_estimates = pd.DataFrame(state_space_estimates)
state_space_estimates.to_csv("state_space_estimates.csv", index=False)
state_space_estimates.describe()
env.close()