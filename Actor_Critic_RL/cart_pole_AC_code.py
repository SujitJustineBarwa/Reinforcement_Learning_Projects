import gymnasium as gym
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Signal_viewer import RealTimeLossPlot,Inside_Episode
from PyQt6 import QtWidgets
import sys

# ===================================================================
#                       ACTOR MODEL (Policy Neural Network)
# ===================================================================
class Actor(nn.Module):
    def __init__(self,state_dim = 4, action_dim = 2,hidden_dim = 8):
        super(Actor,self).__init__()

        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,action_dim)

    def forward(self,state):
        x = F.relu(self.fc1(state))
        action_logits = self.fc2(x)
        action_probs = F.softmax(action_logits,dim = -1)
        return action_probs

    def save_model(self, path="actor.pth"):
        """Save model parameters to a file"""
        torch.save(self.state_dict(), path)
        print(f"Actor model saved to {path}")

    def load_model(self, path="actor.pth"):
        """Load model parameters from a file"""
        self.load_state_dict(torch.load(path))
        self.eval()  # set to eval mode
        print(f"Actor model loaded from {path}")

# ===================================================================
#                       ACTOR MODEL (Policy Neural Network)
# ===================================================================
class Critic(nn.Module):
    def __init__(self,state_dim=4,hidden_dim=8):
        super(Critic,self).__init__()
        
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)

    def forward(self,state):
        x = F.relu(self.fc1(state))
        state_value = self.fc2(x)
        return state_value

    def save_model(self, path="critic.pth"):
        """Save model parameters to a file"""
        torch.save(self.state_dict(), path)
        print(f"Critic model saved to {path}")

    def load_model(self, path="critic.pth"):
        """Load model parameters from a file"""
        self.load_state_dict(torch.load(path))
        self.eval()  # set to eval mode
        print(f"Critic model loaded from {path}")


# ===================================================================
#                       Main Program
# ===================================================================
if __name__ == "__main__":

    # Create our training environment - a cart with a pole that needs balancing
    env = gym.make("CartPole-v1", render_mode="human",sutton_barto_reward=True)
    actor_model = Actor()
    critic_model = Critic()
    if os.path.exists("actor.pth"):
        actor_model.load_model("actor.pth")
    if os.path.exists("critic.pth"):
        critic_model.load_model("critic.pth")

    discount_factor = 0.95
    actor_optimizer = optim.Adam(actor_model.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=1e-2)

    # Plotting windows
    app = QtWidgets.QApplication(sys.argv)
    loss_plot = RealTimeLossPlot()
    loss_plot.show()

    # If you want to see the angular velocity vs action, make the flag below True
    look_inside_episode = False
    if look_inside_episode:
        inside_episode_plot = Inside_Episode()
        inside_episode_plot.show()

    try:                                         # For plot window clean exit
        for ep_num in range(1000):

            # Reset environment to start a new episode
            observation, info = env.reset()
            episode_over = False
            t = 0
            current_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            while not episode_over:

                # Choose an action based on the action probabilities
                action_prob = actor_model.forward(current_state)
                action = np.random.choice([0,1], p=action_prob.detach().numpy().squeeze())

                if look_inside_episode:
                    inside_episode_plot.update(action_prob.detach().numpy().squeeze(),
                                            current_state.detach().numpy().squeeze(),
                                            action)     # Plotting Action probabilities,state and action

                # Observe a new state and reward
                observation, reward, terminated, truncated, info = env.step(action)
                new_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                current_state_value = critic_model.forward(current_state)
                new_state_value = critic_model.forward(new_state)
                advantage = reward + discount_factor * new_state_value - current_state_value

                # Critic loss = MSE(delta) and update
                critic_loss = (reward + discount_factor * new_state_value.detach() - current_state_value).pow(2)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Actor loss = - log π(a|s) * advantage and update
                log_prob = torch.log(action_prob.squeeze(0)[action])
                actor_loss = -log_prob * advantage.detach()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                current_state = new_state
                t += 1
                if look_inside_episode:
                    app.processEvents()

                if terminated or truncated:
                    episode_over = True

            # Status check - Print episode summary and update the loss plots
            os.system('clear')
            print(f"Episode {ep_num+1} finished.")
            print(f"Total time step : {t}, \nTotal actor loss : {actor_loss.item()}, \nTotal critic loss : {critic_loss.item()}")
            actor_model.save_model("actor_model.pth")
            critic_model.save_model("critic_model.pth")
            loss_plot.update_plot(critic_loss.item(), actor_loss.item(),t) 
            if not look_inside_episode:
                app.processEvents()

        env.close()

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting gracefully...")
    finally:
        # Ensure clean shutdown of app
        app.quit()
