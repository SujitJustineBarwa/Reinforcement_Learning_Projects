from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np

class RealTimeLossPlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Loss Visualization")
        self.setGeometry(100, 100, 1500, 500)                    # (x, y, width, height)
 
        # GraphicsLayoutWidget to hold multiple plots in a grid
        self.layout = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.layout)

        self.critic_loss = self.layout.addPlot(row=0, col=0, title="Critic Loss")    # Graph 1
        self.critic_loss.setLabel('left', 'Value function Loss')
        self.critic_loss.showGrid(x=True, y=True)
        self.actor_loss = self.layout.addPlot(row=0, col=1, title="Actor Loss")      # Graph 2
        self.actor_loss.setLabel('left', 'Policy function Loss')
        self.actor_loss.showGrid(x=True, y=True)
        self.survival_time = self.layout.addPlot(row=0, col=2, title="Survival Time")      # Graph 3
        self.survival_time.setLabel('left', 'Survival Time')
        self.survival_time.showGrid(x=True, y=True)

        self.signal_size = 100
        self.critic_loss_data = np.full((self.signal_size,), np.nan)
        self.actor_loss_data = np.full((self.signal_size,), np.nan)
        self.survival_time_data = np.full((self.signal_size,), np.nan)

    def update_plot(self, critic_loss, actor_loss, survival_time, smoothing_alpha=0.05):

        # Update the data arrays
        self.critic_loss_data = np.roll(self.critic_loss_data, -1)
        if not np.isnan(self.critic_loss_data[-2]):
            self.critic_loss_data[-1] = smoothing_alpha * critic_loss + (1 - smoothing_alpha) * self.critic_loss_data[-2]
        else:
            self.critic_loss_data[-1] = critic_loss

        self.actor_loss_data = np.roll(self.actor_loss_data, -1)
        if not np.isnan(self.actor_loss_data[-2]):
            self.actor_loss_data[-1] = smoothing_alpha * actor_loss + (1 - smoothing_alpha) * self.actor_loss_data[-2]
        else:
            self.actor_loss_data[-1] = actor_loss

        self.survival_time_data = np.roll(self.survival_time_data, -1)
        if not np.isnan(self.survival_time_data[-2]):
            self.survival_time_data[-1] = smoothing_alpha * survival_time + (1 - smoothing_alpha) * self.survival_time_data[-2]
        else:
            self.survival_time_data[-1] = survival_time

        # Update the plots
        self.critic_loss.clear()
        self.critic_loss.plot(self.critic_loss_data, pen='r')

        self.actor_loss.clear()
        self.actor_loss.plot(self.actor_loss_data, pen='g')

        self.survival_time.clear()
        self.survival_time.plot(self.survival_time_data, pen='b')

    def reset(self):
        self.critic_loss_data.fill(np.nan)
        self.actor_loss_data.fill(np.nan)
        self.survival_time_data.fill(np.nan)
        self.critic_loss.clear()
        self.actor_loss.clear()
        self.survival_time.clear()

class Inside_Episode(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Inside Episode Visualization")
        self.setGeometry(100, 100, 1200, 800)                    # (x, y, width, height)
 
        # GraphicsLayoutWidget to hold multiple plots in a grid
        self.layout = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.layout)

        self.state_plot = self.layout.addPlot(row=0, col=0, title="State")
        self.state_plot.setLabel('left', 'Angular Velocity')
        self.state_plot.setLabel('bottom', 'Time Step')
        self.state_plot.showGrid(x=True, y=True)

        self.action_prob_plot = self.layout.addPlot(row=1, col=0, title="Action Probabilities")
        self.action_prob_plot.setLabel('left', 'Probability')
        self.action_prob_plot.setLabel('bottom', 'Action')
        self.action_prob_plot.showGrid(x=True, y=True)

        self.action_plot = self.layout.addPlot(row=2, col=0, title="Actions Taken")
        self.action_plot.setLabel('left', 'Action')
        self.action_plot.setLabel('bottom', 'Time Step')
        self.action_plot.showGrid(x=True, y=True)

        self.signal_size = 100
        self.action_prob_data = np.full((self.signal_size, 2), np.nan)
        self.state_data = np.full((self.signal_size,), np.nan)
        self.action_data = np.full((self.signal_size,), np.nan)

    def update(self, action_probs,state,action):
        self.action_prob_data = np.roll(self.action_prob_data, -1)
        self.action_prob_data[-1] = action_probs
        self.action_prob_plot.clear()
        self.action_prob_plot.plot(self.action_prob_data[:,0], pen='r',name='Left push')
        self.action_prob_plot.plot(self.action_prob_data[:,1], pen='b',name='Right push')

        self.state_data = np.roll(self.state_data, -1)
        self.state_data[-1] = state[3]                 # angular velocity
        self.state_plot.clear()
        self.state_plot.plot(self.state_data, pen='g', name='angular velocity')

        self.action_data = np.roll(self.action_data, -1)
        self.action_data[-1] = action
        self.action_plot.clear()
        self.action_plot.plot(self.action_data, pen='b', name='Action Taken')

    def reset(self):
        self.action_prob_plot.clear()