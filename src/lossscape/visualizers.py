# src/lossscape/visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, RadioButtons, Button

class Visualizer:
    """Animate gradient descent on a 2D loss surface with interactive controls."""
    def __init__(self, surface, optimizer, surfaces=None, init_point=None, max_iters=200):
        self.surface = surface
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.surfaces = surfaces  # Store all surfaces for radio button switching
        self.init_point = np.array([-3.0, 3.0]) if init_point is None else np.array(init_point)

        # History
        self.x = self.init_point.copy()
        self.traj = [self.x.copy()]
        self.losses = [self.surface.func(self.x)]
        self.grad_norms = [np.linalg.norm(self.surface.grad(self.x))]

        # Figure setup with 3 subplots
        self.fig, (self.ax_surface, self.ax_loss, self.ax_grad) = plt.subplots(
            1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 0.5, 0.5]}
        )
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, wspace=0.3)
        
        self._make_mesh()
        
        # Point and path on surface
        self.point, = self.ax_surface.plot([self.x[0]], [self.x[1]], 'ro')
        self.path_line, = self.ax_surface.plot([self.x[0]], [self.x[1]], 'w-', linewidth=1.5)

        # Loss plot
        self.ax_loss.set_title('Loss over iterations')
        self.ax_loss.set_xlabel('Iteration')
        self.ax_loss.set_ylabel('Loss')
        self.loss_line, = self.ax_loss.plot(self.losses, '-', color='blue')

        # Grad norm plot
        self.ax_grad.set_title('Gradient norm')
        self.ax_grad.set_xlabel('Iteration')
        self.ax_grad.set_ylabel('||grad||')
        self.grad_line, = self.ax_grad.plot(self.grad_norms, '-', color='red')

        # Learning rate slider
        ax_lr = plt.axes([0.1, 0.1, 0.3, 0.03], facecolor='lightgoldenrodyellow')
        self.lr_slider = Slider(ax_lr, 'LR', 0.001, 1.0, valinit=self.optimizer.lr)
        self.lr_slider.on_changed(self.on_lr_change)

        # Optimizer selection
        ax_opt = plt.axes([0.5, 0.05, 0.15, 0.15], facecolor='lightgoldenrodyellow')
        self.radio_opt = RadioButtons(ax_opt, list(self.available_optimizers().keys()))
        self.radio_opt.on_clicked(self.on_optimizer_change)

        # Surface selection
        ax_surf = plt.axes([0.75, 0.05, 0.2, 0.15], facecolor='lightgoldenrodyellow')
        surf_list = self.surfaces if self.surfaces else self.available_surfaces()
        self.radio_surf = RadioButtons(ax_surf, [s.name for s in surf_list])
        self.radio_surf.on_clicked(self.on_surface_change)

        # Pause button
        ax_pause = plt.axes([0.1, 0.02, 0.1, 0.05])
        self.button_pause = Button(ax_pause, 'Pause')
        self.button_pause.on_clicked(self.on_pause)

        # Reset button
        ax_reset = plt.axes([0.25, 0.02, 0.1, 0.05])
        self.button_reset = Button(ax_reset, 'Reset')
        self.button_reset.on_clicked(self.on_reset)

        # Animation control
        self.paused = False
        self.anim = None

    def _make_mesh(self):
        lim = 4
        xs = np.linspace(-lim, lim, 200)
        ys = np.linspace(-lim, lim, 200)
        X, Y = np.meshgrid(xs, ys)
        Z = self.surface.func((X, Y))
        self.ax_surface.clear()
        self.ax_surface.contourf(X, Y, Z, levels=50, cmap='viridis')
        self.ax_surface.contour(X, Y, Z, levels=20, colors='k', linewidths=0.3, alpha=0.5)
        self.ax_surface.set_title(self.surface.name)
        self.ax_surface.set_xlim(xs.min(), xs.max())
        self.ax_surface.set_ylim(ys.min(), ys.max())

    def available_surfaces(self):
        from lossscape.surfaces import SURFACES
        return SURFACES

    def available_optimizers(self):
        from lossscape.optimizers import OPTIMIZERS
        return OPTIMIZERS

    def on_lr_change(self, val):
        self.optimizer.lr = val

    def on_optimizer_change(self, label):
        from lossscape.optimizers import OPTIMIZERS
        self.optimizer = OPTIMIZERS[label](lr=self.optimizer.lr)

    def on_surface_change(self, label):
        surf_list = self.surfaces if self.surfaces else self.available_surfaces()
        for s in surf_list:
            if s.name == label:
                self.surface = s
                break
        # Reset trajectory
        self.x = np.array([-3.0, 3.0])
        self.traj = [self.x.copy()]
        self.losses = [self.surface.func(self.x)]
        self.grad_norms = [np.linalg.norm(self.surface.grad(self.x))]
        self.ax_surface.clear()
        self._make_mesh()
        self.point, = self.ax_surface.plot([self.x[0]], [self.x[1]], 'ro')
        self.path_line, = self.ax_surface.plot([self.x[0]], [self.x[1]], 'w-', linewidth=1.5)
        # Reset loss and gradient plots
        self.loss_line.set_data([], [])
        self.grad_line.set_data([], [])

    def on_pause(self, event):
        self.paused = not self.paused

    def on_reset(self, event):
        # Reset position
        self.x = np.array([-3.0, 3.0])
        self.traj = [self.x.copy()]
        self.losses = [self.surface.func(self.x)]
        self.grad_norms = [np.linalg.norm(self.surface.grad(self.x))]
        
        # Reset lines
        self.point.set_data([self.x[0]], [self.x[1]])
        self.path_line.set_data([self.x[0]], [self.x[1]])
        self.loss_line.set_data([0], [self.losses[0]])
        self.grad_line.set_data([0], [self.grad_norms[0]])
        
        # Refresh plots
        self.ax_surface.figure.canvas.draw_idle()
        self.ax_loss.figure.canvas.draw_idle()
        self.ax_grad.figure.canvas.draw_idle()

    def _update(self, frame):
        if not self.paused:
            grad = self.surface.grad(self.x)
            self.x = self.optimizer.step(self.x, grad)

            self.traj.append(self.x.copy())
            self.losses.append(self.surface.func(self.x))
            self.grad_norms.append(np.linalg.norm(self.surface.grad(self.x)))

        xs = np.array(self.traj)[:,0]
        ys = np.array(self.traj)[:,1]

        # Update surface
        self.point.set_data([self.x[0]], [self.x[1]])
        self.path_line.set_data(xs, ys)

        # Update loss plot
        self.loss_line.set_data(range(len(self.losses)), self.losses)
        self.ax_loss.set_xlim(0, max(50, len(self.losses)))
        self.ax_loss.set_ylim(min(self.losses)-0.1, max(self.losses)+0.1)

        # Update grad norm plot
        self.grad_line.set_data(range(len(self.grad_norms)), self.grad_norms)
        self.ax_grad.set_xlim(0, max(50, len(self.grad_norms)))
        self.ax_grad.set_ylim(0, max(self.grad_norms)*1.1)

        return self.point, self.path_line, self.loss_line, self.grad_line

    def run(self):
        self.anim = FuncAnimation(self.fig, self._update, frames=self.max_iters, interval=50, blit=False)
        plt.show()
