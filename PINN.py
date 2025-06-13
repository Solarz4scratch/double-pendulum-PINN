import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import matplotlib.animation as animation
import matplotlib

matplotlib.use('TkAgg')

# --- Get next file name ---
def get_next_filename(prefix="newest", label=None, extension="png"):
    i = 1
    while True:
        if label:
            filename = f"{prefix}_{label}_{i}.{extension}"
        else:
            filename = f"{prefix}{i}.{extension}"
        if not os.path.exists(filename):
            return filename
        i += 1

# --- Physics constants (SI units) ---
g = 9.81
l1 = 1.0
l2 = 1.0
m1 = 1.0
m2 = 1.0

# --- Double pendulum ordinary differential equation (ODE) system ---
def double_pendulum_ode(t, y):
    theta1, theta2, theta1_dot, theta2_dot = y
    delta = theta2 - theta1
    denom1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta) ** 2
    denom2 = l2 * denom1

    theta1_ddot = (
        m2 * l1 * theta1_dot ** 2 * np.sin(delta) * np.cos(delta) +
        m2 * g * np.sin(theta2) * np.cos(delta) +
        m2 * l2 * theta2_dot ** 2 * np.sin(delta) -
        (m1 + m2) * g * np.sin(theta1)
    ) / denom1

    theta2_ddot = (
        -m2 * l2 * theta2_dot ** 2 * np.sin(delta) * np.cos(delta) +
        (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
        (m1 + m2) * l1 * theta1_dot ** 2 * np.sin(delta) -
        (m1 + m2) * g * np.sin(theta2)
    ) / denom2

    return [theta1_dot, theta2_dot, theta1_ddot, theta2_ddot]

# --- Initial conditions ---
# Angles are measured from the vertical
theta1_0 = np.pi / 2
theta2_0 = np.pi / 2
theta1_dot_0 = 0.0
theta2_dot_0 = 0.0
y0 = [theta1_0, theta2_0, theta1_dot_0, theta2_dot_0]

# --- Time points ---
t_min, t_max = 0.0, 10.0
N_t = 1000 # Number of time points/steps in interval
t_span = (t_min, t_max)
t_eval = np.linspace(t_min, t_max, N_t) # Create the sequence of time values

# --- Solve ODE for ground truth ---
sol = solve_ivp(
    double_pendulum_ode,
    (t_min, t_max),
    y0,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-9,
    atol=1e-12
)

# --- PINN Model ---
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256),  # wider layer
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),  # additional layer
            nn.SiLU(),
            nn.Linear(128, 2)
        )

    def forward(self, t):
        return self.net(t)

# --- Physics computation functions ---
# Compute mechanical energy in the system (kinetic + potential)
def compute_energies(theta1, theta2, theta1_dot, theta2_dot):
    x1_dot = l1 * theta1_dot * torch.cos(theta1)
    y1_dot = l1 * theta1_dot * torch.sin(theta1)
    x2_dot = x1_dot + l2 * theta2_dot * torch.cos(theta2)
    y2_dot = y1_dot + l2 * theta2_dot * torch.sin(theta2)
    T1 = 0.5 * m1 * (x1_dot**2 + y1_dot**2)
    T2 = 0.5 * m2 * (x2_dot**2 + y2_dot**2)
    T = T1 + T2
    y1 = -l1 * torch.cos(theta1) # 0 potential energy at horizontal
    y2 = y1 - l2 * torch.cos(theta2)
    V = m1 * g * y1 + m2 * g * y2
    return T + V

# Compute square of difference between starting mechanical energy & current mechanical energy
def energy_residual_loss(model, t):
    t.requires_grad_(True)
    theta = model(t)
    theta1 = theta[:, 0:1]
    theta2 = theta[:, 1:2]

    theta1_dot = torch.autograd.grad(theta1, t, torch.ones_like(theta1), create_graph=True)[0]
    theta2_dot = torch.autograd.grad(theta2, t, torch.ones_like(theta2), create_graph=True)[0]

    E_total = compute_energies(theta1, theta2, theta1_dot, theta2_dot)
    E0 = E_total[0].detach()
    energy_loss = ((E_total - E0) ** 2).mean()

    return energy_loss

# Physics-based ODE residual loss
def ode_residual_loss(model, t):
    t.requires_grad_(True)
    theta = model(t)
    theta1 = theta[:, 0:1]
    theta2 = theta[:, 1:2]

    theta1_dot = torch.autograd.grad(theta1, t, torch.ones_like(theta1), create_graph=True)[0]
    theta2_dot = torch.autograd.grad(theta2, t, torch.ones_like(theta2), create_graph=True)[0]
    theta1_ddot = torch.autograd.grad(theta1_dot, t, torch.ones_like(theta1_dot), create_graph=True)[0]
    theta2_ddot = torch.autograd.grad(theta2_dot, t, torch.ones_like(theta2_dot), create_graph=True)[0]

    delta = theta2 - theta1
    denom1 = (m1 + m2) * l1 - m2 * l1 * torch.cos(delta) ** 2
    denom2 = l2 * denom1

    eq1_num = (
        m2 * l1 * theta1_dot ** 2 * torch.sin(delta) * torch.cos(delta) +
        m2 * g * torch.sin(theta2) * torch.cos(delta) +
        m2 * l2 * theta2_dot ** 2 * torch.sin(delta) -
        (m1 + m2) * g * torch.sin(theta1)
    )
    eq1 = theta1_ddot - eq1_num / denom1

    eq2_num = (
        -m2 * l2 * theta2_dot ** 2 * torch.sin(delta) * torch.cos(delta) +
        (m1 + m2) * g * torch.sin(theta1) * torch.cos(delta) -
        (m1 + m2) * l1 * theta1_dot ** 2 * torch.sin(delta) -
        (m1 + m2) * g * torch.sin(theta2)
    )
    eq2 = theta2_ddot - eq2_num / denom2

    loss = (eq1 ** 2).mean() + (eq2 ** 2).mean()
    return loss

# Computes the "activity" in the model, which is higher when there is more movement/acceleration.
# Most likely not going to be used in the final model, since I feel like it's "cheating"
def activity_residual_loss(model, t):
    t.requires_grad_(True)
    theta = model(t)
    theta1 = theta[:, 0:1]
    theta2 = theta[:, 1:2]

    theta1_dot = torch.autograd.grad(theta1, t, torch.ones_like(theta1), create_graph=True)[0]
    theta2_dot = torch.autograd.grad(theta2, t, torch.ones_like(theta2), create_graph=True)[0]
    theta1_ddot = torch.autograd.grad(theta1_dot, t, torch.ones_like(theta1_dot), create_graph=True)[0]
    theta2_ddot = torch.autograd.grad(theta2_dot, t, torch.ones_like(theta2_dot), create_graph=True)[0]

    activity_loss = ((m1/l1*(theta1_dot ** 2 + 0.01*theta2_dot ** 2) + m2/(l1+l2)*(theta1_ddot ** 2 + 0.01*theta2_ddot ** 2)).mean()) ** (-0.5)

    return activity_loss

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # I don't have a gpu :pensive:
    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    t_data = torch.linspace(t_min, t_max, N_t).view(-1, 1).to(device)
    theta0 = torch.tensor([[theta1_0, theta2_0]], dtype=torch.float32).to(device)

    # Collects the losses over time to create a plot later
    ode_losses = []
    energy_losses = []
    activity_losses = []
    ic_losses = []


    ############################################################
    ############################################################
    energy_weight = 0.01 # TODO: FIND THE BEST VALUE FOR THIS HYPERPARAMETER
    ############################################################
    ############################################################

    print("Energy weight: ", energy_weight) # To keep track if I have multiple instances running

    # --- Training (FULL-BATCH) ---
    for epoch in range(5000):
        optimizer.zero_grad()

        # Compute the loss for the entire period of time
        # Note: All losses are computed for the entire dataset (time period)
        ode_loss = ode_residual_loss(model, t_data)
        energy_loss = energy_residual_loss(model, t_data)
        activity_loss = activity_residual_loss(model, t_data)

        # Compute predicted angles at t=0
        pred0 = model(torch.tensor([[0.0]], device=device))

        # Compute predicted derivatives at t=0
        t0 = torch.tensor([[0.0]], device=device, requires_grad=True)
        pred0 = model(t0)
        theta1_0_pred = pred0[:, 0:1]
        theta2_0_pred = pred0[:, 1:2]
        theta1_dot_0_pred = torch.autograd.grad(theta1_0_pred, t0, torch.ones_like(theta1_0_pred), create_graph=True)[0]
        theta2_dot_0_pred = torch.autograd.grad(theta2_0_pred, t0, torch.ones_like(theta2_0_pred), create_graph=True)[0]

        # --- Initial condition (IC) loss ---
        # True initial conditions
        theta0_true = torch.tensor([[theta1_0, theta2_0]], dtype=torch.float32, device=device)
        theta_dot0_true = torch.tensor([[theta1_dot_0, theta2_dot_0]], dtype=torch.float32, device=device)

        # Combine position and velocity IC losses
        pos_loss = ((pred0 - theta0_true) ** 2).mean()
        vel_loss = ((torch.cat([theta1_dot_0_pred, theta2_dot_0_pred], dim=1) - theta_dot0_true) ** 2).mean()

        ic_loss = pos_loss + vel_loss

        # --- Compute total loss ---
        loss = ode_loss + energy_weight * energy_loss + 100.0 * ic_loss
        loss.backward()
        optimizer.step()

        # --- Track/log losses ---
        ode_losses.append(ode_loss.item())
        energy_losses.append(energy_loss.item())
        activity_losses.append(activity_loss.item())
        ic_losses.append(ic_loss.item())

        # Give occasional updates
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: ODE Loss={ode_losses[-1]:.6f} | Energy Loss={energy_losses[-1]:.6f} | "
                  f"Activity Loss={activity_losses[-1]:.6f} | IC Loss={ic_losses[-1]:.6f}")

    # --- Plot loss evolution from training ---
    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
    ax_loss.plot(ode_losses, label="ODE Residual Loss")
    ax_loss.plot(energy_losses, label="Energy Residual Loss")
    ax_loss.plot(ic_losses, label="Initial Condition Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    ax_loss.legend()
    ax_loss.set_title("Loss Evolution During Training")
    ax_loss.grid(True)

    # --- ODE residual loss over time (ground truth vs PINN) ---
    t_tensor = torch.tensor(t_eval, dtype=torch.float32, device=device, requires_grad=True).view(-1, 1)
    theta_pred = model(t_tensor)
    theta1_pred = theta_pred[:, 0:1]
    theta2_pred = theta_pred[:, 1:2]

    theta1_dot = torch.autograd.grad(theta1_pred, t_tensor, torch.ones_like(theta1_pred), create_graph=True)[0]
    theta2_dot = torch.autograd.grad(theta2_pred, t_tensor, torch.ones_like(theta2_pred), create_graph=True)[0]
    theta1_ddot = torch.autograd.grad(theta1_dot, t_tensor, torch.ones_like(theta1_dot), create_graph=True)[0]
    theta2_ddot = torch.autograd.grad(theta2_dot, t_tensor, torch.ones_like(theta2_dot), create_graph=True)[0]

    delta = theta2_pred - theta1_pred
    denom1 = (m1 + m2) * l1 - m2 * l1 * torch.cos(delta) ** 2

    eq1 = ((m1 + m2) * l1 * theta1_ddot +
           m2 * l2 * theta2_ddot * torch.cos(delta) -
           m2 * l2 * theta2_dot ** 2 * torch.sin(delta) -
           (m1 + m2) * g * torch.sin(theta1_pred))

    eq2 = (l2 * theta2_ddot +
           l1 * theta1_ddot * torch.cos(delta) -
           l1 * theta1_dot ** 2 * torch.sin(delta) -
           g * torch.sin(theta2_pred))

    res1 = eq1 / denom1
    res2 = eq2 / l2

    ode_residual_pinn = (res1 ** 2 + res2 ** 2).squeeze().cpu().detach().numpy()

    # --- Compute ground truth total energy ---
    theta1_true = sol.y[0]
    theta2_true = sol.y[1]

    theta1_true_tensor = torch.tensor(theta1_true, dtype=torch.float32, device=device).view(-1, 1)
    theta2_true_tensor = torch.tensor(theta2_true, dtype=torch.float32, device=device).view(-1, 1)

    theta1_true_dot = torch.gradient(theta1_true_tensor.squeeze(), spacing=(t_eval[1] - t_eval[0]))[0].view(-1, 1)
    theta2_true_dot = torch.gradient(theta2_true_tensor.squeeze(), spacing=(t_eval[1] - t_eval[0]))[0].view(-1, 1)

    E_true = compute_energies(theta1_true_tensor, theta2_true_tensor, theta1_true_dot, theta2_true_dot).cpu().detach().numpy()
    E_pinn = compute_energies(theta1_pred, theta2_pred, theta1_dot, theta2_dot).cpu().detach().numpy()

    # --- Plot total energy over time ---
    fig_energy, ax_energy = plt.subplots(figsize=(10, 5))
    ax_energy.plot(t_eval, E_true, label="Ground Truth Energy", color="blue", alpha=0.7)
    ax_energy.plot(t_eval, E_pinn, label="PINN Predicted Energy", color="red", linestyle="--")
    ax_energy.set_xlabel("Time (s)")
    ax_energy.set_ylabel("Total Energy (Joules)")
    ax_energy.set_title("Total Energy of the System Over Time")
    ax_energy.legend()
    ax_energy.grid(True)


    # --- Plot predictions vs true ---
    fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
    ax_pred.plot(t_eval, theta1_true, 'b', label='θ1 True', alpha=0.5)
    ax_pred.plot(t_eval, theta2_true, 'r', label='θ2 True', alpha=0.5)
    ax_pred.plot(t_eval, theta1_pred.cpu().detach().numpy(), 'b--', label='θ1 PINN', linewidth=2)
    ax_pred.plot(t_eval, theta2_pred.cpu().detach().numpy(), 'r--', label='θ2 PINN', linewidth=2)
    ax_pred.set_xlabel('Time (s)')
    ax_pred.set_ylabel('Angle (rad)')
    ax_pred.legend()
    ax_pred.grid(True)
    ax_pred.set_title('Double Pendulum: True vs PINN Trajectories')

    # --- Animation setup ---
    fig_anim, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_xlim(-2.2, 2.2)
    ax1.set_ylim(-2.2, 2.2)
    ax2.set_xlim(-2.2, 2.2)
    ax2.set_ylim(-2.2, 2.2)
    ax1.set_title("True Motion (ODE Solver)")
    ax2.set_title("PINN Predicted Motion | Energy weight: " + str(energy_weight))

    line1_true, = ax1.plot([], [], 'o-', lw=2, color='blue')
    line2_true, = ax1.plot([], [], 'o-', lw=2, color='red')
    line1_pinn, = ax2.plot([], [], 'o-', lw=2, color='blue')
    line2_pinn, = ax2.plot([], [], 'o-', lw=2, color='red')

    # --- Initialize function ---
    def init():
        line1_true.set_data([], [])
        line2_true.set_data([], [])
        line1_pinn.set_data([], [])
        line2_pinn.set_data([], [])
        return line1_true, line2_true, line1_pinn, line2_pinn

    # --- Animation update function ---
    def animate(i):
        # ODE True Data (assumed to be numpy arrays already)
        x1_true = l1 * np.sin(theta1_true[i])
        y1_true = -l1 * np.cos(theta1_true[i])
        x2_true = x1_true + l2 * np.sin(theta2_true[i])
        y2_true = y1_true - l2 * np.cos(theta2_true[i])

        line1_true.set_data([0, x1_true], [0, y1_true])
        line2_true.set_data([x1_true, x2_true], [y1_true, y2_true])

        # Convert tensors to numpy for PINN data
        theta1_i = theta1_pred[i].detach().cpu().numpy().item()
        theta2_i = theta2_pred[i].detach().cpu().numpy().item()

        x1_pinn = l1 * np.sin(theta1_i)
        y1_pinn = -l1 * np.cos(theta1_i)
        x2_pinn = x1_pinn + l2 * np.sin(theta2_i)
        y2_pinn = y1_pinn - l2 * np.cos(theta2_i)

        line1_pinn.set_data([0, x1_pinn], [0, y1_pinn])
        line2_pinn.set_data([x1_pinn, x2_pinn], [y1_pinn, y2_pinn])

        return line1_true, line2_true, line1_pinn, line2_pinn


    # --- Animation configuration ---
    fps = 30  # Desired frames per second

    # Determine duration of animation based on t_max
    t_max = t_eval[-1]
    duration = t_max

    # Determine number of frames
    total_frames = int(duration * fps)

    # Get evenly spaced frame indices matching the dataset length
    frame_indices = np.linspace(0, len(t_eval)-1, total_frames, dtype=int)

    # --- Create animation ---
    ani = animation.FuncAnimation(
        fig_anim,
        animate,
        frames=frame_indices,
        interval=1000 / fps,  # interval in milliseconds
        blit=True,
        init_func=init
    )

    plt.show()

    # --- Save plots and figures ---
    save_data = input("Save figures and animation as GIF? (y/n): ").strip().lower()

    if save_data == 'y':
        label = input("Label: ")

        # Save loss evolution plot
        filename_loss = get_next_filename(prefix="", label=label, extension="png")
        fig_loss.savefig(filename_loss, dpi=300)
        print(f"Loss evolution plot saved as {filename_loss}")

        # Save predictions vs true plot
        filename_pred = get_next_filename(prefix="Ode+Energy_Predictions", label=label, extension="png")
        fig_pred.savefig(filename_pred, dpi=300)
        print(f"Prediction plot saved as {filename_pred}")

        # Save animation
        gif_filename = get_next_filename(prefix="Ode+Energy_Animation", label=label, extension="gif")
        ani.save(gif_filename, fps=15)
        print(f"Animation saved as {gif_filename}")

    plt.close('all')