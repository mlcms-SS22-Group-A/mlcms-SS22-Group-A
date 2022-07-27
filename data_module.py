import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings

from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


def andronov_hopf(x1, x2, alpha):
    """
    returns partial derivatives for andronov hopf system
    @param x1: first spatial coordinate
    @param x2: second spatial coordinate
    @param alpha: parameter of the system
    @return: (dx_1/dt, dx_2/dt)
    """
    return alpha * x1 - x2 - x1 * (x1 ** 2 + x2 ** 2), x1 + alpha * x2 - x2 * (x1 ** 2 + x2 ** 2)


def create_dataset(t_eval, num_samples):
    """
    creates dataset using solve_ivp method, saving num_samples
    @param t_eval: times at which the value of solve_ivp is saved
    @param num_samples: number of trajectories to be created
    @return: trajectories list
    """
    start_positions = np.random.uniform(-2, 2, (num_samples, 2))
    alphas = np.random.uniform(-2, 2, 50)
    sols = []
    sol_alphas = []

    # for each start position create trajectory using solve_ivp
    for idx, start_position in enumerate(start_positions):
        sol = solve_ivp(lambda t, y: andronov_hopf(y[0], y[1], alphas[idx % len(alphas)]), (0, 10), start_position,
                        t_eval=t_eval)
        sols.append(sol.y)
        sol_alphas.append([alphas[idx % len(alphas)] for _ in range(len(sol.y[0]))])

    return np.array(sols), np.array(sol_alphas)


def reshape_dataset(dataset, alphas):
    """
    reshape the dataset such that all data points are in a single dimension, targets are created using the state at
    next index
    @param dataset: dataset to be reshaped
    @param alphas:
    @return: reshaped dataset (first three value in each row is the inputs, remaining are the expected output
    """
    # remove first value of dataset to create target and last value of dataset to create the inputs
    dataset_values = np.delete(dataset, -1, axis=-1)
    dataset_targets = np.delete(dataset, 0, axis=-1)
    alphas = np.delete(alphas, -1, axis=-1).reshape(-1, 1)

    # reshape them so that trajectories are not in their own dimension
    dataset_values = np.moveaxis(dataset_values, 1, -1).reshape((dataset_values.shape[0] * dataset_values.shape[2], 2))
    dataset_values = np.c_[dataset_values, alphas]
    dataset_targets = np.moveaxis(dataset_targets, 1, -1).reshape(
        (dataset_targets.shape[0] * dataset_targets.shape[2], 2))

    # stack values and targets in a single array
    dataset = np.array(np.c_[dataset_values, dataset_targets])
    return dataset


def split_dataset(dataset, train_size, val_size, test_size):
    """
    split dataset into training, validation, and test sets
    @param dataset: dataset to be split
    @param train_size: percentage of train-set
    @param val_size: percentage of validation-set
    @param test_size: percentage of test-set
    @return: split sets
    """
    # split dataset into three parts
    training_set = np.array(dataset[:math.floor(train_size * len(dataset))])
    validation_set = np.array(dataset[
                              math.ceil(train_size * len(dataset)):math.ceil(train_size * len(dataset)) + math.floor(
                                  val_size * len(dataset))])
    test_set = np.array(dataset[math.ceil(val_size * len(dataset)):math.ceil(val_size * len(dataset)) + math.floor(
        test_size * len(dataset))])

    return training_set, validation_set, test_set


def evaluate_model(model, dataset, hparams, device):
    """
    evaluate the model using the following formula: score = 1 / (2 * MSELoss)
    @param model: nn model
    @param dataset: dataset on which model will be evaluated
    @param hparams: hyper parameters of the model
    @param device: gpu or cpu
    @return: score of the model
    """
    model.eval()
    model.to(device)
    criterion = torch.nn.MSELoss()
    dataloader = DataLoader(dataset, batch_size=hparams["batch_size"], shuffle=False)
    loss = 0
    # for each batch cumulate loss
    for batch in dataloader:
        pos = batch[:, :3].to(device)
        pos_target = batch[:, 3:].to(device)

        pred = pos.float()[:, :2] + hparams["delta_t"] * model.forward(pos.float()).to(device)

        loss += criterion(pred, pos_target.float()).item()
    # divide loss by the number of batches, then multiply with a scaling factor to acquire score
    return 1.0 / (2 * (loss / len(dataloader)))


def recreate_trajectory(model, start_position, t_start, t_end, delta_t, device):
    """
    recreate a trajectory given the starting position and timespan using the nn model
    @param model: nn model
    @param start_position: starting position of the trajectory
    @param t_start: start time
    @param t_end: end time
    @param delta_t: timestep size
    @param device: gpu or cpu
    @return: created trajectory
    """
    trajectory = [start_position]
    last_traj = torch.tensor([start_position])
    last_traj = last_traj.to(device)
    t0 = t_start

    while t0 < t_end:
        # model.forward() outputs the derivative at current state
        last_traj[:, :2] = last_traj[:, :2] + delta_t * model.forward(last_traj.float())
        trajectory.append(np.array(last_traj.detach().numpy().reshape(-1)))
        t0 = round(t0 + delta_t, 2)

    return np.array(trajectory)


def plot_trajectory(sol, t_eval, fig_title, save_fig=False, figure_save_path=""):
    """
    plots a given trajectory in a given timespan
    @param sol: trajectory to be plotted
    @param t_eval: timespan
    @param fig_title: title of the figure
    @param save_fig: boolean, if set to True, figure will be saved
    @param figure_save_path: save path for figure
    """
    # plot the given trajectory
    fig = plt.figure(figsize=(10, 10))
    ax0 = plt.axes(projection="3d")
    ax0.plot(t_eval, sol[:, 0], sol[:, 1], color="r")

    ax0.set_title(fig_title, fontsize=24)
    ax0.set_xlabel(r"$t$", fontsize=20)
    ax0.set_ylabel(r"$x_1$", fontsize=20)
    ax0.set_zlabel(r"$x_2$", fontsize=20)

    if save_fig:
        fig.savefig(figure_save_path)


def compute_and_plot_phase_portrait(model, alpha, fig_title, save_fig=False, figure_save_path=""):
    """
    computes and plots phase portrait
    @param model: nn model
    @param alpha: parameter of dynamical system
    @param fig_title: title of the figure
    @param save_fig: boolean, if set to True, figure will be saved
    @param figure_save_path: save path for figure
    """
    # get derivatives at discrete locations using nn model
    positions = np.empty((40, 40, 2))
    derivatives = np.empty((40, 40, 2))
    for idx1, i in enumerate(np.linspace(-2, 2, 40)):
        for idx2, j in enumerate(np.linspace(-2, 2, 40)):
            positions[idx1][idx2] = np.array((i, j))
            position = torch.tensor([[i, j, alpha]])
            derivatives[idx1][idx2] = (model.forward(position.float())).detach().numpy()

    # plot phase portrait
    fig = plt.figure(figsize=(25, 25))
    ax = fig.add_subplot()
    ax.quiver(positions[:, :, 0], positions[:, :, 1], derivatives[:, :, 0], derivatives[:, :, 1], units="xy", scale=7.5)
    ax.set_title(fig_title, fontsize=24)
    ax.set_xlabel(r"$x_1$", fontsize=20)
    ax.set_ylabel(r"$x_2$", fontsize=20)

    if save_fig:
        fig.savefig(figure_save_path)


def compute_bifurcation_diagram(model, delta_t):
    """
    computes steady states of the system over a range of alpha values by starting trajectories at random positions and
    letting them converge (update of the positions over time is made by trained nn models)
    @param model: nn model used in the integration
    @param delta_t: timestep of integration
    @return alphas: range of alpha values
            steady_states: set of steady state for each alpha value in alphas list
    """
    alphas = np.linspace(-2, 2, 41)
    steady_states = []
    for alpha in alphas:
        steady_states_current_alpha = []
        for idx1, i in enumerate(np.linspace(-2, 2, 10)):
            for idx2, j in enumerate(np.linspace(-2, 2, 10)):
                position = torch.tensor([[i, j, alpha]])
                counter = 0
                while True and (counter < 100):
                    # predict the derivative at current position
                    derivative = model.forward(position.float())
                    # convergence check
                    if np.linalg.norm(np.array(derivative.detach().numpy().reshape(-1))) < 1e-3:
                        break
                    position[:, :2] += delta_t * derivative
                    counter += 1
                steady_states_current_alpha.append(np.round(position[:, :2].detach().numpy().reshape(-1), 1).tolist())
        # we cumulate steady states in a set to avoid multiple occurrence of a steady state multiple times
        steady_states.append(np.array(list(set(tuple(x) for x in steady_states_current_alpha))))

    return alphas, np.array(steady_states)


def plot_bifurcation_diagram(alphas, steady_states, fig_title, save_fig=False, fig_save_path=""):
    """
    plots the computed bifurcation diagram
    @param alphas: values at which steady states are calculated
    @param steady_states: steady states of the system
    @param fig_title: title of the figure
    @param save_fig: boolean, if set to True, figure will be saved
    @param fig_save_path: save path for figure
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    for x, y in zip(alphas, steady_states):
        ax.scatter([x] * len(y), y[:, 0], y[:, 1], c="b")

    ax.set_title(fig_title, fontsize=24)
    ax.set_xlabel(r"$\alpha$", fontsize=20)
    ax.set_ylabel(r"$x_1$", fontsize=20)
    ax.set_zlabel(r"$x_2$", fontsize=20)

    if save_fig:
        fig.savefig(fig_save_path)
