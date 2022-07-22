import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


def andronov_hopf(x1, x2, alpha):
    return alpha * x1 - x2 - x1 * (x1 ** 2 + x2 ** 2), x1 + alpha * x2 - x2 * (x1 ** 2 + x2 ** 2)


def create_dataset(alpha, t_eval, num_samples):
    start_positions = np.random.uniform(-2, 2, (num_samples, 2))
    sols = []

    for start_position in start_positions:
        sol = scipy.integrate.solve_ivp(lambda t, y: andronov_hopf(y[0], y[1], alpha), (0, 10), start_position,
                                        t_eval=t_eval)
        sols.append(sol["y"])

    return np.array(sols)


def reshape_dataset(dataset):
    dataset_values = np.delete(dataset, -1, axis=-1)
    dataset_targets = np.delete(dataset, 0, axis=-1)

    dataset_values = np.moveaxis(dataset_values, 1, -1).reshape((dataset_values.shape[0] * dataset_values.shape[2], 2))
    dataset_targets = np.moveaxis(dataset_targets, 1, -1).reshape((dataset_targets.shape[0] * dataset_targets.shape[2], 2))

    dataset = np.stack((dataset_values, dataset_targets))
    dataset = np.moveaxis(dataset, 0, 1)
    return dataset


def split_dataset(dataset, train_size, val_size, test_size):
    training_set = np.array(dataset[:math.floor(train_size*len(dataset)), :, :])
    validation_set = np.array(dataset[math.ceil(train_size*len(dataset)):math.ceil(train_size*len(dataset))+math.floor(val_size*len(dataset))])
    test_set = np.array(dataset[math.ceil(val_size*len(dataset)):math.ceil(val_size*len(dataset))+math.floor(test_size*len(dataset))])

    return training_set, validation_set, test_set


def evaluate_model(model, dataset, hparams, device):
    """
    TODO
    """
    model.eval()
    model.to(device)
    criterion = torch.nn.MSELoss()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=hparams["batch_size"], shuffle=False)
    loss = 0
    for batch in dataloader:
        pos = batch[:, 0]
        pos_target = batch[:, 1]

        pred = model.forward(pos.float()).to(device)

        loss += criterion(pred, pos_target.float()).item()
    return 1.0 / (2 * (loss / len(dataloader)))


def recreate_trajectory(model, start_position, t_start, t_end, delta_t, device):
    """
    TODO
    """
    trajectory = [start_position]
    last_traj = torch.tensor(start_position)
    last_traj = last_traj.to(device)
    t0 = t_start

    while t0 < t_end - delta_t:
        last_traj = model.forward(last_traj.float())
        trajectory.append(last_traj.detach().numpy())
        t0 += delta_t

    return trajectory


def bok_püsür(sols, t_eval):
    # plot both trajectories in 3D
    fig = plt.figure(figsize=(10, 10))
    ax0 = plt.axes(projection="3d")
    ax0.plot(t_eval, sols[0][0], sols[0][1], label=r"Trajectory with starting point $(-3, -3)$", color="r")

    ax0.set_xlabel(r"$t$")
    ax0.set_ylabel(r"$x_1$")
    ax0.set_zlabel(r"$x_2$")

    print("FIRST VALUE: ", sols[0][0][0], sols[0][1][0])
    print("LAST VALUE: ", sols[0][0][-1], sols[0][1][-1])
