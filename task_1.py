import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

def get_derivatives(x1, x2, alpha):
    """
    andronov hopf change name
    """
    return alpha * x1 - x2 - x1 * (x1 ** 2 + x2 ** 2), x1 + alpha * x2 - x2 * (x1 ** 2 + x2 ** 2)

def generate_dataset(alpha, delta_t, range_start, range_end, t_start, t_end, num_trajectories):
    """
    Generates trajectory dataset from Andronov Hopf.
    :param alpha: alpha parameter in andronov hopf (see lecture slides)
    :param delta_t: time step size
    :param (range_start, range_end): range of the initial points
    :param (t_start, t_end): evaluation are done in time set [t_start, t_end), with time step delta_t
    :param num_trajectories: number of trajectories to generate (dataset size)
    :returns: solution as numpy array in shape (num_trajectories, 2, num_datapoints)
              where num_datapoints depends on t_start, t_end and delta_t.
              e.g. for (0,1) = (t_start, t_end) and delta_t = 0.01, we would get 100 datapoints 
                   for each trajectory with start time 0 and end time 0.99.
    """
    # randomly generate starting points, andronov hopf is 2-dimensional
    start_positions = np.random.uniform(range_start, range_end, (num_trajectories, 2))
    # linear spaced time set where we want to evaluate the function
    t_eval = np.arange(t_start, t_end, delta_t)
    
    # compute the solution set
    sols = []
    for start_position in start_positions:
        sol = scipy.integrate.solve_ivp(lambda t, y: get_derivatives(y[0], y[1], alpha), 
                                        (t_start - 1, t_end + 1) , start_position, t_eval=t_eval)
        # we only need the solution vectors
        sols.append(sol.y)
    
    return np.array(sols)

def reshape_for_training(raw_data):
    """
    Prepares the raw data for training by creating the corresponding targets and performing some reshaping.
    :param raw_data: numpy array of shape (num_trajectories, 2, num_datapoints)
    :returns: The prepared dataset as numpy array of shape (num_trajectories * num_datapoints, 2, 2) 
    """
    train_dataset_values = raw_data
    
    # We delete all the first datapoints from the targets since we cannot use them
    # in the training (Euler method only looks at the previous values so the initial datapoint
    # of a projectory is never a target for another datapoint.)
    # Same idea also applies to the last datapoint of the values, it has not target value so 
    # we have to ignore it as well.
    train_dataset_targets = np.delete(train_dataset_values, 0, axis=-1)
    train_dataset_values = np.delete(train_dataset_values, -1, axis=-1)
    
    # At this point, both target and value sets have the shape (num_trajectories, 2, num_datapoints - 1)

    # Combine all the trajectories into a single axis
    train_dataset_values = np.moveaxis(train_dataset_values, 1, -1).reshape((train_dataset_values.shape[0] * 
                                                                             train_dataset_values.shape[2], 2))
    train_dataset_targets = np.moveaxis(train_dataset_targets, 1,-1).reshape((train_dataset_targets.shape[0] *
                                                                              train_dataset_targets.shape[2], 2))

    # At this point, both target and value sets have the shape (num_trajectories * (num_datapoints - 1), 2)
    
    #print("train_dataset_targets shape: ", train_dataset_targets.shape)

    # Finally we combine value and target sets to get a single dataset 
    train_dataset = np.stack((train_dataset_values, train_dataset_targets))
    train_dataset = np.moveaxis(train_dataset, 0, 1)

    # Our final dataset should have the shape (num_trajectories * (num_datapoints - 1), 2, 2)
    
    return train_dataset


def plot_trajectory(trajectory, t_start, t_end, delta_t, save_figure=False):
    """
    Plots the given trajectory in 3d. 
    :param trajectory: Trajectory set in 2d. Should have the shape (2, num_datapoints)
    :param (t_start, t_end): evaluation are done in time set [t_start, t_end), with time step delta_t
    :param delta_t: time step size
    """
    
    evaluation_times = np.arange(t_start, t_end, delta_t)
    assert(len(trajectory[0]) == len(trajectory[1]) == len(evaluation_times))
    
    # plot both trajectories in 3D
    fig = plt.figure(figsize=(10, 10))
    ax0 = plt.axes(projection="3d")
    
    ax0.plot(evaluation_times, trajectory[0], trajectory[1], color="r")
    label = "Trajectory with starting point (" + str(np.format_float_scientific(trajectory[0][0], precision=3)) + ", " + str(np.format_float_scientific(trajectory[1][0], precision=3)) + ")"
    ax0.set_title(label)
    ax0.set_xlabel(r"$t$")
    ax0.set_ylabel(r"$x_1$")
    ax0.set_zlabel(r"$x_2$")