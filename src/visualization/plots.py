import numpy as np
import matplotlib.pyplot as plt


def plot_data(x, y, labels, colours, title="Training set"):
    for class_id in np.unique(y):
        inx = np.where(y == class_id)
        plt.scatter(x[inx, 0], x[inx, 1], label=labels[class_id], c=colours[class_id])
    plt.title(title)
    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Sepal width (cm)")
    plt.legend()


def get_meshgrid(x0_range, x1_range, num_points=100):
    x0 = np.linspace(x0_range[0], x0_range[1], num_points)
    x1 = np.linspace(x1_range[0], x1_range[1], num_points)
    return np.meshgrid(x0, x1)


def contour_plot(
    x0_range, x1_range, prob_fn, batch_shape, colours, levels=None, num_points=100
):
    x0, x1 = get_meshgrid(x0_range, x1_range, num_points=num_points)
    points = np.expand_dims(np.array([x0.ravel(), x1.ravel()]).T, 1)
    z = prob_fn(points)
    z = np.array(z).T.reshape(batch_shape, *x0.shape)
    for batch in np.arange(batch_shape):
        if levels:
            plt.contourf(x0, x1, z[batch], alpha=0.2, colors=colours, levels=levels)
        else:
            plt.contour(x0, x1, z[batch], colors=colours[batch], alpha=0.3)
