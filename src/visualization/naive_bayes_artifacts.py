import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.models.generative_logistic import get_logistic_regression_params
from src.models.naive_bayes import get_prior
from src.models.naive_bayes import predict_class
from src.training.optim import learn_stdevs
from src.visualization.plots import contour_plot
from src.visualization.plots import get_meshgrid
from src.visualization.plots import plot_data

LABELS = {0: "Iris-Setosa", 1: "Iris-Versicolour", 2: "Iris-Virginica"}
LABEL_COLOURS = ["blue", "orange", "green"]
LABELS_BINARY = {0: "Iris-Setosa", 1: "Iris-Versicolour / Iris-Virginica"}
LABEL_COLOURS_BINARY = ["blue", "red"]


def generate_naive_bayes_experiment_figures(
    x_train,
    y_train,
    x_test,
    y_test,
    prior,
    class_conditionals,
    figures_dir,
    binary_epochs=50,
    seed=42,
):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    figures_dir.mkdir(parents=True, exist_ok=True)

    _save_training_scatter(x_train, y_train, figures_dir)
    _save_class_conditional_contours(x_train, y_train, class_conditionals, figures_dir)
    predictions_multiclass = _save_decision_regions(
        x_train,
        y_train,
        prior,
        class_conditionals,
        figures_dir,
    )

    y_train_binary = np.array(y_train)
    y_train_binary[np.where(y_train_binary == 2)] = 1
    y_test_binary = np.array(y_test)
    y_test_binary[np.where(y_test_binary == 2)] = 1

    prior_binary = get_prior(y_train_binary)
    scales = tf.Variable([1.0, 1.0], dtype=tf.float32)
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.01)

    nlls, scales_arr, class_conditionals_binary = learn_stdevs(
        x_train,
        y_train_binary,
        scales,
        optimiser,
        epochs=int(binary_epochs),
    )
    predictions_binary = predict_class(prior_binary, class_conditionals_binary, x_test)
    accuracy_binary = float(np.mean(predictions_binary == y_test_binary))

    _save_binary_loss_scales(nlls, scales_arr, figures_dir)
    _save_logistic_contours(
        x_train,
        y_train_binary,
        prior_binary,
        class_conditionals_binary,
        figures_dir,
    )

    return {
        "multiclass_predictions": predictions_multiclass,
        "binary_predictions": predictions_binary,
        "binary_accuracy": accuracy_binary,
        "binary_nlls": nlls,
        "binary_scales": scales_arr,
    }


def _save_training_scatter(x_train, y_train, figures_dir):
    plt.figure(figsize=(8, 5))
    plot_data(x_train, y_train, LABELS, LABEL_COLOURS)
    plt.tight_layout()
    plt.savefig(figures_dir / "01_scatter_training.png", dpi=150)
    plt.close()


def _save_class_conditional_contours(x_train, y_train, class_conditionals, figures_dir):
    plt.figure(figsize=(10, 6))
    plot_data(x_train, y_train, LABELS, LABEL_COLOURS)
    x0_min, x0_max = float(x_train[:, 0].min()), float(x_train[:, 0].max())
    x1_min, x1_max = float(x_train[:, 1].min()), float(x_train[:, 1].max())
    contour_plot(
        (x0_min, x0_max),
        (x1_min, x1_max),
        class_conditionals.prob,
        3,
        LABEL_COLOURS,
    )
    plt.title("Training set with class-conditional density contours")
    plt.tight_layout()
    plt.savefig(figures_dir / "02_class_conditionals_contours.png", dpi=150)
    plt.close()


def _save_decision_regions(x_train, y_train, prior, class_conditionals, figures_dir):
    plt.figure(figsize=(10, 6))
    plot_data(x_train, y_train, LABELS, LABEL_COLOURS)
    x0_min, x0_max = float(x_train[:, 0].min()), float(x_train[:, 0].max())
    x1_min, x1_max = float(x_train[:, 1].min()), float(x_train[:, 1].max())
    contour_plot(
        (x0_min, x0_max),
        (x1_min, x1_max),
        lambda points: predict_class(prior, class_conditionals, points),
        1,
        LABEL_COLOURS,
        levels=[-0.5, 0.5, 1.5, 2.5],
        num_points=500,
    )
    plt.title("Training set with decision regions")
    plt.tight_layout()
    plt.savefig(figures_dir / "03_decision_regions.png", dpi=150)
    plt.close()

    return predict_class(prior, class_conditionals, x_train)


def _save_binary_loss_scales(nlls, scales_arr, figures_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(nlls)
    axes[0].set_title("Loss vs epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Average negative log-likelihood")

    for index in [0, 1]:
        axes[1].plot(
            scales_arr[:, index],
            color=LABEL_COLOURS_BINARY[index],
            label=LABELS_BINARY[index],
        )
    axes[1].set_title("Standard deviation ML estimates vs epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Standard deviation")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(figures_dir / "04_binary_loss_scales.png", dpi=150)
    plt.close(fig)


def _save_logistic_contours(
    x_train,
    y_train_binary,
    prior_binary,
    class_conditionals_binary,
    figures_dir,
):
    weights, bias = get_logistic_regression_params(
        prior_binary, class_conditionals_binary
    )

    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    plot_data(x_train, y_train_binary, LABELS_BINARY, LABEL_COLOURS_BINARY)
    x0_min, x0_max = float(x_train[:, 0].min()), float(x_train[:, 0].max())
    x1_min, x1_max = float(x_train[:, 1].min()), float(x_train[:, 1].max())
    x0_mesh, x1_mesh = get_meshgrid((x0_min, x0_max), (x1_min, x1_max))

    logits = np.dot(np.array([x0_mesh.ravel(), x1_mesh.ravel()]).T, weights) + bias
    posterior = tf.math.sigmoid(logits)
    contour = axes.contour(
        x0_mesh,
        x1_mesh,
        np.array(posterior).reshape(*x0_mesh.shape),
        levels=10,
    )
    axes.clabel(contour, inline=True, fontsize=10)

    contour_plot(
        (x0_min, x0_max),
        (x1_min, x1_max),
        lambda points: predict_class(prior_binary, class_conditionals_binary, points),
        1,
        LABEL_COLOURS_BINARY,
        levels=[-0.5, 0.5, 1.5],
        num_points=300,
    )

    plt.title("Training set with logistic prediction contours")
    fig.tight_layout()
    fig.savefig(figures_dir / "05_logistic_regression_contours.png", dpi=150)
    plt.close(fig)
