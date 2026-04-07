import numpy as np
import matplotlib.pyplot as plt


def generate_cnn_experiment_figures(
    model,
    model_name,
    x_train,
    x_test,
    y_test,
    x_c_test,
    y_c_test,
    figures_dir,
    prediction_indices_mnist=(0, 1577),
    prediction_indices_corrupted=(0, 3710),
    prediction_indices_both=(9241,),
    bayesian_ensemble_size=50,
):
    figures_dir.mkdir(parents=True, exist_ok=True)

    _save_dataset_strip(
        x_train,
        min(8, x_train.shape[0]),
        figures_dir / "02_mnist_samples.png",
        "MNIST samples",
    )
    _save_dataset_strip(
        x_c_test,
        min(8, x_c_test.shape[0]),
        figures_dir / "03_mnist_corrupted_samples.png",
        "MNIST-C samples",
    )

    run_ensemble = model_name == "bayesian_cnn"
    ensemble_size = int(bayesian_ensemble_size) if run_ensemble else 1

    _save_prediction_examples(
        model,
        x_test,
        y_test,
        prediction_indices_mnist,
        figures_dir,
        prefix="04_prediction_mnist",
        run_ensemble=run_ensemble,
        ensemble_size=ensemble_size,
    )
    _save_prediction_examples(
        model,
        x_c_test,
        y_c_test,
        prediction_indices_corrupted,
        figures_dir,
        prefix="05_prediction_mnist_c",
        run_ensemble=run_ensemble,
        ensemble_size=ensemble_size,
    )

    for image_index in prediction_indices_both:
        if image_index >= x_test.shape[0] or image_index >= x_c_test.shape[0]:
            continue
        figure, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(10, 4),
            gridspec_kw={"width_ratios": [2, 4]},
        )
        _plot_prediction_panel(
            axes[0, 0],
            axes[0, 1],
            x_test,
            y_test,
            model,
            image_index,
            run_ensemble=run_ensemble,
            ensemble_size=ensemble_size,
            title_prefix="MNIST",
        )
        _plot_prediction_panel(
            axes[1, 0],
            axes[1, 1],
            x_c_test,
            y_c_test,
            model,
            image_index,
            run_ensemble=run_ensemble,
            ensemble_size=ensemble_size,
            title_prefix="MNIST-C",
        )
        figure.tight_layout()
        figure.savefig(
            figures_dir / f"06_prediction_comparison_{int(image_index)}.png",
            dpi=150,
        )
        plt.close(figure)

    mnist_entropy = _save_entropy_distribution(
        model,
        x_test,
        y_test,
        figures_dir / "07_entropy_mnist.png",
        "MNIST",
    )
    corrupted_entropy = _save_entropy_distribution(
        model,
        x_c_test,
        y_c_test,
        figures_dir / "08_entropy_mnist_c.png",
        "MNIST-C",
    )

    return {
        "mnist_entropy": mnist_entropy,
        "mnist_c_entropy": corrupted_entropy,
    }


def _save_dataset_strip(data, num_images, output_path, title):
    figure, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(2 * num_images, 2))
    if num_images == 1:
        axes = [axes]

    for image_index in range(num_images):
        axes[image_index].imshow(data[image_index, ..., 0], cmap="gray")
        axes[image_index].axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_prediction_examples(
    model,
    data,
    true_labels,
    image_indices,
    figures_dir,
    prefix,
    run_ensemble,
    ensemble_size,
):
    for image_index in image_indices:
        if image_index >= data.shape[0]:
            continue
        figure, (axis_image, axis_probs) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(10, 2),
            gridspec_kw={"width_ratios": [2, 4]},
        )
        _plot_prediction_panel(
            axis_image,
            axis_probs,
            data,
            true_labels,
            model,
            image_index,
            run_ensemble=run_ensemble,
            ensemble_size=ensemble_size,
        )
        figure.tight_layout()
        figure.savefig(figures_dir / f"{prefix}_{int(image_index)}.png", dpi=150)
        plt.close(figure)


def _plot_prediction_panel(
    axis_image,
    axis_probs,
    data,
    true_labels,
    model,
    image_index,
    run_ensemble=False,
    ensemble_size=1,
    title_prefix="",
):
    image = data[image_index]
    true_label = int(np.squeeze(true_labels[image_index]))

    local_ensemble_size = int(ensemble_size if run_ensemble else 1)
    predicted_probabilities = np.empty(
        shape=(local_ensemble_size, 10), dtype=np.float32
    )

    for ensemble_index in range(local_ensemble_size):
        predicted_probabilities[ensemble_index] = _predict_probabilities(model, image)[
            0
        ]

    pct_2p5 = np.percentile(predicted_probabilities, 2.5, axis=0)
    pct_97p5 = np.percentile(predicted_probabilities, 97.5, axis=0)

    axis_image.imshow(image[..., 0], cmap="gray")
    axis_image.axis("off")
    prefix = f"{title_prefix} | " if title_prefix else ""
    axis_image.set_title(f"{prefix}True: {true_label}")

    bars = axis_probs.bar(np.arange(10), pct_97p5, color="red")
    bars[true_label].set_color("green")
    axis_probs.bar(
        np.arange(10),
        np.clip(pct_2p5 - 0.02, 0.0, 1.0),
        color="white",
        linewidth=1,
        edgecolor="white",
    )
    axis_probs.set_xticks(np.arange(10))
    axis_probs.set_ylim([0, 1])
    axis_probs.set_ylabel("Probability")
    axis_probs.set_title("Model estimated probabilities")


def _save_entropy_distribution(model, x, labels, output_path, dataset_name):
    probabilities = _predict_probabilities_batch(model, x)
    probabilities = np.clip(probabilities, 1e-8, 1.0)

    entropy = -np.sum(probabilities * np.log2(probabilities), axis=1)
    predicted = np.argmax(probabilities, axis=1)
    labels_flat = np.squeeze(labels)
    correct = predicted == labels_flat

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    stats = {}
    for axis, mask, category in zip(
        axes,
        [correct, ~correct],
        ["Correct", "Incorrect"],
    ):
        entropy_category = entropy[mask]
        if entropy_category.size == 0:
            entropy_category = np.array([0.0], dtype=np.float32)

        mean_entropy = float(np.mean(entropy_category))
        stats[f"mean_entropy_{category.lower()}"] = mean_entropy
        stats[f"count_{category.lower()}"] = int(mask.sum())

        num_samples = entropy_category.shape[0]
        weights = (1 / num_samples) * np.ones(num_samples)
        axis.hist(entropy_category, weights=weights)
        axis.annotate(f"Mean: {mean_entropy:.3f} bits", (0.5, 0.9), ha="center")
        axis.set_xlabel("Entropy (bits)")
        axis.set_ylim([0, 1])
        axis.set_ylabel("Probability")
        axis.set_title(
            f"{category}ly labelled ({mask.sum() / x.shape[0] * 100:.1f}% of total)"
        )

    figure.suptitle(f"Entropy distribution - {dataset_name}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    stats["accuracy"] = float(correct.mean())
    return stats


def _predict_probabilities(model, image):
    predictions = model(image[np.newaxis, :])
    return _extract_probabilities(predictions)


def _predict_probabilities_batch(model, x):
    predictions = model(x)
    return _extract_probabilities(predictions)


def _extract_probabilities(predictions):
    if hasattr(predictions, "mean"):
        probs = predictions.mean().numpy()
    else:
        probs = predictions.numpy()

    if probs.ndim == 1:
        probs = probs[np.newaxis, :]

    probs = np.asarray(probs, dtype=np.float32)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return probs / row_sums
