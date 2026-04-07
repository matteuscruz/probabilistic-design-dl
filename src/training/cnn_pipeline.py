import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras
import matplotlib.pyplot as plt

from src.data.mnist import load_data
from src.models.bayesian_cnn import build_bayesian_cnn_model
from src.models.bayesian_cnn import get_convolutional_reparameterization_layer
from src.models.bayesian_cnn import get_dense_variational_layer
from src.models.bayesian_cnn import get_posterior
from src.models.bayesian_cnn import get_prior
from src.models.cnn_deterministic import get_deterministic_model
from src.models.cnn_probabilistic import get_probabilistic_model
from src.models.cnn_probabilistic import nll
from src.training.experiment_manager import ExperimentManager
from src.visualization.cnn_artifacts import generate_cnn_experiment_figures

tfd = tfp.distributions


def _divergence_fn_factory(num_samples):
    def _divergence_fn(q, p, _):
        return tfd.kl_divergence(q, p) / num_samples

    return _divergence_fn


def run_cnn_pipeline(config, experiment_manager=None):
    seed = config.train.get("seed", config.split.get("random_state", 42))
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    data_root = config.data.get("root", "data")
    mnist_name = config.data.get("mnist_name", "MNIST")
    corrupted_name = config.data.get("mnist_corrupted_name", "MNIST_corrupted")

    (x_train, y_train, y_train_oh), (x_test, y_test, y_test_oh) = load_data(
        mnist_name,
        data_root=data_root,
    )
    (_, _, _), (x_c_test, y_c_test, y_c_test_oh) = load_data(
        corrupted_name,
        data_root=data_root,
    )

    model_name = config.model.get("name", "cnn_deterministic")
    epochs = int(config.train.get("epochs", 1))
    learning_rate = float(config.train.get("learning_rate", 0.001))
    verbose = int(config.train.get("verbose", 0))
    if experiment_manager is None:
        experiment_manager = ExperimentManager.create(
            config=config, model_name=model_name
        )

    if model_name == "cnn_deterministic":
        model = get_deterministic_model(
            input_shape=(28, 28, 1),
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            metrics=["accuracy"],
        )
        fit_history = model.fit(x_train, y_train, epochs=epochs, verbose=verbose)
        accuracy_test = float(model.evaluate(x_test, y_test, verbose=False)[1])
        accuracy_corrupted = float(model.evaluate(x_c_test, y_c_test, verbose=False)[1])

    elif model_name == "cnn_probabilistic":
        model = get_probabilistic_model(
            input_shape=(28, 28, 1),
            loss=nll,
            optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
            metrics=["accuracy"],
        )
        fit_history = model.fit(x_train, y_train_oh, epochs=epochs, verbose=verbose)
        accuracy_test = float(model.evaluate(x_test, y_test_oh, verbose=False)[1])
        accuracy_corrupted = float(
            model.evaluate(x_c_test, y_c_test_oh, verbose=False)[1]
        )

    elif model_name == "bayesian_cnn":
        divergence_fn = _divergence_fn_factory(x_train.shape[0])
        convolutional_layer = get_convolutional_reparameterization_layer(
            input_shape=(28, 28, 1),
            divergence_fn=divergence_fn,
        )
        dense_variational_layer = get_dense_variational_layer(
            get_prior,
            get_posterior,
            kl_weight=1 / x_train.shape[0],
        )
        model = build_bayesian_cnn_model(
            convolutional_layer,
            dense_variational_layer,
            loss=nll,
            optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
            metrics=["accuracy"],
        )
        fit_history = model.fit(x_train, y_train_oh, epochs=epochs, verbose=verbose)
        accuracy_test = float(model.evaluate(x_test, y_test_oh, verbose=False)[1])
        accuracy_corrupted = float(
            model.evaluate(x_c_test, y_c_test_oh, verbose=False)[1]
        )

    else:
        raise ValueError(f"Unsupported CNN model: {model_name}")

    artifacts_cfg = config.artifacts
    if experiment_manager.enabled:
        if artifacts_cfg.get("save_model", True):
            model_path = experiment_manager.model_path(f"{model_name}.h5")
            try:
                model.save(model_path)
            except Exception:
                weights_path = experiment_manager.model_path(f"{model_name}.weights.h5")
                model.save_weights(weights_path)
                experiment_manager.save_json(
                    "model",
                    "model_save_fallback.json",
                    {
                        "reason": "full_model_serialization_failed",
                        "weights_path": str(weights_path.name),
                    },
                )

        if artifacts_cfg.get("save_history", True):
            history_dict = fit_history.history
            metric_names = sorted(history_dict.keys())
            rows = []
            for epoch in range(epochs):
                row = [epoch + 1]
                for metric_name in metric_names:
                    metric_values = history_dict.get(metric_name, [])
                    row.append(
                        float(metric_values[epoch])
                        if epoch < len(metric_values)
                        else ""
                    )
                rows.append(row)

            experiment_manager.save_csv(
                "history",
                "training_history.csv",
                headers=["epoch", *metric_names],
                rows=rows,
            )
            experiment_manager.save_csv(
                "history",
                "metrics.csv",
                headers=["metric", "value"],
                rows=[
                    ["accuracy_test", accuracy_test],
                    ["accuracy_corrupted", accuracy_corrupted],
                    ["epochs", epochs],
                    ["learning_rate", learning_rate],
                ],
            )

        if artifacts_cfg.get("save_figures", True):
            history_dict = fit_history.history
            metric_names = sorted(history_dict.keys())
            figure, axis = plt.subplots(figsize=(10, 5))
            for metric_name in metric_names:
                axis.plot(history_dict[metric_name], label=metric_name)
            axis.set_title(f"{model_name} training history")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Metric value")
            axis.legend()
            figure.tight_layout()
            figure.savefig(
                experiment_manager.figure_path("training_history.png"),
                dpi=150,
            )
            plt.close(figure)

            entropy_summary = generate_cnn_experiment_figures(
                model=model,
                model_name=model_name,
                x_train=x_train,
                x_test=x_test,
                y_test=y_test,
                x_c_test=x_c_test,
                y_c_test=y_c_test,
                figures_dir=experiment_manager.figures_dir,
                bayesian_ensemble_size=int(
                    artifacts_cfg.get("cnn_bayesian_ensemble_size", 50)
                ),
            )

            if artifacts_cfg.get("save_history", True):
                experiment_manager.save_json(
                    "history",
                    "entropy_summary.json",
                    entropy_summary,
                )

    return {
        "model": model_name,
        "seed": seed,
        "accuracy_test": accuracy_test,
        "accuracy_corrupted": accuracy_corrupted,
    }
