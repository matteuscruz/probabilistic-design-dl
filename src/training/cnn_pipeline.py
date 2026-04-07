import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras

from src.data.mnist import load_data
from src.models.bayesian_cnn import build_bayesian_cnn_model
from src.models.bayesian_cnn import get_convolutional_reparameterization_layer
from src.models.bayesian_cnn import get_dense_variational_layer
from src.models.bayesian_cnn import get_posterior
from src.models.bayesian_cnn import get_prior
from src.models.cnn_deterministic import get_deterministic_model
from src.models.cnn_probabilistic import get_probabilistic_model
from src.models.cnn_probabilistic import nll

tfd = tfp.distributions


def _divergence_fn_factory(num_samples):
    def _divergence_fn(q, p, _):
        return tfd.kl_divergence(q, p) / num_samples

    return _divergence_fn


def run_cnn_pipeline(config):
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

    if model_name == "cnn_deterministic":
        model = get_deterministic_model(
            input_shape=(28, 28, 1),
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            metrics=["accuracy"],
        )
        model.fit(x_train, y_train, epochs=epochs, verbose=verbose)
        accuracy_test = float(model.evaluate(x_test, y_test, verbose=False)[1])
        accuracy_corrupted = float(model.evaluate(x_c_test, y_c_test, verbose=False)[1])

    elif model_name == "cnn_probabilistic":
        model = get_probabilistic_model(
            input_shape=(28, 28, 1),
            loss=nll,
            optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
            metrics=["accuracy"],
        )
        model.fit(x_train, y_train_oh, epochs=epochs, verbose=verbose)
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
        model.fit(x_train, y_train_oh, epochs=epochs, verbose=verbose)
        accuracy_test = float(model.evaluate(x_test, y_test_oh, verbose=False)[1])
        accuracy_corrupted = float(
            model.evaluate(x_c_test, y_c_test_oh, verbose=False)[1]
        )

    else:
        raise ValueError(f"Unsupported CNN model: {model_name}")

    return {
        "model": model_name,
        "seed": seed,
        "accuracy_test": accuracy_test,
        "accuracy_corrupted": accuracy_corrupted,
    }
