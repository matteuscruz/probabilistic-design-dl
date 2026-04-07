import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("tensorflow_probability")
import tensorflow as tf

from src.models.naive_bayes import get_prior
from src.training.optim import learn_stdevs
from src.models.generative_logistic import get_logistic_regression_params


def test_learn_stdevs_returns_expected_shapes():
    x = np.array(
        [
            [5.1, 3.5],
            [4.9, 3.0],
            [6.2, 2.9],
            [6.0, 2.7],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1])

    scales = tf.Variable([1.0, 1.0], dtype=tf.float32)
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.01)
    epochs = 8

    losses, scales_arr, class_conditionals = learn_stdevs(x, y, scales, optimiser, epochs)

    assert losses.shape == (epochs,)
    assert scales_arr.shape == (epochs, 2)
    assert tuple(class_conditionals.loc.shape) == (2, 2)
    assert np.all(np.isfinite(losses))


def test_get_logistic_regression_params_shapes():
    x = np.array(
        [
            [1.0, 1.0],
            [1.2, 0.9],
            [3.0, 3.1],
            [2.8, 2.9],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1])

    prior = get_prior(y)
    scales = tf.Variable([1.0, 1.0], dtype=tf.float32)
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.01)
    _, _, class_conditionals = learn_stdevs(x, y, scales, optimiser, epochs=5)

    w, w0 = get_logistic_regression_params(prior, class_conditionals)
    assert w.shape == (2,)
    assert np.asarray(w0).shape == ()
    assert np.isfinite(w0)
