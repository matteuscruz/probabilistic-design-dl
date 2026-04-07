import pytest


def _load_modules():
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_probability")

    import tensorflow as tf

    from src.models.bayesian_cnn import build_bayesian_cnn_model
    from src.models.bayesian_cnn import (
        get_convolutional_reparameterization_layer,
    )
    from src.models.bayesian_cnn import get_dense_variational_layer
    from src.models.bayesian_cnn import get_posterior
    from src.models.bayesian_cnn import get_prior
    from src.models.cnn_deterministic import get_deterministic_model
    from src.models.cnn_probabilistic import get_probabilistic_model
    from src.models.cnn_probabilistic import nll

    return {
        "tf": tf,
        "build_bayesian_cnn_model": build_bayesian_cnn_model,
        "get_convolutional_reparameterization_layer": (
            get_convolutional_reparameterization_layer
        ),
        "get_dense_variational_layer": get_dense_variational_layer,
        "get_posterior": get_posterior,
        "get_prior": get_prior,
        "get_deterministic_model": get_deterministic_model,
        "get_probabilistic_model": get_probabilistic_model,
        "nll": nll,
    }


def test_build_deterministic_model():
    modules = _load_modules()
    tf = modules["tf"]

    model = modules["get_deterministic_model"](
        input_shape=(28, 28, 1),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer="rmsprop",
        metrics=["accuracy"],
    )
    assert len(model.layers) == 4


def test_build_probabilistic_model():
    modules = _load_modules()

    model = modules["get_probabilistic_model"](
        input_shape=(28, 28, 1),
        loss=modules["nll"],
        optimizer="rmsprop",
        metrics=["accuracy"],
    )
    assert len(model.layers) == 5


def test_build_bayesian_model():
    modules = _load_modules()
    tf = modules["tf"]

    def divergence_fn(q, p, _):
        return tf.reduce_sum(q.log_prob(0.0) - p.log_prob(0.0)) * 0.0

    conv_layer = modules["get_convolutional_reparameterization_layer"](
        (28, 28, 1),
        divergence_fn,
    )
    dense_variational = modules["get_dense_variational_layer"](
        prior_fn=modules["get_prior"],
        posterior_fn=modules["get_posterior"],
        kl_weight=1 / 32,
    )

    model = modules["build_bayesian_cnn_model"](
        convolutional_layer=conv_layer,
        dense_variational_layer=dense_variational,
        loss=modules["nll"],
        optimizer="rmsprop",
        metrics=["accuracy"],
    )
    assert len(model.layers) == 5
