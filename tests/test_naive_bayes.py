import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("tensorflow_probability")

from src.models.naive_bayes import get_prior
from src.models.naive_bayes import get_class_conditionals
from src.models.naive_bayes import predict_class


def test_get_prior_probabilities():
    y = np.array([0, 0, 1, 2, 2, 2])
    prior = get_prior(y)
    probs = prior.probs_parameter().numpy()

    assert probs.shape == (3,)
    assert np.isclose(np.sum(probs), 1.0)
    assert np.allclose(probs, np.array([2 / 6, 1 / 6, 3 / 6], dtype=np.float32))


def test_get_class_conditionals_shapes_and_means():
    x = np.array(
        [
            [1.0, 2.0],
            [1.2, 2.2],
            [3.0, 4.0],
            [2.8, 4.2],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1])
    dist = get_class_conditionals(x, y)

    assert tuple(dist.loc.shape) == (2, 2)
    means = dist.loc.numpy()
    assert np.allclose(means[0], np.array([1.1, 2.1]), atol=1e-6)
    assert np.allclose(means[1], np.array([2.9, 4.1]), atol=1e-6)


def test_predict_class_supports_higher_rank_batch_shape():
    x_train = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [2.0, 2.0],
            [2.2, 2.1],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1])

    prior = get_prior(y_train)
    conditionals = get_class_conditionals(x_train, y_train)

    x_test = np.array(
        [
            [[0.1, 0.0], [2.1, 2.0]],
            [[0.0, 0.1], [2.3, 2.2]],
        ],
        dtype=np.float32,
    )
    pred = predict_class(prior, conditionals, x_test)
    assert pred.shape == (2, 2)
    assert set(np.unique(pred)).issubset({0, 1})
