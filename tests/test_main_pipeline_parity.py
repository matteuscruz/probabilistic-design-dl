import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("tensorflow_probability")
import tensorflow as tf

from main import run
from src.data.loaders import load_iris_sepal_dataset
from src.data.split import train_test_split_dataset
from src.models.naive_bayes import get_prior
from src.models.naive_bayes import get_class_conditionals
from src.models.naive_bayes import predict_class
from src.training.optim import learn_stdevs


SEED = 42
EXPECTED_MULTICLASS_ACCURACY = 0.7000
EXPECTED_BINARY_ACCURACY = 1.0000
EXPECTED_MULTICLASS_PREDICTIONS = [
    0, 1, 1, 1, 0, 2, 0, 0, 2, 1, 2, 2, 2, 2, 0,
    0, 0, 1, 1, 1, 0, 2, 2, 1, 1, 2, 2, 0, 2, 0,
]
EXPECTED_BINARY_PREDICTIONS = [
    0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0,
]


def _build_multiclass_outputs():
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    x, y = load_iris_sepal_dataset()
    x_train, x_test, y_train, y_test = train_test_split_dataset(
        x,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=True,
    )

    prior = get_prior(y_train)
    class_conditionals = get_class_conditionals(x_train, y_train)
    predictions = predict_class(prior, class_conditionals, x_test)
    accuracy = float(np.mean(predictions == y_test))

    return x_train, x_test, y_train, y_test, predictions, accuracy


def _build_binary_outputs(x_train, x_test, y_train, y_test):
    y_train_binary = np.array(y_train)
    y_train_binary[np.where(y_train_binary == 2)] = 1

    y_test_binary = np.array(y_test)
    y_test_binary[np.where(y_test_binary == 2)] = 1

    prior_binary = get_prior(y_train_binary)
    scales = tf.Variable([1.0, 1.0], dtype=tf.float32)
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.01)
    _, _, class_conditionals_binary = learn_stdevs(
        x_train,
        y_train_binary,
        scales,
        optimiser,
        epochs=500,
    )

    predictions_binary = predict_class(prior_binary, class_conditionals_binary, x_test)
    accuracy_binary = float(np.mean(predictions_binary == y_test_binary))

    return predictions_binary, accuracy_binary


def test_main_pipeline_accuracy_matches_notebook_baseline():
    accuracy_main = run()
    assert accuracy_main == pytest.approx(EXPECTED_MULTICLASS_ACCURACY, abs=1e-9)


def test_multiclass_predictions_match_notebook_baseline():
    _, _, _, _, predictions, accuracy = _build_multiclass_outputs()
    assert accuracy == pytest.approx(EXPECTED_MULTICLASS_ACCURACY, abs=1e-9)
    assert predictions.tolist() == EXPECTED_MULTICLASS_PREDICTIONS


def test_binary_predictions_match_notebook_baseline():
    x_train, x_test, y_train, y_test, _, _ = _build_multiclass_outputs()
    predictions_binary, accuracy_binary = _build_binary_outputs(x_train, x_test, y_train, y_test)

    assert accuracy_binary == pytest.approx(EXPECTED_BINARY_ACCURACY, abs=1e-9)
    assert predictions_binary.tolist() == EXPECTED_BINARY_PREDICTIONS
