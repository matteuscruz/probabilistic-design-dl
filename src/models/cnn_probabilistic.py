import tensorflow_probability as tfp
import tf_keras as keras


def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


def _to_tf_keras_optimizer(optimizer):
    if isinstance(optimizer, keras.optimizers.Optimizer):
        return optimizer
    if hasattr(optimizer, "get_config"):
        return keras.optimizers.get(
            {
                "class_name": optimizer.__class__.__name__,
                "config": optimizer.get_config(),
            }
        )
    return optimizer


def get_probabilistic_model(input_shape, loss, optimizer, metrics):
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                8,
                (5, 5),
                activation="relu",
                padding="VALID",
                input_shape=input_shape,
            ),
            keras.layers.MaxPooling2D((6, 6)),
            keras.layers.Flatten(),
            keras.layers.Dense(10),
            tfp.layers.OneHotCategorical(
                10,
                convert_to_tensor_fn=tfp.distributions.OneHotCategorical.mode,
            ),
        ]
    )
    model.compile(
        loss=loss, optimizer=_to_tf_keras_optimizer(optimizer), metrics=metrics
    )
    return model
