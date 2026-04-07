import tensorflow as tf


def get_deterministic_model(input_shape, loss, optimizer, metrics):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                8,
                (5, 5),
                activation="relu",
                padding="VALID",
                input_shape=input_shape,
            ),
            tf.keras.layers.MaxPooling2D((6, 6)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
