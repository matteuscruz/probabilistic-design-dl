import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras

tfd = tfp.distributions
tfpl = tfp.layers


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


def get_convolutional_reparameterization_layer(input_shape, divergence_fn):
    return tfpl.Convolution2DReparameterization(
        filters=8,
        kernel_size=(5, 5),
        activation="relu",
        padding="VALID",
        input_shape=input_shape,
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(
            is_singular=False,
        ),
        kernel_divergence_fn=divergence_fn,
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_divergence_fn=divergence_fn,
    )


def spike_and_slab(event_shape, dtype):
    return tfd.Mixture(
        cat=tfd.Categorical(probs=[0.5, 0.5]),
        components=[
            tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(event_shape, dtype=dtype),
                    scale=tf.ones(event_shape, dtype=dtype),
                ),
                reinterpreted_batch_ndims=1,
            ),
            tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(event_shape, dtype=dtype),
                    scale=10.0 * tf.ones(event_shape, dtype=dtype),
                ),
                reinterpreted_batch_ndims=1,
            ),
        ],
        name="spike_and_slab",
    )


def get_prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return keras.Sequential(
        [
            tfpl.DistributionLambda(
                lambda _: spike_and_slab(event_shape=n, dtype=dtype)
            ),
        ]
    )


def get_posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return keras.Sequential(
        [
            tfpl.VariableLayer(2 * n, dtype=dtype),
            tfpl.IndependentNormal(n),
        ]
    )


def get_dense_variational_layer(prior_fn, posterior_fn, kl_weight):
    return tfpl.DenseVariational(
        units=10,
        make_prior_fn=prior_fn,
        make_posterior_fn=posterior_fn,
        kl_weight=kl_weight,
        kl_use_exact=False,
    )


def build_bayesian_cnn_model(
    convolutional_layer,
    dense_variational_layer,
    loss,
    optimizer,
    metrics,
):
    model = keras.Sequential(
        [
            convolutional_layer,
            keras.layers.MaxPooling2D(pool_size=(6, 6)),
            keras.layers.Flatten(),
            dense_variational_layer,
            tfpl.OneHotCategorical(
                10,
                convert_to_tensor_fn=tfd.Distribution.mode,
            ),
        ]
    )
    model.compile(
        loss=loss,
        optimizer=_to_tf_keras_optimizer(optimizer),
        metrics=metrics,
        experimental_run_tf_function=False,
    )
    return model
