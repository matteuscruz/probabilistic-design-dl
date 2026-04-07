import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def get_prior(y):
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1-D array")

    num_samples = y.shape[0]
    if num_samples == 0:
        raise ValueError("y must contain at least one sample")

    num_classes = int(np.max(y)) + 1
    counts = np.bincount(y, minlength=num_classes)
    probs = counts.astype(np.float32) / float(num_samples)
    return tfd.Categorical(probs=tf.convert_to_tensor(probs, dtype=tf.float32))


def get_class_conditionals(x, y, min_scale=1e-6):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 2:
        raise ValueError("x must be a 2-D array with shape (num_samples, num_features)")
    if y.ndim != 1:
        raise ValueError("y must be a 1-D array")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")

    num_samples, num_features = x.shape
    if num_samples == 0:
        raise ValueError("x must contain at least one sample")

    num_classes = int(np.max(y)) + 1
    means = np.zeros((num_classes, num_features), dtype=np.float32)
    stds = np.ones((num_classes, num_features), dtype=np.float32)

    for class_id in range(num_classes):
        mask = y == class_id
        if np.any(mask):
            class_samples = x[mask]
            means[class_id] = np.mean(class_samples, axis=0)
            class_variance = np.var(class_samples, axis=0, ddof=0)
            stds[class_id] = np.sqrt(np.maximum(class_variance, min_scale))

    return tfd.MultivariateNormalDiag(
        loc=tf.convert_to_tensor(means, dtype=tf.float32),
        scale_diag=tf.convert_to_tensor(stds, dtype=tf.float32),
    )


def predict_class(prior, class_conditionals, x):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    if x_tensor.shape.rank is None or x_tensor.shape.rank < 2:
        raise ValueError("x must have rank >= 2 and last dimension num_features")

    num_features = x_tensor.shape[-1]
    x_flat = tf.reshape(x_tensor, [-1, num_features])
    x_expanded = tf.expand_dims(x_flat, axis=1)

    log_likelihoods = class_conditionals.log_prob(x_expanded)
    num_classes = tf.shape(log_likelihoods)[-1]
    log_priors = prior.log_prob(tf.range(num_classes))
    log_post = log_likelihoods + log_priors

    y_hat_flat = tf.argmax(log_post, axis=-1, output_type=tf.int32)
    output_shape = tf.shape(x_tensor)[:-1]
    y_hat = tf.reshape(y_hat_flat, output_shape)
    return y_hat.numpy()
