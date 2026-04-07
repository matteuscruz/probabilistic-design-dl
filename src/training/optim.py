import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def learn_stdevs(x, y, scales, optimiser, epochs):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)

    if x_tensor.shape.rank != 2:
        raise ValueError("x must have shape (num_samples, num_features)")
    if y_tensor.shape.rank != 1:
        raise ValueError("y must have shape (num_samples,)")
    if x_tensor.shape[0] != y_tensor.shape[0]:
        raise ValueError("x and y must have the same number of samples")

    num_samples = tf.shape(x_tensor)[0]
    num_classes = tf.reduce_max(y_tensor) + 1

    means = []
    for class_id in tf.range(num_classes):
        class_mask = tf.equal(y_tensor, class_id)
        class_samples = tf.boolean_mask(x_tensor, class_mask)
        class_mean = tf.reduce_mean(class_samples, axis=0)
        means.append(class_mean)
    means = tf.stack(means, axis=0)

    loss_history = np.zeros(epochs, dtype=np.float32)
    scales_history = np.zeros((epochs, x_tensor.shape[1]), dtype=np.float32)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            safe_scales = tf.maximum(scales, 1e-6)
            tiled_scales = tf.tile(
                tf.expand_dims(safe_scales, axis=0), [num_classes, 1]
            )
            class_conditionals = tfd.MultivariateNormalDiag(
                loc=means, scale_diag=tiled_scales
            )

            log_likelihoods = class_conditionals.log_prob(
                tf.expand_dims(x_tensor, axis=1)
            )
            indices = tf.stack([tf.range(num_samples), y_tensor], axis=1)
            true_class_log_likelihood = tf.gather_nd(log_likelihoods, indices)
            loss = -tf.reduce_mean(true_class_log_likelihood)

        gradients = tape.gradient(loss, [scales])
        optimiser.apply_gradients(zip(gradients, [scales]))

        loss_history[epoch] = float(loss.numpy())
        scales_history[epoch, :] = scales.numpy()

    final_scales = tf.maximum(scales, 1e-6)
    final_dist = tfd.MultivariateNormalDiag(
        loc=means,
        scale_diag=tf.tile(tf.expand_dims(final_scales, axis=0), [num_classes, 1]),
    )
    return loss_history, scales_history, final_dist
