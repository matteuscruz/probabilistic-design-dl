import numpy as np


def get_logistic_regression_params(prior, class_conditionals):
    prior_probs = prior.probs_parameter().numpy()
    if prior_probs.shape[0] != 2:
        raise ValueError("This function expects a binary prior distribution with 2 classes")

    means = class_conditionals.loc.numpy()
    if means.shape[0] != 2:
        raise ValueError("This function expects binary class-conditionals with 2 classes")

    stds = class_conditionals.stddev().numpy()
    variances = np.square(stds)
    shared_variance = np.mean(variances, axis=0)
    sigma_inv = np.diag(1.0 / (shared_variance + 1e-6))

    mu0, mu1 = means[0], means[1]
    w = sigma_inv @ (mu0 - mu1)

    p_y0, p_y1 = prior_probs[0], prior_probs[1]
    w0 = (
        -0.5 * mu0.T @ sigma_inv @ mu0
        + 0.5 * mu1.T @ sigma_inv @ mu1
        + np.log((p_y0 + 1e-12) / (p_y1 + 1e-12))
    )
    return w, np.asarray(w0)
