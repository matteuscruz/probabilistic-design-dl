import argparse
import numpy as np

from src.config.runtime import load_config
from src.data.loaders import load_iris_sepal_dataset
from src.data.split import train_test_split_dataset
from src.models.naive_bayes import get_prior
from src.models.naive_bayes import get_class_conditionals
from src.models.naive_bayes import predict_class
from src.evaluation.metrics import accuracy
from src.training.experiment_manager import ExperimentManager
from src.training.cnn_pipeline import run_cnn_pipeline
from src.visualization.naive_bayes_artifacts import (
    generate_naive_bayes_experiment_figures,
)


def run(config_path="config/default.yaml"):
    cfg = load_config(config_path)
    model_name = cfg.model.get("name", "naive_bayes")

    if model_name == "naive_bayes":
        experiment_manager = ExperimentManager.create(
            config=cfg,
            model_name=model_name,
            config_path=config_path,
        )
        seed = cfg.train.get("seed", cfg.split.get("random_state", 42))
        np.random.seed(seed)

        x, y = load_iris_sepal_dataset()
        x_train, x_test, y_train, y_test = train_test_split_dataset(
            x,
            y,
            test_size=cfg.split["test_size"],
            random_state=cfg.split["random_state"],
            stratify=cfg.split["stratify"],
        )
        prior = get_prior(y_train)
        class_conditionals = get_class_conditionals(x_train, y_train)
        predictions = predict_class(prior, class_conditionals, x_test)
        accuracy_value = accuracy(y_test, predictions)

        artifacts_cfg = cfg.artifacts
        if experiment_manager.enabled:
            if artifacts_cfg.get("save_model", True):
                experiment_manager.save_npz_model(
                    "naive_bayes.npz",
                    prior_probs=prior.probs_parameter().numpy(),
                    means=class_conditionals.loc.numpy(),
                    stds=class_conditionals.stddev().numpy(),
                )

            if artifacts_cfg.get("save_history", True):
                experiment_manager.save_csv(
                    "history",
                    "metrics.csv",
                    headers=["metric", "value"],
                    rows=[
                        ["accuracy_test", float(accuracy_value)],
                        ["num_train_samples", int(x_train.shape[0])],
                        ["num_test_samples", int(x_test.shape[0])],
                    ],
                )
                prediction_rows = [
                    [int(target), int(prediction)]
                    for target, prediction in zip(y_test, predictions)
                ]
                experiment_manager.save_csv(
                    "history",
                    "predictions_multiclass.csv",
                    headers=["y_true", "y_pred"],
                    rows=prediction_rows,
                )

            if artifacts_cfg.get("save_figures", True):
                figure_outputs = generate_naive_bayes_experiment_figures(
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    prior=prior,
                    class_conditionals=class_conditionals,
                    figures_dir=experiment_manager.figures_dir,
                    binary_epochs=int(artifacts_cfg.get("naive_binary_epochs", 50)),
                    seed=int(seed),
                )

                if artifacts_cfg.get("save_history", True):
                    binary_predictions = figure_outputs["binary_predictions"]
                    binary_nlls = figure_outputs["binary_nlls"]
                    binary_scales = figure_outputs["binary_scales"]

                    experiment_manager.save_csv(
                        "history",
                        "metrics_binary.csv",
                        headers=["metric", "value"],
                        rows=[
                            [
                                "accuracy_binary",
                                float(figure_outputs["binary_accuracy"]),
                            ],
                            ["binary_epochs", int(binary_nlls.shape[0])],
                        ],
                    )
                    experiment_manager.save_csv(
                        "history",
                        "binary_nlls.csv",
                        headers=["epoch", "nll"],
                        rows=[
                            [epoch, float(loss)]
                            for epoch, loss in enumerate(binary_nlls, start=1)
                        ],
                    )
                    experiment_manager.save_csv(
                        "history",
                        "binary_scales.csv",
                        headers=["epoch", "scale_0", "scale_1"],
                        rows=[
                            [epoch, float(scale_0), float(scale_1)]
                            for epoch, (scale_0, scale_1) in enumerate(
                                binary_scales,
                                start=1,
                            )
                        ],
                    )
                    binary_targets = np.array(y_test)
                    binary_targets[np.where(binary_targets == 2)] = 1
                    experiment_manager.save_csv(
                        "history",
                        "predictions_binary.csv",
                        headers=["y_true", "y_pred"],
                        rows=[
                            [int(target), int(prediction)]
                            for target, prediction in zip(
                                binary_targets,
                                binary_predictions,
                            )
                        ],
                    )

        return accuracy_value

    if model_name in {
        "cnn_deterministic",
        "cnn_probabilistic",
        "bayesian_cnn",
    }:
        experiment_manager = ExperimentManager.create(
            config=cfg,
            model_name=model_name,
            config_path=config_path,
        )
        return run_cnn_pipeline(cfg, experiment_manager=experiment_manager)

    raise ValueError(f"Unsupported model name: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run probabilistic-design-dl models")
    parser.add_argument(
        "config_path",
        nargs="?",
        default="config/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--config",
        dest="config_override",
        default=None,
        help="Path to YAML config file (overrides positional config_path)",
    )
    args = parser.parse_args()
    config_path = args.config_override or args.config_path

    result = run(config_path=config_path)
    if isinstance(result, dict):
        print(
            f"Model={result['model']} | "
            f"Accuracy(MNIST)={result['accuracy_test']:.4f} | "
            f"Accuracy(MNIST-C)={result['accuracy_corrupted']:.4f}"
        )
    else:
        print(f"Accuracy: {result:.4f}")
