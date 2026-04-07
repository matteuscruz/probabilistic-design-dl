import argparse

from src.config.runtime import load_config
from src.data.loaders import load_iris_sepal_dataset
from src.data.split import train_test_split_dataset
from src.models.naive_bayes import get_prior
from src.models.naive_bayes import get_class_conditionals
from src.models.naive_bayes import predict_class
from src.evaluation.metrics import accuracy
from src.training.cnn_pipeline import run_cnn_pipeline


def run(config_path="config/default.yaml"):
    cfg = load_config(config_path)
    model_name = cfg.model.get("name", "naive_bayes")

    if model_name == "naive_bayes":
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
        return accuracy(y_test, predictions)

    if model_name in {
        "cnn_deterministic",
        "cnn_probabilistic",
        "bayesian_cnn",
    }:
        return run_cnn_pipeline(cfg)

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
