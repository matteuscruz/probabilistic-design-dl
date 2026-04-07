from src.config.runtime import load_config
from src.data.loaders import load_iris_sepal_dataset
from src.data.split import train_test_split_dataset
from src.models.naive_bayes import get_prior
from src.models.naive_bayes import get_class_conditionals
from src.models.naive_bayes import predict_class
from src.evaluation.metrics import accuracy


def run(config_path="config/default.yaml"):
	cfg = load_config(config_path)
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


if __name__ == "__main__":
	score = run()
	print(f"Accuracy: {score:.4f}")
