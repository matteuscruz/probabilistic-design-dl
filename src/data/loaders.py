from sklearn import datasets


def load_iris_sepal_dataset():
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    return x, y
