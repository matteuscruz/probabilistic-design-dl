from sklearn import model_selection


def train_test_split_dataset(x, y, test_size=0.2, random_state=None, stratify=True):
    stratify_target = y if stratify else None
    return model_selection.train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )
