import numpy as np

from ..data.dataset import Dataset


def train_test_split(
    dataset,
    test_size=0.2,
    random_state=None
):
    """
    Split a Dataset into training and testing datasets.

    Parameters
    ----------
    dataset : Dataset
        Dataset to split.

    test_size : float, default=0.2
        Proportion of samples to include in the test Dataset.

    random_state : int, optional
        Seed used to generate the random permutation.

    Returns
    -------
    train_dataset : Dataset
        Dataset containing the training samples.

    test_dataset : Dataset
        Dataset containing the testing samples.
    """
    if test_size <= 0 or test_size >= 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = dataset.X.shape[0]

    indices = np.random.permutation(n_samples)

    n_test = int(n_samples * test_size)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    if dataset.has_label():
        train_y = dataset.y[train_indices]
        test_y = dataset.y[test_indices]
    else:
        train_y = None
        test_y = None

    train_dataset = Dataset(
        X=dataset.X[train_indices],
        y=train_y,
        features=dataset.features,
        label=dataset.label
    )

    test_dataset = Dataset(
        X=dataset.X[test_indices],
        y=test_y,
        features=dataset.features,
        label=dataset.label
    )

    return train_dataset, test_dataset