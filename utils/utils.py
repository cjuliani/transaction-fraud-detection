import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_balanced_train_data(X, y):
    """Returns random train and test data with balanced sampling.

    Args:
        X: main features (dataframe).
        y: classes.
    """
    # Solve sampling imbalance
    N = y[y == 1].count()
    X_class_0 = X[y == 0].sample(N)  # under-sampling from the biggest class dataset
    X_class_1 = X[y == 1].sample(N, replace=True)  # normal sampling from smallest datasets

    # Define train features and classes
    X_to_train = np.concatenate([X_class_0, X_class_1], axis=0)
    y_to_train = np.concatenate([np.array([0] * N), np.array([1] * N)])

    return X_to_train, y_to_train, X_class_0, X_class_1


def plot_distributions_per_class(data, feature_names):
    for k in feature_names:
        plt.figure()
        for cls in data:
            sns.distplot(cls[k], bins=30)
        plt.title('feature: ' + str(k))
        plt.show()


def check_null_values(features):
    """Check if values of features have NaN.

    Args:
        features: dataframe to check.
    """
    results = features.isna().apply(lambda x: x.astype(int)).sum()
    print(results)
