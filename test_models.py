import pandas as pd
import numpy as np
import argparse

from distutils.util import strtobool
from sklearn.preprocessing import normalize
from utils.utils import get_balanced_train_data
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

bool_fn = lambda x: bool(strtobool(str(x)))
parser = argparse.ArgumentParser()
parser.add_argument("--n_splits", default='5', help='Define number of splits for cross validator.')
ARGS, unknown = parser.parse_known_args()


if __name__ == '__main__':
    # Collect data, concatenate and normalize.
    y = pd.read_pickle("data/y.pkl")
    X_PROCESSED = pd.read_pickle("data/X_PROCESSED.pkl")
    X_REDEFINED = pd.read_pickle("data/X_REDEFINED.pkl")
    X_FINAL = pd.concat([X_PROCESSED, X_REDEFINED], axis=1)
    X_FINAL = pd.DataFrame(normalize(X_FINAL, norm='max', axis=0), columns=X_FINAL.columns.tolist())

    # Transform specific data.
    to_transform = ['std_v6', 'mean_v6', 'v6']
    X_FINAL[to_transform] = np.power(X_FINAL[to_transform], 0.1)

    # Get train data (balanced between classes).
    X_to_train, y_to_train, _, _ = get_balanced_train_data(X_FINAL, y)

    # Define models.
    clf1 = RandomForestClassifier()
    clf2 = BaggingClassifier(KNeighborsClassifier(n_neighbors=2), max_samples=0.5, max_features=0.5)
    clf3 = MLPClassifier(
        solver='adam', max_iter=200, learning_rate_init=1e-2,
        learning_rate='adaptive', alpha=1e-2,
        hidden_layer_sizes=(16, 12, 8, 4), random_state=2)

    # Train and test model.
    fig = plt.figure()
    for name, model in zip(['random-forest', 'knn-ensemble', 'mlp'], [clf1, clf2, clf3]):
        skf = StratifiedKFold(n_splits=int(ARGS.n_splits))  # k-fold cross-validator

        cnt = 1
        f1_val, rec_val, prec_val = [], [], []
        for train_ix, test_ix in skf.split(X_to_train, y_to_train):
            # Define train and test sets
            X_train, X_test = X_to_train[train_ix], X_to_train[test_ix]
            y_train, y_test = y_to_train[train_ix], y_to_train[test_ix]

            # Test model.
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Calculate standard metrics of test.
            rec = recall_score(y_test, predictions)
            prec = precision_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            # Estimate ROC curve with AUC score.
            y_pred_proba = model.predict_proba(X_test)[::, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)

            f1_val += [f1]
            rec_val += [rec]
            prec_val += [prec]

            msg = '{} (step {}) - f1: {:.3f}, recall: {:.3f}, precision: {:.3f}'
            print(msg.format(name, cnt, f1, rec, prec))
            cnt += 1

        msg = 'Mean: f1: {:.3f} (+/- {:.3f}), recall: {:.3f} (+/- {:.3f}), precision: {:.3f} (+/- {:.3f})'
        print(msg.format(
            np.mean(f1_val), np.std(f1_val),
            np.mean(rec_val), np.std(rec_val),
            np.mean(prec_val), np.std(prec_val)))

        # Plot ROC curve of last model test.
        plt.plot(fpr, tpr, label="AUC ({}) = {:.3f}".format(name, auc))
        print('')

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()
