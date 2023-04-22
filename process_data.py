import numpy as np
import pandas as pd
import argparse
import warnings
import matplotlib.pyplot as plt
from multiprocess import Pool
from tqdm import tqdm
from functools import partial

from utils.config import *
from utils.utils import get_balanced_train_data, plot_distributions_per_class
from sklearn.preprocessing import OrdinalEncoder, normalize
from distutils.util import strtobool

warnings.filterwarnings('ignore')  # disable tensor slicing warnings when calculating gradients

bool_fn = lambda x: bool(strtobool(str(x)))
parser = argparse.ArgumentParser()
parser.add_argument("--plot_distributions", type=bool_fn, default='False',
                    help='Plot data distributions after processing.')
parser.add_argument("--pool_workers", default='4', help='Define number of workers for multiprocessing.')
ARGS, unknown = parser.parse_known_args()


def read_data(path):
    """Reads CSV file.

    Returns:
        X: main features (dataframe).
        y: classes.
    """
    df = pd.read_csv(path)
    flags = df['Flag']
    features = df.loc[:, df.columns != 'Flag']
    return features, flags


def plot_class_distribution(cls):
    """Plots class distributions to check potential imbalance.

    Args:
        cls: class values to check.
    """
    _ = plt.subplots(figsize=(10, 5))
    plt.bar(cls.index.astype(str), cls.values / sum(cls.values))
    plt.title("Class Distributions")
    plt.xlabel("Class")
    plt.show()


def process_numerical_data(dataframe, feature_names):
    feat_numerical = dataframe[feature_names]
    feat_v6 = pd.cut(feat_numerical["v6"],
                        bins=[-1,
                              feat_numerical["v6"].quantile(0.25),
                              feat_numerical["v6"].quantile(0.50),
                              feat_numerical["v6"].quantile(0.75),
                              feat_numerical["v6"].max()],
                        labels=[1, 2, 3, 4]).astype("int")
    feat_numerical['v6_label'] = feat_v6
    return feat_numerical


def process_categorical_data(dataframe, feature_names):
    # Define encoder for categorical data.
    enc = OrdinalEncoder()

    # Define dataframe.
    feat_categorical = dataframe[feature_names].fillna('UNKNOWN')

    # Build dictionary of countries to assign specific code value.
    GeoIpCountry = ['IN', 'UNKNOWN', 'SG', 'NP', 'US', 'CA', 'AU', 'FR', 'AE', 'DE', 'RO',
                    'SA', 'KW', 'BD', 'BR', 'MY', 'NG', 'UA', 'QA', 'GB', 'GH', 'TZ',
                    'HK', 'SE', 'MM', 'JP', 'PL', 'OM', 'IT', 'BH', 'IE', 'ZM', 'NL',
                    'PH', 'RU', 'TR', 'ID', 'KH', 'TH', 'UY']
    GeoIpCountry_values = {code: value for code, value in zip(GeoIpCountry, range(len(GeoIpCountry)))}

    # Convert country name into respective code value.
    feat_categorical['v9'] = feat_categorical['v9'].apply(lambda x: GeoIpCountry_values[x])
    feat_categorical['v8'] = feat_categorical['v8'].apply(lambda x: GeoIpCountry_values[x])

    # Check if v9 is same as v8 (set 1 if not, 0 otherwise).
    location_mismatch = (feat_categorical['v8'] - feat_categorical['v9']).apply(lambda x: int(x != 0))
    location_mismatch = pd.DataFrame(location_mismatch, columns=['CountryCodeMismatch'])

    # Concat converted country codes with other categorical data.
    tmp = enc.fit_transform(feat_categorical[CAT_DATA[:4]]).astype(int)
    feat_categorical[CAT_DATA[:4]] = pd.DataFrame(tmp, columns=CAT_DATA[:4])
    return pd.concat(
        [feat_categorical[CAT_DATA[:4]], feat_categorical['v9'],
         feat_categorical['v8'], location_mismatch], axis=1)


def process_time_data(dataframe):
    t1 = pd.to_datetime(dataframe['v1'], format='%M:%S.%f')
    t2 = pd.to_datetime(dataframe['v2'], format='%M:%S.%f')
    feat_time = (t2 - t1).dt.total_seconds()  # get different (t_completed - t_initiated)
    feat_time = feat_time.apply(lambda x: np.log(np.abs(x)))
    return pd.DataFrame(feat_time, columns=['TransactionTime_feat'])


def process_date_data(dataframe, feature_names):
    feat_date = dataframe[feature_names].apply(pd.to_datetime)
    feat_date = feat_date.apply(lambda x: x - pd.to_datetime(feat_date.min().to_list()[0]))
    return (feat_date / np.timedelta64(1, 'D')).astype(int)


def reshape_to_original_dataframe(original_df, extra_df_from_groups, indices):
    tmp_array = extra_df_from_groups.to_numpy()
    empty_array = np.zeros((original_df.shape[0], tmp_array.shape[1]))
    for i, row in enumerate(tmp_array):
        empty_array[indices[i]] = row

    return pd.DataFrame(empty_array, columns=extra_df_from_groups.columns)


def build_new_features(key, groups, dataframe, numpy):
    # Get person ID indices.
    idx = groups[key]

    # Number of transactions.
    num_transactions = dataframe.iloc[idx].shape[0]

    # Unique v3.
    variation_v3 = dataframe.iloc[idx]['v3'].unique().shape[0]

    # v6 frequency.
    v6 = dataframe.iloc[idx]['v6'].sort_values()
    mean_v6 = v6.mean()
    std_v6 = numpy.nan_to_num(v6.std())

    mean_v6_per_item = dataframe.iloc[idx]['v6'].sum() / dataframe.iloc[idx]['ItemName'].shape[0]

    lifetime_v6 = v6.max() - v6.min()
    v6_intervals = v6.diff().fillna(0.)
    mean_v6_freq = v6_intervals.mean()
    std_v6_freq = numpy.nan_to_num(v6_intervals.std())
    mean_v6_rel_freq = numpy.nan_to_num((v6_intervals / lifetime_v6).mean())

    # Statistics on transaction time.
    timing = dataframe.iloc[idx]['TransactionTime_feat']
    max_timing = timing.max()
    mean_timing = timing.mean()
    std_timing = numpy.nan_to_num(timing.std())

    return [
        num_transactions, variation_v3, mean_v6_per_item, mean_v6, std_v6,
        lifetime_v6, mean_v6_freq, std_v6_freq, mean_v6_rel_freq,
        max_timing, mean_timing, std_timing
    ], idx.values.tolist()


def build_features_multiprocessing(dataframe, groups, library, n_workers=4):
    # Define task and inputs to iterate.
    task = partial(build_new_features, dataframe=dataframe, groups=groups, numpy=library)
    inputs = groups.keys()

    # Process function with workers.
    pool = Pool(processes=n_workers)
    results = []
    for result in tqdm(pool.imap_unordered(task, inputs, chunksize=10000), total=len(inputs)):
        results.append(result)

    # Redefine data to return.
    feature_names = [
        'num_transactions', 'var_v3', 'avg_v6_per_item', 'mean_v6', 'std_v6',
        'lifetime_v6', 'avg_v6_freq', 'std_v6_freq', 'avg_v6_rel_freq',
        'max_timing', 'mean_timing', 'std_timing'
    ]
    new_features = pd.DataFrame(np.array([item[0] for item in results]), columns=feature_names)
    indices = [item[1] for item in results]

    return new_features, indices


if __name__ == '__main__':
    # Read data
    X, y = read_data(r'data/sample_dataset_data_scientist.csv')
    y.to_pickle("data/y.pkl")
    print('Data collection: done.')

    # Check class distribution
    if ARGS.plot_distributions:
        plot_class_distribution(y.value_counts())

    # Process data
    X_NUM = process_numerical_data(X, NUM_DATA)
    X_CAT = process_categorical_data(X, CAT_DATA)
    X_TIME = process_time_data(X)
    X_DATE = process_date_data(X, DATE_DATA)

    # (Concat features
    X_PROCESSED = pd.concat([X_NUM, X_CAT, X_TIME, X_DATE], axis=1)
    X_PROCESSED.to_pickle("data/X_PROCESSED.pkl")
    print('Data processing: done.')

    #  Build new features by User_Id
    grouped = X_PROCESSED.groupby(['v4'])
    unique = X_PROCESSED['v4'].unique().tolist()

    print('Building new features:')
    new_features, indices = build_features_multiprocessing(
        X_PROCESSED, grouped.groups, np, n_workers=int(ARGS.pool_workers))

    # Redefine original dataframe given new features extracted for User_Id.
    X_REDEFINED = reshape_to_original_dataframe(X_PROCESSED, new_features, indices)
    X_REDEFINED.to_pickle("data/X_REDEFINED.pkl")
    print('New features built.')

    # Check feature distributions between classes.
    if ARGS.plot_distributions:
        # Concatenate new features with original ones
        X_FINAL = pd.concat([X_PROCESSED, X_REDEFINED], axis=1)
        X_FINAL = pd.DataFrame(normalize(X_FINAL, norm='max', axis=0), columns=X_FINAL.columns.tolist())
        X_FINAL[['std_v6', 'mean_v6', 'v6']] = np.power(X_FINAL[['std_v6', 'mean_v6', 'v6']], 0.1)

        # Collect data per class and plot respective distributions.
        _, _, X_class_0, X_class_1 = get_balanced_train_data(X_FINAL, y)
        plot_distributions_per_class([X_class_0, X_class_1], X_FINAL.columns)
