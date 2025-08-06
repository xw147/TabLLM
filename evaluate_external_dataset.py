import argparse
from datetime import datetime
import logging
import pickle
import json
from collections import Counter, defaultdict
from pathlib import Path
import datasets
import numpy as np
import pandas as pd
import xgboost as xgboost
from datasets import DatasetDict, concatenate_datasets, Dataset
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from tabpfn import TabPFNClassifier


from create_external_datasets import load_train_validation_test
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

datasets.enable_caching()
logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Hyper-parameters
    parameters = {
        'lr': {
            'penalty': ['l1', 'l2'],
            'C': [100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        },
        'lightgbm': {
            'num_leaves': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
            'lambda_l1': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10.],
            'lambda_l2': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10.],
            'learning_rate': [0.01, 0.03, 0.1, 0.3],
        },
        'xgboost': {
            # 'max_depth': [2, 4, 6, 8, 10, 12],
            # 'alpha': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.],
            # 'lambda': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.],
            # 'eta': [0.01, 0.03, 0.1, 0.3],

            'max_depth': [2, 3, 4, 5],
            'eta': [0.01, 0.1, 0.3], # learning rate
            'alpha': [0.1, 1.0, 10.0],      # L1 regularization
            'lambda': [0.1, 1.0, 10.0],   # L2 regularization
        },
        'tabpfn': {
            'device': ['cuda'],
            'n_estimators': [32]
        },
        'gpt3': {
          # Dummy model entry, for zero-shot predictions from gpt3
          # To get run it, set shot size to 4
          'dummy': []
        }
    }

    # Hijack parameter for running all
    # args_datasets = ['car', 'income', 'heart', 'diabetes', 'blood', 'bank', 'jungle', 'creditg', 'calhousing']
    args_datasets = ['ico']
    all_results = pd.DataFrame([], index=args_datasets)
    all_results_sd = pd.DataFrame([], index=args_datasets)
    all_metrics_per_shot = {}
    for args.dataset in args_datasets:
        # Configuration
        data_dir = Path("/work/TabLLM/datasets")
        data_dir = data_dir / args.dataset

        models = ['tabpfn']
        assert(len(models)) == 1  # For current output only one model is supported
        # models = ['output_datasets']
        ts = datetime.now().strftime("-%Y%m%d-%H%M%S")
        # metric = 'roc_auc'  # accuracy
        metric = 'average_precision' # 'auprc', used for hyperparameter tuning
        num_shots = [4, 8, 16, 32, 64, 128, 'all'] #, 256, 512, 1024, 2048, 4096, 8192, 16384, 50000, 'all']  # ['all']
        seeds = [42, 1024, 0, 1, 32]   # , 45, 655, 186, 126, 836]
        seeded_results = defaultdict(list)
        if metric == 'roc_auc' and args.dataset == 'car':
            # This computes the roc_auc_score for ovr on macro level:
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
            metric = 'roc_auc_ovr'
        for model in models:
            all_metrics_per_shot = {}  # Initialize for each model
            # Set ordinal encoding based on model name
            categorical_encoding = 'ordinal' # 'one-hot'
            if model.endswith(' ordinal'):
                model = model.split(' ordinal', maxsplit=1)[0]
                categorical_encoding = 'ordinal'
            print(f"Evaluate dataset {args.dataset} with model {model} and encoding {categorical_encoding}.")
            for i, seed in enumerate(seeds):

                # Need to replicate exactly tfew cohorts here by following create_external_dataset and tfew code.
                # 1. Load data as in create_external_datasets to get same splits
                dataset = load_train_validation_test(args.dataset, data_dir)
                if i == 0:
                    print(f"Original columns: {list(dataset['train'].columns)}")
                dataset = DatasetDict({k: Dataset.from_pandas(v, preserve_index=False) for k, v in dataset.items()})
                dataset = concatenate_datasets(list(dataset.values()))
                # 2. Apply method from tfew loading, but skip the data loading from disk (mainly caps validation set)
                if model == 'gpt3':
                    dataset = add_gpt3_zero_shot_predictions(dataset, args.dataset, data_dir)
                dataset = DatasetDict({k: read_orig_dataset(dataset, seed, k) for k in ['train', 'validation', 'test']})

                # Prepare data specifically for model and dataset, for this back to pandas again
                dataset = DatasetDict({k: v.to_pandas() for k, v in dataset.items()})
                dataset = prepare_data(args.dataset, model, dataset, enc=categorical_encoding, scale=model not in ['tabpfn', 'gpt3'])
                if i == 0:
                    print(f"prepared columns: {list(dataset['train'].columns)}")

                # Load into hf datasets to replicate tfew methods
                dataset = {k: Dataset.from_pandas(v, preserve_index=False) for k, v in dataset.items()}
                dataset = DatasetDict(dataset)

                dataset_validation = dataset['validation'].remove_columns(['idx'])
                dataset_test = dataset['test'].remove_columns(['idx'])

                for num_shot in num_shots:
                    if num_shot not in all_metrics_per_shot:
                        all_metrics_per_shot[num_shot] = []  # Initialize list for each shot size
                    # Store the original num_shot for metrics tracking
                    original_num_shot = num_shot
                    if num_shot == 'all' and model == 'tabpfn':
                        num_shot = 1024  # This is the expected maximum input size of tabpfn
                    if num_shot == 'all':
                        # Just shuffle dataset and flatten indices for better performance
                        dataset_train = (dataset['train'].remove_columns(['idx'])).shuffle(seed)
                        dataset_validation = dataset_validation.shuffle(seed).flatten_indices()
                        dataset_test = dataset_test.shuffle(seed).flatten_indices()
                        X_unlab = np.array([])
                    else:
                        # 3. Sample few shot examples as in tfew
                        old_dataset_train = dataset['train']
                        dataset_train = sample_few_shot_data(dataset['train'], num_shot, seed)
                        dataset_train = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset_train))
                        # Debug show idx and labels of selected subsets
                        debug_examples = [str({'label': x['label'], 'idx': x['idx']}) for i, x in enumerate(dataset_train) if i < 8]
                        print(f"\t{seed}: {'; '.join(debug_examples)}")
                        # Put remaining training examples into unlabeled
                        X_unlab = old_dataset_train.filter(lambda ex: ex['idx'] not in dataset_train['idx'])
                        dataset_train = dataset_train.remove_columns(['idx'])

                    if model == 'output_datasets':
                        folder = Path('./numpy_datasets/' + args.dataset + ts)
                        folder.mkdir(parents=True, exist_ok=True)
                        with open(folder / (args.dataset + '_numshot' + str(num_shot) + '_seed' + str(seed) + '.p'), 'wb') as f:
                            X_train = dataset_train.remove_columns(['label'])
                            y_train = list(dataset_train['label'])
                            X_valid = dataset_validation.remove_columns(['label']).to_pandas()
                            y_valid = list(dataset_validation['label'])
                            X_test = dataset_test.remove_columns(['label']).to_pandas()
                            y_test = list(dataset_test['label'])
                            # Use 20% for testing and use cv during training
                            X_test = pd.concat([X_valid, X_test], axis=0, ignore_index=True)
                            y_test = y_valid + y_test
                            data_export = {'x_train': X_train, 'y_train': y_train, 'x_unlab': X_unlab,
                                           'x_test': X_test, 'y_test': y_test}
                            pickle.dump(data_export, f)
                            # np.savez(f, x_train=X_train, y_train=y_train, x_unlab=X_unlab, x_test=X_test, y_test=y_test)
                        continue

                    gpt_3_label_idx = []
                    if model == 'gpt3':
                        # Offset by one cause label removed later
                        gpt_3_label_idx = [(i - 1) for i, x in enumerate(list(dataset_train.column_names)) if x.startswith('gpt3_output')]

                    # In depth debug linear model
                    # print(list(dataset_train.remove_columns(['label']).to_pandas().columns))
                    X_train = dataset_train.remove_columns(['label']).to_pandas().to_numpy()
                    y_train = np.array(dataset_train['label'])
                    X_valid = dataset_validation.remove_columns(['label']).to_pandas().to_numpy()
                    y_valid = np.array(dataset_validation['label'])
                    X_test = dataset_test.remove_columns(['label']).to_pandas().to_numpy()
                    y_test = np.array(dataset_test['label'])
                    # Use 20% for testing and use cv during training
                    X_test = np.concatenate([X_valid, X_test], axis=0)
                    y_test = np.concatenate([y_valid, y_test], axis=0).astype(int)
                    X_valid = np.array([])
                    y_valid = np.array([])

                    # Replace the existing evaluation section:
                    if model != 'gpt3':
                        primary_metric, all_metrics = evaluate_model(seed, model, metric, parameters[model], 
                                                                    X_train, y_train, X_valid, y_valid, X_test, y_test)
                        # Store all metrics for this shot/seed combination
                        all_metrics_per_shot[original_num_shot].append(all_metrics)
                        results = primary_metric  # For backward compatibility
                    else:
                        # Handle GPT-3 case as before
                        if metric == 'roc_auc':
                            results = roc_auc_score(y_test, X_test[:, gpt_3_label_idx])
                        elif metric == 'roc_auc_ovr':
                            results = roc_auc_score(y_test, X_test[:, gpt_3_label_idx], multi_class='ovr', average='macro')

                    seeded_results[num_shot] = seeded_results[num_shot] + [results]

            # Save results after all experiments for this model are completed
            if model != 'gpt3' and model != 'output_datasets':
                model_params = parameters[model] if model in parameters else {}
                dataset_info = {
                    "dataset_name": args.dataset,
                    "n_features": X_train.shape[1] if 'X_train' in locals() else None,
                    "classification_type": "binary"
                }
                
                save_model_results(
                    model_name=model,
                    all_metrics_per_shot=all_metrics_per_shot,
                    model_params=model_params,
                    dataset_info=dataset_info,
                    output_file=f"{args.dataset}_{model}_results.json"
                )

            # Collect outputs per dataset
            for k, v in seeded_results.items():
                all_results.loc[args.dataset, str(k)] = round(float(np.mean(v)), 2)
                all_results_sd.loc[args.dataset, str(k)] = round(float(np.std(v)), 2)

    
    # Print the collective results for one model
    print(f"\nRow-wise results for shots: {list(all_results.columns)}.")
    for i, row in all_results.iterrows():
        print(i, end=': ')
        for j in range(0, len(row)):
            sd = f"{all_results_sd.loc[i].iloc[j]:.2f}"[1:]
            print(f"& ${row.iloc[j]:.2f}_{{{sd}}}$   ", end='')
        print('')

    print(f"\nAveraged results for shots: {list(all_results.columns)}.")
    # Print the averaged results
    for c in list(all_results.columns):
        print(f"({np.mean(all_results[c]):.2f}, {np.std(all_results[c]):.2f}), ", end='')


def evaluate_model(seed, model, metric, parameters, X_train, y_train, X_valid, y_valid, X_test, y_test):
    print(f"\tUse {X_train.shape[0]} train, {X_valid.shape[0]} valid, and {X_test.shape[0]} test examples.")

    def get_lr():
        # Kept balanced bc for all 'shot' experiments no effect, only used for 'all' which should be better.
        # In contrast to IBC: removed tol=1e-1
        return LogisticRegression(class_weight='balanced', penalty='l1', fit_intercept=True, solver='liblinear',
                                  random_state=seed, verbose=0, max_iter=200)

    def get_light_gbm():
        return lgb.LGBMClassifier(class_weight='balanced', num_threads=1, random_state=seed)

    def get_xgboost():
        # No class_weight parameter, only scale_pos_weight, but should not be a difference for all shot experiments.
        # eval_metric gives same as non, but without a warning
        return xgboost.XGBClassifier(eval_metric='logloss', nthread=1, random_state=seed)

    def get_tabpfn():
        # Use default configuration
        return TabPFNClassifier()

    def compute_metric(clf_in, X, y):
        
        
        # Get predictions and probabilities for binary classification
        y_pred = clf_in.predict(X)
        y_proba = clf_in.predict_proba(X)[:, 1]  # Probability of positive class
        
        # Calculate precision, recall, f1 for binary classification
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
        auprc = average_precision_score(y, y_proba)
        
        all_metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auprc': auprc
        }
        
        # Return specific metric for compatibility with existing code
        if metric == 'precision':
            return all_metrics['precision']
        elif metric == 'recall':
            return all_metrics['recall']
        elif metric == 'f1_score':
            return all_metrics['f1_score']
        elif metric == 'average_precision':
            return all_metrics['auprc']
        else:
            # For other metrics, keep existing behavior
            if metric == 'roc_auc':
                return roc_auc_score(y, y_proba)
            elif metric == 'accuracy':
                return np.sum(y_pred == np.array(y)) / len(y)
            else:
                raise ValueError("Undefined metric.")

    # Do a 4-fold cross validation on the training set for parameter tuning
    folds = min(Counter(y_train).values()) if min(Counter(y_train).values()) < 4 else 4 # If less than 4 examples
    if folds < 4:
        print(f"Manually reduced folds to {folds} since this is maximum number of labeled examples.")

    if folds > 1:
        inner_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    else:
        print(f"Warning: Increased folds from {folds} to 2 (even though not enough labels) and use simple KFold.")
        folds = 2
        inner_cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    estimator = None
    if model == 'lr':
        estimator = get_lr()
    elif model == 'lightgbm':
        estimator = get_light_gbm()
    elif model == 'xgboost':
        estimator = get_xgboost()
    elif model == 'tabpfn':
        estimator = get_tabpfn()
    # Add verbose 4 for more detailed output

    clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=inner_cv, 
                   scoring=metric, n_jobs=40, verbose=0)
    clf.fit(X_train, y_train)

    # Calculate all metrics for binary classification
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Calculate all metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    auprc = average_precision_score(y_test, y_proba)
    
    all_metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auprc': auprc
    }
    
    # Return the specific metric being evaluated plus all metrics
    if metric in all_metrics:
        return all_metrics[metric], all_metrics
    else:
        # For backward compatibility with other metrics
        score_test = compute_metric(clf, X_test, y_test)
        return score_test, all_metrics

    #     one_hot_test = pd.get_dummies(dataset['test'][[c for c in list(dataset['test']) if c != 'label']])[columns]
    #     pred = np.argmax(lr.predict_proba(one_hot_test), axis=1)
    #     acc_test = np.sum(pred == np.array(dataset['test']['label'].tolist()))/pred.shape[0]
    #     print("C: %.4f, Val acc: %.2f, Test acc: %.2f" % (C, acc_valid, acc_test))
    # return


def prepare_data(dataset_name, model_name, dataset, enc=None, scale=True):
    def remove_columns(data_dict, columns):
        return {k: v[[c for c in list(v) if c not in columns]] for k, v in data_dict.items()}

    def remove_constants(data):
        return data[[c for c in data if data[c].nunique() > 1]]

    # Preprocessing
    if dataset_name == "titanic":
        # These do not add predictive value, Cabin should still provide some information
        dataset = remove_columns(dataset, ['Name', 'Ticket'])
        # Manually add nan column for missing age entries
        def replace_age_nans(data):
            data['Age_nan'] = (pd.isna(data['Age']))
            data = data.fillna({'Age': 0})
            return data
        dataset = {k: replace_age_nans(v) for k, v in dataset.items()}

    numeric_columns = [c for c in dataset['train'].select_dtypes(include=np.number).columns.tolist() if c not in ['idx', 'label']]
    cat_columns = [c for c in dataset['train'].columns.tolist() if (c not in (numeric_columns + ['idx', 'label']))]

    # Encoding of categorical values
    if enc == 'ordinal':
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        ordinal_encoder.fit(dataset['train'][cat_columns])

    def encode_categorical(data):
        if enc is not None and model_name != 'output_datasets':
            if enc == 'one-hot':
                # 0-1 encoding for categorical columns
                data = remove_constants(pd.get_dummies(data, dummy_na=True))
            elif enc == 'ordinal' and data.shape[0] > 0:
                # ordinal encoding
                data[cat_columns] = ordinal_encoder.transform(data[cat_columns])
        return data
    dataset['train'] = encode_categorical(dataset['train'])
    if model_name == 'tabpfn' and len(dataset['train'].columns) > 100:
        sparse_cols = (np.argsort(np.count_nonzero(dataset['train'], axis=0)))[:-100]
        print(f"Found {len(dataset['train'].columns)}, tabpfn can only handle 100, remove {len(sparse_cols)} sparsest.")
        dataset['train'].drop(columns=dataset['train'].columns[sparse_cols], inplace=True)

    if scale and len(numeric_columns) > 0:
        # z-normalization of numerical columns
        scaler = StandardScaler()
        scaler.fit(dataset['train'][numeric_columns])
        dataset['train'][numeric_columns] = scaler.transform(dataset['train'][numeric_columns])

    def create_valid_test(split):
        # Replicate steps performed on training data
        data = dataset[split]
        data = encode_categorical(data)
        if scale and len(numeric_columns) > 0 and data.shape[0] > 0:
            # z-normalization of numerical columns
            data[numeric_columns] = scaler.transform(data[numeric_columns])

        # Find missing columns
        missing_columns = [c for c in dataset['train'].columns if c not in data.columns]
        
        # Add all missing columns at once using pd.concat instead of one by one
        if missing_columns:
            # Create DataFrame with missing columns filled with 0
            missing_data = pd.DataFrame(0, index=data.index, columns=missing_columns)
            # Concatenate all at once
            data = pd.concat([data, missing_data], axis=1)
        
        # Put columns in same order as train dataset
        return data[dataset['train'].columns]

    dataset['validation'] = create_valid_test('validation')
    dataset['test'] = create_valid_test('test')
    assert (len(dataset['train'].columns) == len(dataset['validation'].columns) == len(dataset['test'].columns))
    return dataset


def read_orig_dataset(orig_data, seed, split):
    # External datasets are not yet shuffled, so do it now
    data = orig_data.train_test_split(test_size=0.20, seed=seed)
    data2 = data['test'].train_test_split(test_size=0.50, seed=seed)
    # No validation/test split used for external datasets
    dataset_dict = DatasetDict({'train': data['train'],
                                'validation': concatenate_datasets([data2['train'], data2['test']]),
                                'test': Dataset.from_dict({'note': [], 'label': []})})
    orig_data = dataset_dict[split]

    # In case dataset has no idx per example, add that here bc manually created ones might not have an idx.
    if 'idx' not in orig_data.column_names:
        orig_data = orig_data.add_column(name='idx', column=range(0, orig_data.num_rows))

    return orig_data


def sample_few_shot_data(orig_data, num_shot, few_shot_random_seed):
    saved_random_state = np.random.get_state()
    np.random.seed(few_shot_random_seed)
    # Create a balanced dataset for categorical data
    labels = {label: len([ex['idx'] for ex in orig_data if ex['label'] == label])
              for label in list(set(ex['label'] for ex in orig_data))}
    num_labels = len(labels.keys())
    ex_label = int(num_shot / num_labels)
    ex_last_label = num_shot - ((num_labels - 1) * ex_label)
    ex_per_label = (num_labels - 1) * [ex_label] + [ex_last_label]
    assert sum(ex_per_label) == num_shot

    # Select num instances per label
    old_num_labels = []
    datasets_per_label = []
    for i, label in enumerate(labels.keys()):
        indices = [ex['idx'] for ex in orig_data if ex['label'] == label]
        old_num_labels.append(len(indices))
        # Sample with replacement from label indices
        samples_indices = list(np.random.choice(indices, ex_per_label[i]))
        datasets_per_label.append(orig_data.select(samples_indices))
    orig_data = concatenate_datasets(datasets_per_label)

    # Check new labels
    old_labels = labels
    labels = {label: len([ex['idx'] for ex in orig_data if ex['label'] == label])
              for label in list(set(ex['label'] for ex in orig_data))}
    print(f"Via sampling with replacement old label distribution {old_labels} to new {labels}")
    assert sum(labels.values()) == num_shot
    assert len(orig_data) == num_shot

    np.random.set_state(saved_random_state)
    # Now randomize and (selection of num_shots redundant now bc already done).
    # Call to super method directly inserted here
    saved_random_state = np.random.get_state()
    np.random.seed(few_shot_random_seed)
    orig_data = [x for x in orig_data]
    np.random.shuffle(orig_data)
    selected_data = orig_data[: num_shot]
    np.random.set_state(saved_random_state)
    return selected_data


def add_gpt3_zero_shot_predictions(dataset, task, data_dir):
    gpt3_output = pd.read_csv(data_dir.parent / 'gpt-3-zero-shot' / ('outputs-' + task + '.csv'))
    if task == 'car':
        splitted_predictions = [[], [], [], []]
        for p in gpt3_output['output0'].to_list():
            preds = [float(x) for x in p.split(', ')]
            preds = [p / sum(preds) for p in preds]
            for i, k in enumerate(preds):
                splitted_predictions[i].append(k)
        for i, l in enumerate(splitted_predictions):
            dataset = dataset.add_column('gpt3_output' + str(i), l)
        print('')
    else:
        dataset = dataset.add_column('gpt3_output', gpt3_output['output0'])

    return dataset


def result_str(scores):
    if len(scores) > 1:
        return f"{np.mean(scores):.2f} ({np.std(scores):.2f})"
    else:
        return f"{scores[0] * 100:.2f}"


def parse_args():
    parser = argparse.ArgumentParser(description="Create note dataset from cohort.")
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        type=str
    )

    args = parser.parse_args()

    return args




def save_model_results(model_name, all_metrics_per_shot, model_params=None, 
                      dataset_info=None, output_file=None):
    """
    Save ML model results to JSON file for binary classification
    
    Args:
        model_name: Name of the ML model
        all_metrics_per_shot: Dictionary with shot sizes as keys and list of metric dictionaries as values
                             e.g., {2: [{'precision': 0.8, 'recall': 0.7, ...}, ...], 4: [...], ...}
        model_params: Dictionary of model hyperparameters
        dataset_info: Dictionary with dataset information
        output_file: Path to save JSON file
    """
    
    # Process metrics for each shot size
    processed_results = {}
    metric_names = ['precision', 'recall', 'f1_score', 'auprc']
    
    for shot_size, metrics_list in all_metrics_per_shot.items():
        processed_results[str(shot_size)] = {}
        
        # Calculate mean and std for each metric
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list]
            processed_results[str(shot_size)][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "raw_values": [float(v) for v in values],
                "count": len(values)
            }
    
    # Create results structure
    results = {
        "model_info": {
            "model_name": model_name,
            "model_type": "binary_classification",
            "hyperparameters": model_params or {},
            "training_time": datetime.now().isoformat()
        },
        "experimental_setup": dataset_info or {},
        "results_by_shot_size": processed_results,
        "summary_statistics": {
            "metric_names": metric_names,
            "shot_sizes": list(all_metrics_per_shot.keys()),
            "seeds_per_shot": len(list(all_metrics_per_shot.values())[0]) if all_metrics_per_shot else 0
        },
        "metadata": {
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    }
    
    # Save to file
    if output_file is None:
        output_file = f"{model_name.lower().replace(' ', '_')}_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return results

if __name__ == '__main__':
    main()