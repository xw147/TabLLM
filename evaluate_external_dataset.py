# to do:
# move gpt evaluation into another file
# currently use all set for metrics calculation, may simulate 80/20 for N random times

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
from sympy import python
import xgboost as xgboost
from datasets import DatasetDict, concatenate_datasets, Dataset
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from tabpfn import TabPFNClassifier


from create_external_datasets import load_train_validation_test
from ico_config import ICO_LABEL_STRATEGIES, is_metadata_column, apply_ico_label_strategy
from path_config import GPT_EVAL_INPUT_CSV, GPT_EVAL_OUTPUT_CSV, DATASETS_DIR
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
            # 'n_estimators': [32]
        },
        'gpt3': {
          # Dummy model entry, for zero-shot predictions from gpt3
          # To get run it, set shot size to 4
        #   'dummy': [],
          'model': ['gpt-3.5-turbo-instruct'],  # modern instruct model
          'prompt': ['text'],
          'temperature': [0],
          'max_tokens': [1],
          'logprobs': [5]
        }
    }

    # Hijack parameter for running all
    # args_datasets = ['car', 'income', 'heart', 'diabetes', 'blood', 'bank', 'jungle', 'creditg', 'calhousing']
    args_datasets = ['ico']

    # ── Configuration: set model and label strategy here ──────────────────────
    args.model = 'lightgbm'           # choices: 'lr', 'xgboost', 'lightgbm', 'tabpfn', 'gpt3'
    args.label_strategy = 'all' # choices: 'all', 'high_only', 'low_only'
    # ──────────────────────────────────────────────────────────────────────────

    models = [args.model]
    row_keys = [f"{d}_{args.model}" for d in args_datasets]
    all_results = pd.DataFrame([], index=row_keys)
    all_results_sd = pd.DataFrame([], index=row_keys)
    all_metrics_per_shot = {}
    for args.dataset in args_datasets:
        # Configuration
        data_dir = DATASETS_DIR / args.dataset

        # models defined above
        # models = ['output_datasets']
        ts = datetime.now().strftime("-%Y%m%d-%H%M%S")
        # metric = 'roc_auc'  # accuracy
        metric = 'average_precision' # 'auprc', used for hyperparameter tuning
        num_shots = [4, 8, 16, 32, 64, 128] #, 256, 512, 'all'] # 1024, 2048, 4096, 8192, 16384, 50000, 'all']  # ['all']
        seeds = [42, 1024, 0, 1, 32]   # , 45, 655, 186, 126, 836]
        if metric == 'roc_auc' and args.dataset == 'car':
            # This computes the roc_auc_score for ovr on macro level:
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
            metric = 'roc_auc_ovr'
        for model in models:
            seeded_results = defaultdict(list)  # Reset per model

            # ── GPT-3 / GPT-4o: evaluate directly from output CSV, skip dataset pipeline ──
            if model == 'gpt3':
                evaluate_gpt3_from_csv(GPT_EVAL_INPUT_CSV, args.dataset, output_file=GPT_EVAL_OUTPUT_CSV)
                continue
            # ────────────────────────────────────────────────────────────────────────────────

            # Set ordinal encoding based on model name
            categorical_encoding = 'ordinal' # 'one-hot'
            if model.endswith(' ordinal'):
                model = model.split(' ordinal', maxsplit=1)[0]
                categorical_encoding = 'ordinal'
            label_strategy_info = f", label_strategy='{args.label_strategy}'" if args.dataset == 'ico' else ""
            print(f"Evaluate dataset {args.dataset} with model {model} and encoding {categorical_encoding}{label_strategy_info}.")
            for i, seed in enumerate(seeds):

                # Need to replicate exactly tfew cohorts here by following create_external_dataset and tfew code.
                # 1. Load data as in create_external_datasets to get same splits
                dataset = load_train_validation_test(args.dataset, data_dir,
                                                     ico_label_strategy=args.label_strategy)
                # Strip metadata-only columns before ML training (kept only for analysis)
                dataset = {k: v[[c for c in v.columns if not is_metadata_column(c)]] for k, v in dataset.items()}
                if i == 0:
                    print(f"Original columns: {list(dataset['train'].columns)}")
                dataset = DatasetDict({k: Dataset.from_pandas(v, preserve_index=False) for k, v in dataset.items()})
                dataset = concatenate_datasets(list(dataset.values()))
                # 2. Apply method from tfew loading, but skip the data loading from disk (mainly caps validation set)
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
                    # Store the original num_shot for output file naming
                    original_num_shot = num_shot
                    if num_shot == 'all' and model == 'tabpfn':
                        num_shot = 1024  # This is the expected maximum input size of tabpfn

                    # Skip if this (model, dataset, num_shot, seed, label_strategy) run already exists
                    run_path = get_run_dir(model, args.dataset, original_num_shot, seed, args.label_strategy)
                    if (run_path / "dev_scores.json").exists():
                        print(f"\tSkipping (already exists): {run_path}/dev_scores.json")
                        # Still need to populate seeded_results for summary printing
                        with open(run_path / "dev_scores.json") as _f:
                            _s = json.load(_f)
                        seeded_results[original_num_shot].append(_s["AUC"])
                        continue
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
                    gpt_3_prob_idx = []
                    if model == 'gpt3':
                        # Offset by one cause label removed later
                        gpt_3_label_idx = [(i - 1) for i, x in enumerate(list(dataset_train.column_names)) if x.startswith('gpt3_pred_label')]
                        gpt_3_prob_idx = [(i - 1) for i, x in enumerate(list(dataset_train.column_names)) if x.startswith('gpt3_pos_prob')]

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
                        save_run_results(model, args.dataset, original_num_shot, seed, args.label_strategy, all_metrics, len(y_test))
                        results = primary_metric  # For backward compatibility
                    else:
                        # Handle GPT-3 case as before
                        y_pred = X_test[:, gpt_3_label_idx]
                        y_proba = X_test[:, gpt_3_prob_idx]

                        df_test = pd.DataFrame(np.column_stack([y_test, y_pred, y_proba]), columns=['y_test', 'y_pred', 'y_proba'])
                        print(df_test)

                        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
                        auprc = average_precision_score(y_test, y_proba)
                        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        auroc = roc_auc_score(y_test, y_proba)
                        f1_binary = f1_score(y_test, y_pred, average='binary', zero_division=0)
                        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                        acc = accuracy_score(y_test, y_pred)
                        all_metrics = {
                        'precision': precision,
                        'recall': recall,
                        'f1_binary': f1_binary,
                        'auprc': auprc,
                        'specificity': specificity,
                        'auroc': auroc,
                        'macro_f1': macro_f1,
                        'accuracy': acc
                        }

                        if metric == 'roc_auc':
                            results = roc_auc_score(y_test, y_proba)
                        elif metric == 'roc_auc_ovr':
                            results = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                        elif metric == 'average_precision':
                            results = average_precision_score(y_test, y_proba)

                        save_run_results(model, args.dataset, original_num_shot, seed, args.label_strategy, all_metrics, len(y_test))

                    seeded_results[num_shot] = seeded_results[num_shot] + [results]

            # Collect outputs per dataset+model
            row_key = f"{args.dataset}_{model}"
            for k, v in seeded_results.items():
                all_results.loc[row_key, str(k)] = round(float(np.mean(v)), 2)
                all_results_sd.loc[row_key, str(k)] = round(float(np.std(v)), 2)

    
    # Print the collective results (one row per dataset+model combination)
    label_tag = f" [label_strategy={args.label_strategy}]" if args_datasets == ['ico'] else ""
    print(f"\nRow-wise results{label_tag} for shots: {list(all_results.columns)}.")
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

        df_test = pd.DataFrame(np.column_stack([y_test, y_pred, y_proba]), columns=['y_test', 'y_pred', 'y_proba'])
        print(df_test)
        
        # Calculate precision, recall, f1 for binary classification
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
        auprc = average_precision_score(y, y_proba)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        auroc = roc_auc_score(y, y_proba)
        f1_binary = f1_score(y, y_pred, average='binary', zero_division=0)
        macro_f1 = f1_score(y, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y, y_pred)
        
        all_metrics = {
            'precision': precision,
            'recall': recall,
            'f1_binary': f1_binary,
            'auprc': auprc,
            'specificity': specificity,
            'auroc': auroc,
            'macro_f1': macro_f1,
            'accuracy': acc
        }
        
        # Return specific metric for compatibility with existing code
        if metric == 'precision':
            return all_metrics['precision']
        elif metric == 'recall':
            return all_metrics['recall']
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
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auroc = roc_auc_score(y_test, y_proba)
    f1_binary = f1_score(y_test, y_pred, average='binary', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    
    all_metrics = {
        'precision': precision,
        'recall': recall,
        'f1_binary': f1_binary,
        'auprc': auprc,
        'specificity': specificity,
        'auroc': auroc,
        'macro_f1': macro_f1,
        'accuracy': acc
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

    # Impute NaNs with column median — only for lr (which does not handle NaN natively).
    all_feature_cols = [c for c in dataset['train'].columns if c not in ['idx', 'label']]
    if model_name == 'lr' and dataset['train'][all_feature_cols].isnull().any().any():
        imputer = SimpleImputer(strategy='median')
        imputer.fit(dataset['train'][all_feature_cols])
        dataset['train'][all_feature_cols] = imputer.transform(dataset['train'][all_feature_cols])
    else:
        imputer = None

    if scale and len(numeric_columns) > 0:
        # z-normalization of numerical columns
        scaler = StandardScaler()
        scaler.fit(dataset['train'][numeric_columns])
        dataset['train'][numeric_columns] = scaler.transform(dataset['train'][numeric_columns])

    def create_valid_test(split):
        # Replicate steps performed on training data
        data = dataset[split]
        data = encode_categorical(data)
        if imputer is not None and data.shape[0] > 0:
            imp_cols = [c for c in all_feature_cols if c in data.columns]
            data[imp_cols] = imputer.transform(data[imp_cols])
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
    # Strip metadata columns before splitting — they are not model features
    meta_cols = [c for c in orig_data.column_names if is_metadata_column(c)]
    if meta_cols:
        orig_data = orig_data.remove_columns(meta_cols)
    data = orig_data.train_test_split(test_size=0.20, seed=seed)
    data2 = data['test'].train_test_split(test_size=0.50, seed=seed)
    # Build an empty test slice with the correct schema
    empty_test = data['train'].select([])
    # No validation/test split used for external datasets
    dataset_dict = DatasetDict({'train': data['train'],
                                'validation': concatenate_datasets([data2['train'], data2['test']]),
                                'test': empty_test})
    orig_data = dataset_dict[split]

    # In case dataset has no idx per example, add that here bc manually created ones might not have an idx.
    if 'idx' not in orig_data.column_names:
        n = orig_data.num_rows
        if n > 0:
            orig_data = orig_data.add_column(name='idx', column=list(range(n)))
        else:
            # add_column fails on 0-row datasets in this version of datasets;
            # reconstruct via from_dict which handles empty tables correctly.
            schema = {col: [] for col in orig_data.column_names}
            schema['idx'] = []
            orig_data = datasets.Dataset.from_dict(schema)

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




def evaluate_gpt3_from_csv(gpt_csv_path, dataset_name, output_file=None):
    """
    Evaluate GPT predictions from a pre-computed output CSV.

    Derives ground truth directly from the 'riskLevel' column in the CSV for
    all three ICO label strategies ('all', 'high_only', 'low_only'), so that
    row alignment between the GPT run and the raw dataset is not required.

    Required CSV columns: riskLevel, pred_label, pos_prob

    Saves a summary CSV with one row per strategy.
    Returns the summary DataFrame.
    """
    import os

    df = pd.read_csv(gpt_csv_path)

    required_cols = {'riskLevel', 'pred_label', 'pos_prob'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"GPT output CSV missing required columns: {missing}")

    # Ensure numeric types
    df['riskLevel'] = pd.to_numeric(df['riskLevel'], errors='coerce')
    df['pred_label'] = pd.to_numeric(df['pred_label'], errors='coerce').astype(int)
    df['pos_prob'] = pd.to_numeric(df['pos_prob'], errors='coerce')

    rows = []
    for strategy_name, strategy in ICO_LABEL_STRATEGIES.items():
        df_strat = df.copy()
        # Drop rows whose riskLevel is excluded by this strategy
        if strategy['drop']:
            df_strat = df_strat[~df_strat['riskLevel'].isin(strategy['drop'])].reset_index(drop=True)

        y_test = df_strat['riskLevel'].isin(strategy['positive']).astype(int).values
        y_pred = df_strat['pred_label'].values
        y_proba = df_strat['pos_prob'].values

        n_total = len(y_test)
        n_pos = int(y_test.sum())
        n_neg = n_total - n_pos
        print(f"\n[gpt3] strategy={strategy_name}  n={n_total}  pos={n_pos}  neg={n_neg}")

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        auprc = average_precision_score(y_test, y_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        auroc = roc_auc_score(y_test, y_proba)
        f1_binary = f1_score(y_test, y_pred, average='binary', zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        row = {
            'dataset': dataset_name,
            'strategy': strategy_name,
            'n_total': n_total,
            'n_positive': n_pos,
            'n_negative': n_neg,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'auprc': round(auprc, 4),
            'specificity': round(specificity, 4),
            'auroc': round(auroc, 4),
            'f1_binary': round(f1_binary, 4),
            'macro_f1': round(macro_f1, 4),
            'accuracy': round(acc, 4),
        }
        for k, v in row.items():
            print(f"  {k}: {v}")
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = f"output/{dataset_name}_gpt3_metrics_summary_{ts}.csv"
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    summary_df.to_csv(output_file, index=False)
    print(f"\nGPT metrics summary saved to {output_file}")
    return summary_df


def get_run_dir(model, dataset, num_shot, seed, label_strategy):
    """Return the Path for a run's output directory (no side effects)."""
    label_strategy_tag = label_strategy if dataset == 'ico' else 'default'
    return Path("exp_out") / f"{model}_{dataset}_numshot{num_shot}_seed{seed}_{label_strategy_tag}"


def save_run_results(model, dataset, num_shot, seed, label_strategy, all_metrics, num_test):
    """Save per-run results in t-few dev_scores.json format.

    Output path: exp_out/{model}_{dataset}_numshot{N}_seed{S}_{label_strategy}/dev_scores.json

    This mirrors the t-few experiment output structure so get_result_table.py can
    aggregate results across seeds using the same logic for both t-few and ML models.
    """
    run_dir = get_run_dir(model, dataset, num_shot, seed, label_strategy)
    run_dir.mkdir(parents=True, exist_ok=True)
    scores = {
        "AUC":         float(all_metrics["auroc"]),
        "PR":          float(all_metrics["auprc"]),
        "macro_f1":    float(all_metrics["macro_f1"]),
        "f1_binary":   float(all_metrics["f1_binary"]),
        "sensitivity": float(all_metrics["recall"]),
        "specificity": float(all_metrics["specificity"]),
        "precision":   float(all_metrics["precision"]),
        "accuracy":    float(all_metrics["accuracy"]),
        "num":         int(num_test),
    }
    with open(run_dir / "dev_scores.json", "w") as f:
        json.dump(scores, f)
    print(f"\tSaved run results → {run_dir}/dev_scores.json")

if __name__ == '__main__':
   
    main()

    # ML baseline with strict fraud only (riskLevel 2 & 3)
    # python evaluate_external_dataset.py --dataset ico --label_strategy high_only