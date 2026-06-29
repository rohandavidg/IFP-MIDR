#!/usr/bin/env python3
"""
IFP-MIDR Analysis Pipeline
===========================
Runs embedding CV comparison, HPO for all models (random + gene splits),
assembles result_all_df, and saves outputs to ../output/.

Usage:
    python run_ifp_midr_pipeline.py [--skip-cv] [--skip-hpo] [--split random|gene|both]

Arguments:
    --skip-cv     Skip embedding CV comparison (load from CSV if available)
    --skip-hpo    Skip HPO (load from saved JSON files)
    --split       Which split strategy to run for HPO (default: both)
"""

import os
import sys
import ast
import json
import joblib
import argparse
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for cluster use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon, kruskal
from scikit_posthocs import posthoc_dunn
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, GroupShuffleSplit, StratifiedShuffleSplit
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
EMB_JSON      = 'AF2_data_with_emb_l1_l2_average.json'
FEATURES_CSV  = '../alphafold/AF2_disorder_all_features.csv'
DBNSFP_TSV    = '../dbNSFP/dbNSFP4.8a_variant.disorder_genes.tsv'
NOEMB_CSV     = 'results_noemb_results.initial.multiple_models.csv'
OUTPUT_DIR    = Path('../output')
HPO_DIR       = Path('xgboost_results')
PRED_DIR      = OUTPUT_DIR / 'test_predictions'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HPO_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {'standalone': '#2D6A4F', 'enhanced': '#C1121F'}

# ── CLI args ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--skip-cv',  action='store_true', help='Skip embedding CV')
    p.add_argument('--skip-hpo', action='store_true', help='Skip HPO')
    p.add_argument('--split', default='both', choices=['random', 'gene', 'both'])
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FEATURE MATRIX HELPERS
# ══════════════════════════════════════════════════════════════════════════════

ABS_DELTA_COLS = [
    'abs_delta_asphericity', 'abs_delta_radius_of_gyration',
    'abs_delta_end_to_end_distance', 'abs_delta_scaling_exponent',
    'abs_delta_prefactor', 'abs_delta_hpi_1.5_frac', 'abs_delta_hpi_1.5',
    'abs_delta_hpi_2.0_frac', 'abs_delta_hpi_2.0', 'abs_delta_hpi_2.5_frac',
    'abs_delta_hpi_2.5', 'abs_delta_length',
    'abs_delta_fraction_A', 'abs_delta_fraction_C', 'abs_delta_fraction_D',
    'abs_delta_fraction_E', 'abs_delta_fraction_F', 'abs_delta_fraction_G',
    'abs_delta_fraction_H', 'abs_delta_fraction_I', 'abs_delta_fraction_K',
    'abs_delta_fraction_L', 'abs_delta_fraction_M', 'abs_delta_fraction_N',
    'abs_delta_fraction_P', 'abs_delta_fraction_Q', 'abs_delta_fraction_R',
    'abs_delta_fraction_S', 'abs_delta_fraction_T', 'abs_delta_fraction_V',
    'abs_delta_fraction_W', 'abs_delta_fraction_Y', 'abs_delta_IEP',
    'abs_delta_molecular_weight', 'abs_delta_gravy', 'abs_delta_Asx',
    'abs_delta_Glx', 'abs_delta_Xle', 'abs_delta_Pos_charge',
    'abs_delta_Neg_charge', 'abs_delta_Aromatic', 'abs_delta_Alipatic',
    'abs_delta_lcs_score', 'abs_delta_lcs_fraction',
]


def build_feature_matrix(df, emb_col, extra_rankscores=None):
    keep = [c for c in ABS_DELTA_COLS if c in df.columns]
    if extra_rankscores:
        keep += [c for c in extra_rankscores if c in df.columns]
    emb_expanded = df[emb_col].apply(pd.Series)
    emb_expanded.columns = [f'{emb_col}_value_{i}' for i in range(emb_expanded.shape[1])]
    return pd.concat([df[keep].reset_index(drop=True),
                      emb_expanded.reset_index(drop=True)], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE SELECTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def select_embedding_features(X_train_emb, y_train, target_dim=20):
    sel = SelectKBest(f_classif, k=min(target_dim, X_train_emb.shape[1]))
    sel.fit(X_train_emb, y_train)
    return sel


def apply_selection(X_df, sel, emb_pattern):
    emb_cols     = [c for c in X_df.columns if emb_pattern in c and '_value_' in c]
    non_emb_cols = [c for c in X_df.columns if c not in emb_cols]
    sel_cols     = [emb_cols[i] for i in sel.get_support(indices=True)]
    return X_df[non_emb_cols + sel_cols]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CV FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _run_cv_gene(model_factory, X_df, y, genes, method, model_name,
                 n_splits=10, random_state=42, emb_pattern=None, target_dim=20):
    gkf     = GroupKFold(n_splits=n_splits)
    records = []
    for fold, (tr, va) in enumerate(gkf.split(X_df, y, groups=genes)):
        X_tr, X_va = X_df.iloc[tr], X_df.iloc[va]
        y_tr, y_va = y[tr], y[va]
        if emb_pattern:
            emb_cols = [c for c in X_tr.columns if emb_pattern in c and '_value_' in c]
            if emb_cols:
                sel  = select_embedding_features(X_tr[emb_cols].values, y_tr, target_dim)
                X_tr = apply_selection(X_tr, sel, emb_pattern)
                X_va = apply_selection(X_va, sel, emb_pattern)
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr.apply(pd.to_numeric, errors='coerce'))
        X_va_sc = scaler.transform(X_va.apply(pd.to_numeric, errors='coerce'))
        m       = model_factory()
        m.fit(X_tr_sc, y_tr)
        yp      = m.predict_proba(X_va_sc)[:, 1]
        records.append({
            'fold': fold, 'roc_auc': roc_auc_score(y_va, yp),
            'pr_auc': average_precision_score(y_va, yp),
            'accuracy': accuracy_score(y_va, (yp >= 0.5).astype(int)),
            'method': method, 'model_name': model_name, 'cv_type': 'gene',
        })
    return pd.DataFrame(records)


def _run_cv_random(model_factory, X_df, y, method, model_name,
                   n_splits=10, random_state=42, emb_pattern=None, target_dim=20):
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    records = []
    for fold, (tr, va) in enumerate(skf.split(X_df, y)):
        X_tr, X_va = X_df.iloc[tr], X_df.iloc[va]
        y_tr, y_va = y[tr], y[va]
        if emb_pattern:
            emb_cols = [c for c in X_tr.columns if emb_pattern in c and '_value_' in c]
            if emb_cols:
                sel  = select_embedding_features(X_tr[emb_cols].values, y_tr, target_dim)
                X_tr = apply_selection(X_tr, sel, emb_pattern)
                X_va = apply_selection(X_va, sel, emb_pattern)
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr.apply(pd.to_numeric, errors='coerce'))
        X_va_sc = scaler.transform(X_va.apply(pd.to_numeric, errors='coerce'))
        m       = model_factory()
        m.fit(X_tr_sc, y_tr)
        yp      = m.predict_proba(X_va_sc)[:, 1]
        records.append({
            'fold': fold, 'roc_auc': roc_auc_score(y_va, yp),
            'pr_auc': average_precision_score(y_va, yp),
            'accuracy': accuracy_score(y_va, (yp >= 0.5).astype(int)),
            'method': method, 'model_name': model_name, 'cv_type': 'random',
        })
    return pd.DataFrame(records)


def run_both_cv(model_factory, X_df, y, genes, method, model_name,
                n_splits=10, random_state=42, emb_pattern=None, target_dim=20):
    gene_df   = _run_cv_gene(model_factory, X_df, y, genes, method, model_name,
                              n_splits, random_state, emb_pattern, target_dim)
    random_df = _run_cv_random(model_factory, X_df, y, method, model_name,
                                n_splits, random_state, emb_pattern, target_dim)
    return pd.concat([gene_df, random_df], ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HPO FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train_optimize_xgboost(X, y, genes, eva_type,
                            n_splits=10, n_trials=150, random_state=42,
                            save_path='xgboost_results', emb_pattern=None,
                            target_dim=20, split_strategy='random'):
    os.makedirs(save_path, exist_ok=True)
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if hasattr(y, 'values'):
        y = y.values
    y = np.asarray(y)

    if split_strategy == 'gene':
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_val_idx, test_idx = next(gss.split(X, y, groups=genes))
        inner_splitter = 'gene'
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_val_idx, test_idx = next(sss.split(X, y))
        inner_splitter = 'random'

    X_tv = X.iloc[train_val_idx].reset_index(drop=True)
    X_te = X.iloc[test_idx]
    y_tv = y[train_val_idx]
    y_te = y[test_idx]
    g_tv = genes[train_val_idx]

    print(f"[{eva_type} | {split_strategy}] "
          f"Train: {len(X_tv)} ({int(y_tv.sum())} Del / {int((y_tv==0).sum())} Neu) | "
          f"Test:  {len(X_te)} ({int(y_te.sum())} Del / {int((y_te==0).sum())} Neu)")

    X_tv_arr = X_tv.values.astype(np.float64)
    X_te_arr = X_te.values.astype(np.float64)

    if emb_pattern:
        emb_col_idx = [i for i, c in enumerate(X_tv.columns)
                       if emb_pattern in c and '_value_' in c]
        if emb_col_idx:
            sel       = SelectKBest(f_classif, k=min(target_dim, len(emb_col_idx)))
            sel.fit(X_tv_arr[:, emb_col_idx], y_tv)
            sel_local = sel.get_support(indices=True)
            sel_abs   = [emb_col_idx[i] for i in sel_local]
            non_emb   = [i for i in range(X_tv_arr.shape[1]) if i not in emb_col_idx]
            keep_cols = sorted(non_emb + sel_abs)
            X_tv_arr  = X_tv_arr[:, keep_cols]
            X_te_arr  = X_te_arr[:, keep_cols]

    scaler  = StandardScaler()
    X_tv_sc = scaler.fit_transform(X_tv_arr)
    X_te_sc = scaler.transform(X_te_arr)

    n_neg = int((y_tv == 0).sum())
    n_pos = int((y_tv == 1).sum())
    spw   = n_neg / max(n_pos, 1)

    trial_results = []

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 50, 500),
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'gamma':            trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 10.0),
        }
        cv_iter = (GroupKFold(n_splits=n_splits).split(X_tv_sc, y_tv, groups=g_tv)
                   if inner_splitter == 'gene'
                   else StratifiedKFold(n_splits=n_splits, shuffle=True,
                                        random_state=random_state).split(X_tv_sc, y_tv))
        pr_scores, roc_scores, acc_scores = [], [], []
        for tr, va in cv_iter:
            m = XGBClassifier(eval_metric='logloss', tree_method='hist',
                              device='cuda', scale_pos_weight=spw,
                              random_state=random_state, **params)
            m.fit(X_tv_sc[tr], y_tv[tr])
            yp = m.predict_proba(X_tv_sc[va])[:, 1]
            pr_scores.append(average_precision_score(y_tv[va], yp))
            roc_scores.append(roc_auc_score(y_tv[va], yp))
            acc_scores.append(accuracy_score(y_tv[va], (yp >= 0.5).astype(int)))
        mean_pr = float(np.mean(pr_scores))
        trial_results.append({
            'trial_number': trial.number, 'params': params,
            'pr_auc_scores': pr_scores, 'roc_auc_scores': roc_scores,
            'accuracy_scores': acc_scores, 'mean_pr_auc': mean_pr,
            'mean_roc_auc': float(np.mean(roc_scores)),
            'mean_accuracy': float(np.mean(acc_scores)), 'mean_score': mean_pr,
        })
        return mean_pr

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    trials_df   = pd.DataFrame(trial_results)
    best_params = study.best_params
    best_trial  = trials_df.loc[trials_df['mean_pr_auc'].idxmax()]

    print(f"  Best CV PR-AUC: {best_trial['mean_pr_auc']:.4f} "
          f"± {np.std(best_trial['pr_auc_scores']):.4f}")

    suffix = f'_{split_strategy}'
    trials_df.to_json(f"{save_path}/{eva_type}{suffix}_optuna_trials_detailed.json",
                      orient='records', indent=4)
    with open(f"{save_path}/{eva_type}{suffix}_best_hyperparameters.json", 'w') as f:
        json.dump(best_params, f, indent=4)

    best_model = XGBClassifier(eval_metric='logloss', tree_method='hist',
                                device='cuda', scale_pos_weight=spw,
                                random_state=random_state, **best_params)
    best_model.fit(X_tv_sc, y_tv)
    joblib.dump(best_model, f"{save_path}/{eva_type}{suffix}_best_xgboost_model.pkl")

    y_pred = best_model.predict_proba(X_te_sc)[:, 1]
    test_results = {
        'pr_auc':    float(average_precision_score(y_te, y_pred)),
        'roc_auc':   float(roc_auc_score(y_te, y_pred)),
        'accuracy':  float(accuracy_score(y_te, (y_pred >= 0.5).astype(int))),
        'n_test_samples': len(y_te),
        'split_strategy': split_strategy,
        'test_class_distribution': {
            'negative': int((y_te == 0).sum()),
            'positive': int((y_te == 1).sum()),
        },
    }
    print(f"  Test PR-AUC: {test_results['pr_auc']:.4f} | "
          f"ROC-AUC: {test_results['roc_auc']:.4f}")

    with open(f"{save_path}/{eva_type}{suffix}_test_results.json", 'w') as f:
        json.dump(test_results, f, indent=4)

    cv_summary = {
        'best_trial_number': int(best_trial['trial_number']),
        'split_strategy': split_strategy,
        'cross_validation': {
            'pr_auc_mean':    float(best_trial['mean_pr_auc']),
            'pr_auc_std':     float(np.std(best_trial['pr_auc_scores'])),
            'pr_auc_folds':   [float(x) for x in best_trial['pr_auc_scores']],
            'roc_auc_mean':   float(best_trial['mean_roc_auc']),
            'roc_auc_std':    float(np.std(best_trial['roc_auc_scores'])),
            'roc_auc_folds':  [float(x) for x in best_trial['roc_auc_scores']],
            'accuracy_mean':  float(best_trial['mean_accuracy']),
            'accuracy_std':   float(np.std(best_trial['accuracy_scores'])),
            'accuracy_folds': [float(x) for x in best_trial['accuracy_scores']],
        },
        'test_performance': test_results,
        'hyperparameters':  best_params,
    }
    np.save(f"{save_path}/{eva_type}{suffix}_test_pred_proba.npy",    y_pred)
    np.save(f"{save_path}/{eva_type}{suffix}_test_indices.npy",       test_idx)
    np.save(f"{save_path}/{eva_type}{suffix}_trainval_indices.npy",   train_val_idx)
    with open(f"{save_path}/{eva_type}{suffix}_comprehensive_results.json", 'w') as f:
        json.dump(cv_summary, f, indent=4)

    return (best_params, trials_df, test_results,
            train_val_idx.tolist(), test_idx.tolist(), y_pred, cv_summary)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  RESULT LOADING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_hpo_results(eva_type, split_strategy, save_path='xgboost_results'):
    suffix = f'_{split_strategy}'
    base   = Path(save_path) / f'{eva_type}{suffix}'
    with open(f'{base}_best_hyperparameters.json')   as f: best_params  = json.load(f)
    with open(f'{base}_test_results.json')           as f: test_results = json.load(f)
    with open(f'{base}_comprehensive_results.json')  as f: cv_summary   = json.load(f)
    trials_df = pd.read_json(f'{base}_optuna_trials_detailed.json', orient='records')
    # Load saved arrays if available
    ti_path = f'{base}_test_indices.npy'
    pp_path = f'{base}_test_pred_proba.npy'
    tv_path = f'{base}_trainval_indices.npy'
    test_idx      = np.load(ti_path).tolist() if Path(ti_path).exists() else []
    y_pred_proba  = np.load(pp_path)          if Path(pp_path).exists() else np.array([])
    train_val_idx = np.load(tv_path).tolist() if Path(tv_path).exists() else []
    print(f"  Loaded [{eva_type}|{split_strategy}] "
          f"PR={test_results['pr_auc']:.4f} ROC={test_results['roc_auc']:.4f}")
    return (best_params, trials_df, test_results,
            train_val_idx, test_idx, y_pred_proba, cv_summary)


def extract_fold_scores(trials_json_path, method_label, cv_type):
    if not os.path.exists(trials_json_path):
        print(f"  SKIPPING (not found): {trials_json_path}")
        return pd.DataFrame()
    df   = pd.read_json(trials_json_path)
    best = df.loc[df['mean_pr_auc'].idxmax()]
    def _parse(x):
        return ast.literal_eval(x) if isinstance(x, str) else list(x)
    pr  = _parse(best['pr_auc_scores'])
    roc = _parse(best['roc_auc_scores'])
    acc = _parse(best['accuracy_scores'])
    return pd.DataFrame({
        'fold': range(len(pr)), 'roc_auc': roc, 'pr_auc': pr, 'accuracy': acc,
        'method': method_label, 'model_name': 'XGBoost/Optuna', 'cv_type': cv_type,
    })


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    delta_emb_df = pd.read_json(EMB_JSON, lines=True)

    le        = LabelEncoder()
    y_encoded = le.fit_transform(delta_emb_df['Class'])
    genes     = delta_emb_df['GENES'].values

    # ── Build feature matrices ─────────────────────────────────────────────────
    print("Building feature matrices...")
    average_X_raw  = build_feature_matrix(delta_emb_df, 'delta_emb_average')
    L1_X_raw       = build_feature_matrix(delta_emb_df, 'delta_emb_L1')
    L2_X_raw       = build_feature_matrix(delta_emb_df, 'delta_emb_L2')
    hadamard_X_raw = build_feature_matrix(delta_emb_df, 'delta_emb_hadamard')

    average_eve_X_raw = build_feature_matrix(delta_emb_df, 'delta_emb_average', ['EVE_rankscore'])
    average_esm_X_raw = build_feature_matrix(delta_emb_df, 'delta_emb_average', ['ESM1b_rankscore'])
    average_am_X_raw  = build_feature_matrix(delta_emb_df, 'delta_emb_average', ['AlphaMissense_rankscore'])
    average_all_X_raw = build_feature_matrix(delta_emb_df, 'delta_emb_average',
                                              ['AlphaMissense_rankscore', 'ESM1b_rankscore', 'EVE_rankscore'])

    combined_cols     = ['AlphaMissense_rankscore', 'ESM1b_rankscore', 'EVE_rankscore']
    avail_combined    = [c for c in combined_cols if c in delta_emb_df.columns]
    combined_standalone_X = (delta_emb_df[avail_combined]
                              .apply(pd.to_numeric, errors='coerce')
                              .dropna().reset_index(drop=True))
    valid_idx         = delta_emb_df[avail_combined].dropna().index
    y_combined_sa     = y_encoded[valid_idx]
    genes_combined_sa = genes[valid_idx]

    # ── Embedding CV comparison ────────────────────────────────────────────────
    cv_csv = OUTPUT_DIR / 'cv_embedding_comparison_both.csv'
    if args.skip_cv and cv_csv.exists():
        print(f"Loading CV results from {cv_csv}")
        emd_results_df = pd.read_csv(cv_csv)
    else:
        print("Running embedding CV comparison (this takes a while)...")
        all_cv = []
        for X_raw, method_label, emb_pat in [
            (L1_X_raw,       'L1 + PS + gIDRc',       'delta_emb_L1'),
            (L2_X_raw,       'L2 + PS + gIDRc',       'delta_emb_L2'),
            (average_X_raw,  'Average + PS + gIDRc',  'delta_emb_average'),
            (hadamard_X_raw, 'Hadamard + PS + gIDRc', 'delta_emb_hadamard'),
        ]:
            for factory, model_name in [
                (lambda: XGBClassifier(eval_metric='logloss', tree_method='hist',
                                       device='cuda', random_state=42), 'XGboost'),
                (lambda: RandomForestClassifier(class_weight='balanced', random_state=42), 'Random Forest'),
                (lambda: GaussianNB(), 'Naive Bayes'),
                (lambda: MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42), 'MLP'),
            ]:
                print(f"  CV: {method_label} | {model_name}")
                df_res = run_both_cv(factory, X_raw, y_encoded, genes,
                                     method_label, model_name,
                                     n_splits=10, random_state=42,
                                     emb_pattern=emb_pat, target_dim=20)
                all_cv.append(df_res)
        emd_results_df = pd.concat(all_cv, ignore_index=True)
        emd_results_df.to_csv(cv_csv, index=False)
        print(f"  Saved → {cv_csv}")

    # ── HPO ───────────────────────────────────────────────────────────────────
    hpo_configs = [
        (average_X_raw,         y_encoded,      genes,            'default',                        'delta_emb_average'),
        (average_eve_X_raw,     y_encoded,      genes,            'EVE',                            'delta_emb_average'),
        (average_esm_X_raw,     y_encoded,      genes,            'ESM1B',                          'delta_emb_average'),
        (average_am_X_raw,      y_encoded,      genes,            'AlphaMissense',                  'delta_emb_average'),
        (combined_standalone_X, y_combined_sa,  genes_combined_sa,'Combined_Standalone_AM_ESM_EVE', None),
        (average_all_X_raw,     y_encoded,      genes,            'IFP_MIDR_All_Three',             'delta_emb_average'),
    ]

    splits_to_run = (['gene', 'random'] if args.split == 'both'
                     else [args.split])

    gene_hpo_results   = {}
    random_hpo_results = {}

    for split_strategy in splits_to_run:
        target = gene_hpo_results if split_strategy == 'gene' else random_hpo_results
        for X, y, g, name, emb_pat in hpo_configs:
            result_path = HPO_DIR / f'{name}_{split_strategy}_test_results.json'
            if args.skip_hpo and result_path.exists():
                print(f"  Loading saved HPO: {name} [{split_strategy}]")
                target[name] = load_hpo_results(name, split_strategy)
            else:
                print(f"  Running HPO: {name} [{split_strategy}]")
                target[name] = train_optimize_xgboost(
                    X, y, g, name,
                    emb_pattern=emb_pat, target_dim=20,
                    split_strategy=split_strategy,
                )

    # ── Print comparison table ─────────────────────────────────────────────────
    model_labels = [
        ('default',                        'Baseline'),
        ('EVE',                            '+ EVE'),
        ('ESM1B',                          '+ ESM1b'),
        ('AlphaMissense',                  '+ AlphaMissense'),
        ('Combined_Standalone_AM_ESM_EVE', 'AM+ESM+EVE standalone'),
        ('IFP_MIDR_All_Three',             '+ All Three'),
    ]
    print("\n" + "="*75)
    print(f"{'Model':<35} {'Gene PR':>9} {'Rand PR':>9} {'Gene ROC':>10} {'Rand ROC':>10}")
    print("="*75)
    for name, label in model_labels:
        gpr  = gene_hpo_results.get(name,   {2: {}})[2].get('pr_auc',  float('nan'))
        rpr  = random_hpo_results.get(name, {2: {}})[2].get('pr_auc',  float('nan'))
        groc = gene_hpo_results.get(name,   {2: {}})[2].get('roc_auc', float('nan'))
        rroc = random_hpo_results.get(name, {2: {}})[2].get('roc_auc', float('nan'))
        print(f"  {label:<33} {gpr:>9.4f} {rpr:>9.4f} {groc:>10.4f} {rroc:>10.4f}")
    print("="*75)

    # ── Assemble result_all_df ─────────────────────────────────────────────────
    print("\nAssembling result_all_df...")
    METHOD_LABELS_HPO = [
        ('default',                        'Average + PS + gIDRc'),
        ('EVE',                            'XGBoost + EVE + Average + PS + gIDRc'),
        ('ESM1B',                          'XGBoost + ESM1b + Average + PS + gIDRc'),
        ('AlphaMissense',                  'XGBoost + AlphaMissense + Average + PS + gIDRc'),
        ('Combined_Standalone_AM_ESM_EVE', 'AlphaMissense + ESM1b + EVE (standalone)'),
        ('IFP_MIDR_All_Three',             'IFP-MIDR + AlphaMissense + ESM1b + EVE'),
    ]

    gene_dfs   = [extract_fold_scores(
                      str(HPO_DIR / f'{eva}_gene_optuna_trials_detailed.json'), label, 'gene')
                  for eva, label in METHOD_LABELS_HPO]
    random_dfs = [extract_fold_scores(
                      str(HPO_DIR / f'{eva}_random_optuna_trials_detailed.json'), label, 'random')
                  for eva, label in METHOD_LABELS_HPO]

    # Standalone predictors via gene-stratified CV
    standalone_gene = []
    for pred_col, method_label in [
        ('AlphaMissense_rankscore', 'AlphaMissense'),
        ('EVE_rankscore',           'EVE'),
        ('ESM1b_rankscore',         'ESM1b'),
    ]:
        if pred_col not in delta_emb_df.columns:
            continue
        X_sa = (delta_emb_df[[pred_col]]
                .apply(pd.to_numeric, errors='coerce').fillna(0))
        df_sa = _run_cv_gene(
            lambda: XGBClassifier(eval_metric='logloss', tree_method='hist',
                                  device='cuda', random_state=42),
            X_sa, y_encoded, genes,
            method=method_label, model_name='XGBoost/Optuna',
            n_splits=5, random_state=42, emb_pattern=None,
        )
        df_sa = df_sa.rename(columns={'auc_pr': 'pr_auc'})
        standalone_gene.append(df_sa)

    # Standalone from existing CSV if available
    extra_dfs = []
    if Path(NOEMB_CSV).exists():
        noemb = pd.read_csv(NOEMB_CSV)
        noemb = (noemb[noemb['model'].isin(['AlphaMissense', 'EVE', 'ESM1B'])]
                 [['fold', 'auc', 'pr_auc', 'accuracy', 'method', 'model']].copy())
        noemb.columns = ['fold', 'roc_auc', 'pr_auc', 'accuracy', 'method', 'model_name']
        noemb['method']  = noemb['method'].replace({'ESM1B': 'ESM1b'})
        noemb['cv_type'] = 'random'
        extra_dfs.append(noemb)

    all_dfs = (extra_dfs + standalone_gene
               + [d for d in gene_dfs   if not d.empty]
               + [d for d in random_dfs if not d.empty])

    result_all_df = pd.concat(all_dfs, ignore_index=True)
    result_all_df['pr_auc']  = pd.to_numeric(result_all_df['pr_auc'],  errors='coerce')
    result_all_df['roc_auc'] = pd.to_numeric(result_all_df['roc_auc'], errors='coerce')
    result_all_df = result_all_df.dropna(subset=['pr_auc', 'roc_auc'])
    result_all_df.to_csv(OUTPUT_DIR / 'all_model_results.csv', index=False)
    print(f"  Saved → {OUTPUT_DIR / 'all_model_results.csv'}")
    print("\nMethods per cv_type:")
    print(result_all_df.groupby(['cv_type', 'method'])['pr_auc'].agg(['count', 'mean']).round(4))

    # ── Save test set predictions ──────────────────────────────────────────────
    print("\nSaving hold-out test set predictions...")

    all_pred_dfs = []

    for split_strategy, hpo_dict in [('gene',   gene_hpo_results),
                                      ('random', random_hpo_results)]:
        if not hpo_dict:
            continue

        for name, label in [
            ('default',                        'IFP-MIDR Baseline'),
            ('EVE',                            'IFP-MIDR + EVE'),
            ('ESM1B',                          'IFP-MIDR + ESM1b'),
            ('AlphaMissense',                  'IFP-MIDR + AlphaMissense'),
            ('Combined_Standalone_AM_ESM_EVE', 'AM + ESM1b + EVE (standalone)'),
            ('IFP_MIDR_All_Three',             'IFP-MIDR + All Three'),
        ]:
            if name not in hpo_dict:
                continue

            (_, _, test_results, train_val_idx,
             test_idx, y_pred_proba, _) = hpo_dict[name]

            if len(test_idx) == 0 or len(y_pred_proba) == 0:
                print(f"  WARNING: no test indices/predictions for {label} [{split_strategy}] — skipping")
                continue

            # Pull metadata for test variants
            src_df = delta_emb_df

            # For combined standalone, source df is subset — use full df with valid_idx mapping
            if name == 'Combined_Standalone_AM_ESM_EVE':
                src_df = delta_emb_df.iloc[valid_idx].reset_index(drop=True)

            test_meta = src_df.iloc[test_idx].copy().reset_index(drop=True)

            test_meta['y_pred_proba'] = y_pred_proba
            test_meta['y_pred_label'] = (y_pred_proba >= 0.5).astype(int)
            test_meta['y_true']       = test_meta['Class'].map({'Deleterious': 1, 'Neutral': 0})
            test_meta['correct']      = (test_meta['y_pred_label'] == test_meta['y_true']).astype(int)
            test_meta['model']        = label
            test_meta['split']        = split_strategy

            keep_cols = [c for c in [
                'GENES', 'mutation', 'UNIPROT_ID', 'CLNHGVS', 'CLNSIG',
                'CLNREVSTAT', 'Class', 'y_true', 'y_pred_proba',
                'y_pred_label', 'correct', 'model', 'split',
            ] if c in test_meta.columns]

            out_df = test_meta[keep_cols].copy()

            # Per-model file
            fname = PRED_DIR / f'{label.replace(" ", "_").replace("+", "plus")}_{split_strategy}_predictions.tsv'
            out_df.to_csv(fname, sep='\t', index=False)
            print(f"  Saved {len(out_df)} variants → {fname.name} "
                  f"(PR={test_results['pr_auc']:.4f} ROC={test_results['roc_auc']:.4f})")

            all_pred_dfs.append(out_df)

    # ── Master combined predictions file ──────────────────────────────────────
    if all_pred_dfs:
        master = pd.concat(all_pred_dfs, ignore_index=True)
        master_path = PRED_DIR / 'all_models_test_predictions.tsv'
        master.to_csv(master_path, sep='\t', index=False)
        print(f"\nMaster predictions saved → {master_path} ({len(master)} rows)")

        # Summary table
        summary = (master
                   .groupby(['model', 'split'])
                   .apply(lambda g: pd.Series({
                       'n_variants': len(g),
                       'n_del':      int(g['y_true'].sum()),
                       'n_neu':      int((g['y_true'] == 0).sum()),
                       'tp':         int(((g['y_true'] == 1) & (g['y_pred_label'] == 1)).sum()),
                       'fp':         int(((g['y_true'] == 0) & (g['y_pred_label'] == 1)).sum()),
                       'fn':         int(((g['y_true'] == 1) & (g['y_pred_label'] == 0)).sum()),
                       'tn':         int(((g['y_true'] == 0) & (g['y_pred_label'] == 0)).sum()),
                       'accuracy':   round(g['correct'].mean(), 4),
                   }))
                   .reset_index())
        summary_path = PRED_DIR / 'prediction_summary.csv'
        summary.to_csv(summary_path, index=False)
        print(f"Summary saved → {summary_path}")
        print("\nPREDICTION SUMMARY")
        print("=" * 90)
        print(summary.to_string(index=False))

    print("\nPipeline complete.")


if __name__ == '__main__':
    main()
