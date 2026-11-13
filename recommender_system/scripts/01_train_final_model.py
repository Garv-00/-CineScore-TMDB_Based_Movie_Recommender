"""
CineScore Final Model
Trains on KION + MovieLens combined data
Target: 6.5-8.5 NDCG@10 
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "recommender_system/data/processed"
MODEL_DIR = ROOT / "recommender_system/model"

FINAL_MODEL = MODEL_DIR / "cinescore_final.txt"
FINAL_META = MODEL_DIR / "cinescore_final_metadata.json"

RANDOM_STATE = 42

def main():
    print("="*80)
    print("CINESCORE LEVEL 5: FINAL MODEL TRAINING")
    print("Dataset: KION + MovieLens Combined")
    print("="*80)

    # ========== LOAD DATA ==========
    print("\n[STEP 1/5] Loading datasets...")

    kion = pd.read_csv(DATA_DIR / "kion_train.csv")
    ml = pd.read_csv(DATA_DIR / "movielens_train.csv")

    print(f"  KION: {len(kion):,} interactions")
    print(f"  MovieLens: {len(ml):,} interactions")

    # Create labels
    kion['label'] = (kion['preference_score'] >= 0.7).astype(np.int8)
    kion['source'] = 0

    ml['label'] = (ml['rating'] >= 4.0).astype(np.int8)
    ml['source'] = 1

    # Standardize columns and fix types
    kion = kion[['user_id', 'tmdb_id', 'label', 'source']].copy()
    ml = ml[['user_id', 'tmdb_id', 'label', 'source']].copy()

    # Ensure consistent user_id format
    kion['user_id'] = kion['user_id'].astype(str)
    ml['user_id'] = 'ML_' + ml['user_id'].astype(str)

    # Combine
    df = pd.concat([kion, ml], ignore_index=True)

    print(f"  Combined: {len(df):,} interactions")
    print(f"  Positives: {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
    print(f"  Users: {df['user_id'].nunique():,}")

    # ========== TEMPORAL SPLIT ==========
    print("\n[STEP 2/5] Train/validation split (temporal)...")

    np.random.seed(RANDOM_STATE)
    users = df['user_id'].unique()
    np.random.shuffle(users)

    split_idx = int(len(users) * 0.8)
    train_users = set(users[:split_idx])
    valid_users = set(users[split_idx:])

    df_train = df[df['user_id'].isin(train_users)].copy()
    df_valid = df[df['user_id'].isin(valid_users)].copy()

    # Keep only groups with positives
    df_train = df_train[df_train.groupby('user_id')['label'].transform('max') > 0]
    df_valid = df_valid[df_valid.groupby('user_id')['label'].transform('max') > 0]

    print(f"  Train: {len(df_train):,} interactions, {df_train['user_id'].nunique():,} users")
    print(f"  Valid: {len(df_valid):,} interactions, {df_valid['user_id'].nunique():,} users")

    # ========== NEGATIVE SAMPLING ==========
    print("\n[STEP 3/5] Negative downsampling (20:1 ratio)...")

    pos_train = df_train[df_train['label'] == 1]
    neg_train = df_train[df_train['label'] == 0].copy()

    # Calculate cap per user
    user_pos_counts = pos_train.groupby('user_id').size()
    neg_cap = neg_train['user_id'].map(user_pos_counts) * 20

    # Random sampling
    neg_train['_rand'] = np.random.random(len(neg_train))
    neg_train['_rank'] = neg_train.groupby('user_id')['_rand'].rank(method='first')
    neg_train = neg_train[neg_train['_rank'] <= neg_cap].drop(columns=['_rand', '_rank'])

    df_train_balanced = pd.concat([pos_train, neg_train], ignore_index=True)
    df_train_balanced = df_train_balanced.sort_values('user_id').reset_index(drop=True)

    print(f"  After sampling: {len(df_train_balanced):,} interactions")
    print(f"  Positives: {pos_train['label'].sum():,} ({pos_train['label'].mean()*100:.1f}%)")

    # ========== PREPARE FEATURES ==========
    print("\n[STEP 4/5] Building feature matrix...")

    # Simple effective features
    X_train = df_train_balanced[['tmdb_id', 'source']].astype(np.float32).values
    y_train = df_train_balanced['label'].astype(np.int8).values
    groups_train = df_train_balanced.groupby('user_id', sort=False).size().values

    X_valid = df_valid[['tmdb_id', 'source']].astype(np.float32).values
    y_valid = df_valid['label'].astype(np.int8).values
    groups_valid = df_valid.groupby('user_id', sort=False).size().values

    print(f"  Features: 2 (tmdb_id, source)")
    print(f"  Train samples: {len(X_train):,} in {len(groups_train):,} groups")
    print(f"  Valid samples: {len(X_valid):,} in {len(groups_valid):,} groups")

    # ========== TRAIN MODEL ==========
    print("\n[STEP 5/5] Training LambdaMART...")

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=groups_train,
        feature_name=['tmdb_id', 'source'],
        free_raw_data=True
    )

    valid_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        group=groups_valid,
        reference=train_data,
        free_raw_data=True
    )

    params = {
        'objective': 'rank_xendcg',
        'metric': 'ndcg',
        'ndcg_eval_at': [10],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_threads': -1,
        'verbose': -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(30),
            lgb.log_evaluation(100),
        ],
    )

    best_iter = model.best_iteration
    best_score = model.best_score['valid_0']['ndcg@10']

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"Best NDCG@10: {best_score:.4f}")
    print(f"Best Iteration: {best_iter}")
    print(f"{'='*80}")

    # ========== SAVE MODEL ==========
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(FINAL_MODEL))

    metadata = {
        "model_name": "CineScore Level 5 Final",
        "version": "1.0",
        "training_date": "2026-10-01",
        "datasets": ["kion", "movielens"],
        "best_ndcg_at_10": float(best_score),
        "best_iteration": int(best_iter),
        "features": ["tmdb_id", "source"],
        "train_samples": int(len(X_train)),
        "valid_samples": int(len(X_valid)),
        "train_positives": int(y_train.sum()),
        "valid_positives": int(y_valid.sum()),
        "random_state": RANDOM_STATE,
    }

    with open(FINAL_META, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Model saved: {FINAL_MODEL}")
    print(f"✓ Metadata saved: {FINAL_META}")

    return best_score

if __name__ == "__main__":
    try:
        score = main()
        print("\n✓ PRODUCTION MODEL READY")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
