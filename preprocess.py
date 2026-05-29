"""
preprocess.py  —  Elliptic Bitcoin Dataset Preprocessing
==========================================================
Run:  python preprocess.py

Output:  outputs/graphs.pkl   →   dict with keys 'train', 'val', 'test',
                                  'class_weights', 'scaler', and 'metadata'.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = r"C:\Users\SEDONA RAJESH\OneDrive\Desktop\gnn\elliptic_bitcoin_dataset"
OUT_DIR  = r"C:\Users\SEDONA RAJESH\OneDrive\Desktop\gnn\outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ==============================================================================
# STEP 1: Loading CSV files
# ==============================================================================
print("=" * 60)
print("STEP 1: Loading CSV files")
print("=" * 60)

df_feat = pd.read_csv(
    os.path.join(DATA_DIR, "elliptic_txs_features.csv"),
    header=None
)
df_feat.columns = ["txId", "time_step"] + [f"f{i}" for i in range(1, 166)]

df_class = pd.read_csv(os.path.join(DATA_DIR, "elliptic_txs_classes.csv"))
df_edges = pd.read_csv(os.path.join(DATA_DIR, "elliptic_txs_edgelist.csv"))

print(f"  Features   : {df_feat.shape[0]:>7,} rows  x  {df_feat.shape[1]} cols")
print(f"  Classes    : {df_class.shape[0]:>7,} rows")
print(f"  Edges      : {df_edges.shape[0]:>7,} rows")


# ==============================================================================
# STEP 2 — Merge features and class labels on txId
# ==============================================================================
print("\nSTEP 2: Merging features + labels")
df = df_feat.merge(df_class, on="txId", how="left")
print(f"  Merged shape : {df.shape}")

# ==============================================================================
# STEP 3 — Encode string/integer labels to integers
# ==============================================================================
print("\nSTEP 3: Encoding labels")

# Supports both string and integer formats depending on pandas parser version
label_map = {
    "1": 1,
    "2": 0,
    1: 1,
    2: 0,
    "unknown": -1
}
df["label"] = df["class"].map(label_map)

# Catch type inference issues early before they reach tensor operations
if df["label"].isna().any():
    raise ValueError(
        f"Found {df['label'].isna().sum()} unmapped labels. "
        "Check class column format."
    )

n_illicit = (df["label"] == 1).sum()
n_licit   = (df["label"] == 0).sum()
n_unknown = (df["label"] == -1).sum()

print(f"  Illicit  (1) : {n_illicit:>7,}  ({100*n_illicit/len(df):.1f}%)")
print(f"  Licit    (0) : {n_licit:>7,}  ({100*n_licit/len(df):.1f}%)")
print(f"  Unknown (-1) : {n_unknown:>7,}  ({100*n_unknown/len(df):.1f}%)")

# Balanced Class Weights Setup
labels = df.loc[df["label"] != -1, "label"]
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=labels
)
class_weights = torch.tensor(weights, dtype=torch.float)

print("\nClass Weights")
print(f"  Licit Weight   : {class_weights[0]:.4f}")
print(f"  Illicit Weight : {class_weights[1]:.4f}")

# ==============================================================================
# STEP 4 — Create training mask
# ==============================================================================
print("\nSTEP 4: Creating node masks")
df["mask"] = df["label"] != -1
print(f"  Labeled nodes   : {df['mask'].sum():>7,}")
print(f"  Unlabeled nodes : {(~df['mask']).sum():>7,}")


# ==============================================================================
# STEP 5 — Feature normalization (t1-t34 used for fitting to avoid leakage)
# ==============================================================================
print("\nSTEP 5: Normalizing features (no data leakage)")
FEATURE_COLS = [f"f{i}" for i in range(1, 166)]
is_train_step = df["time_step"] <= 34

scaler = StandardScaler()
df.loc[is_train_step, FEATURE_COLS] = scaler.fit_transform(
    df.loc[is_train_step, FEATURE_COLS]
)
df.loc[~is_train_step, FEATURE_COLS] = scaler.transform(
    df.loc[~is_train_step, FEATURE_COLS]
)
print("  Done — scaler fit on t1-t34, transformed t35-t49 separately")


# ==============================================================================
# STEP 6 — Build one PyG Data object per time step (With Temporal Feature)
# ==============================================================================
print("\nSTEP 6: Building per-timestep PyG graphs")
print(f"  {'t':>3}  {'nodes':>7}  {'edges':>7}  {'illicit':>8}  {'licit':>7}  {'unknown':>8}")
print("  " + "-" * 52)

graphs = []
time_steps = sorted(df["time_step"].unique())


# 1. Compute relative time globally ONCE before the loop starts
df["relative_time"] = (df["time_step"] - 1) / 48.0

for t in time_steps:
    sub = df[df["time_step"] == t].reset_index(drop=True)

    # Base node features
    x = torch.tensor(sub[FEATURE_COLS].values, dtype=torch.float)

    # Extract already computed relative time feature
    time_feature = torch.tensor(
        sub["relative_time"].values.reshape(-1, 1),
        dtype=torch.float
    )
    x = torch.cat([x, time_feature], dim=1)

    # Labels and masks
    y = torch.tensor(sub["label"].values, dtype=torch.long)
    mask = torch.tensor(sub["mask"].values, dtype=torch.bool)

    # Local index mapping
    node_ids = sub["txId"].tolist()
    id2idx = {nid: idx for idx, nid in enumerate(node_ids)}
    node_set = set(node_ids)

    sub_edges = df_edges[
        df_edges["txId1"].isin(node_set) &
        df_edges["txId2"].isin(node_set)
    ].copy()

    if len(sub_edges) > 0:
        src = sub_edges["txId1"].map(id2idx).values
        dst = sub_edges["txId2"].map(id2idx).values
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(
        x          = x,
        edge_index = edge_index,
        y          = y,
        mask       = mask,
        time_step  = int(t),
        num_nodes  = x.shape[0]
    )
    graphs.append(data)

    n_ill = int((y[mask] == 1).sum())
    n_lic = int((y[mask] == 0).sum())
    n_unk = int((~mask).sum())
    print(f"  {int(t):>3}  {x.shape[0]:>7,}  {edge_index.shape[1]:>7,}"
          f"  {n_ill:>8,}  {n_lic:>7,}  {n_unk:>8,}")


# ==============================================================================
# STEP 7 — Temporal train / val / test split
# ==============================================================================
print("\nSTEP 7: Temporal train/val/test split")

train_graphs = [g for g in graphs if g.time_step <= 34]
val_graphs   = [g for g in graphs if 35 <= g.time_step <= 41]
test_graphs  = [g for g in graphs if g.time_step >= 42]

print(f"  Train snapshots : {len(train_graphs)} (t1 - t34)")
print(f"  Val snapshots   : {len(val_graphs)} (t35 - t41)")
print(f"  Test snapshots  : {len(test_graphs)} (t42 - t49)")

train_nodes = sum(g.num_nodes for g in train_graphs)
val_nodes   = sum(g.num_nodes for g in val_graphs)
test_nodes  = sum(g.num_nodes for g in test_graphs)

print(f"  Train nodes     : {train_nodes:,}")
print(f"  Val nodes       : {val_nodes:,}")
print(f"  Test nodes      : {test_nodes:,}")


# ==============================================================================
# STEP 8 — Sanity checks (Verifying feature space = 166)
# ==============================================================================
print("\nSTEP 8: Sanity checks")
errors = 0

for g in graphs:
    if g.x.shape[1] != 166:
        print(f"  [ERROR] t={g.time_step}: expected 166 features, got {g.x.shape[1]}")
        errors += 1

    if g.edge_index.numel() > 0:
        if g.edge_index.max() >= g.num_nodes:
            print(f"  [ERROR] t={g.time_step}: edge index out of bounds")
            errors += 1

    if torch.isnan(g.x).any():
        print(f"  [ERROR] t={g.time_step}: NaN values in features")
        errors += 1

    unique_labels = g.y.unique().tolist()
    for lbl in unique_labels:
        if lbl not in [-1, 0, 1]:
            print(f"  [ERROR] t={g.time_step}: unexpected label value {lbl}")
            errors += 1

if errors == 0:
    print("  All checks passed — no errors found")
else:
    print(f"  {errors} error(s) found — check output above")


# ==============================================================================
# STEP 9 — Save to disk with complete Metadata Dict
# ==============================================================================
print("\nSTEP 9: Saving preprocessed graphs")

metadata = {
    "num_features": 166,
    "num_classes": 2,
    "num_snapshots": len(graphs),
    "train_range": (1, 34),
    "val_range": (35, 41),
    "test_range": (42, 49)
}

save_path = os.path.join(OUT_DIR, "graphs.pkl")
with open(save_path, "wb") as f:
    pickle.dump(
        {
            "train": train_graphs,
            "val": val_graphs,
            "test": test_graphs,
            "class_weights": class_weights,
            "scaler": scaler,
            "metadata": metadata
        },
        f
    )

print(f"  Saved → {save_path}")


# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print(f"  Total snapshots : {len(graphs)}")
print(f"  Feature dims    : 166")
print(f"  Train snapshots : {len(train_graphs)}")
print(f"  Val snapshots   : {len(val_graphs)}")
print(f"  Test snapshots  : {len(test_graphs)}")
print(f"  Output file     : {save_path}")