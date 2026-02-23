# Generated from: Netflix_Prize_CaseStudy_2006_vs_2025 (1).ipynb
# Converted at: 2026-02-23T16:32:49.803Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# 
# # Netflix Prize Case Study – 2006 vs 2025 
# 
# This notebook is structured so you can **show live implementation** 
# - **2006-style baseline:** SVD++ (classical matrix factorization) using the `surprise` library.
# - **2025-style model (deep dive):** **HG-TNR (Hybrid Graph + Temporal Transformer Recommender)** – a compact, production-friendly variant implemented in PyTorch that:
#   - Learns **item-item relations** (graph regularization from co-rating co-occurrence).
#   - Models **temporal sequences** per user with a **Transformer** (attention over time).
#   - Predicts **ratings** (for RMSE) and can compute **ranking metrics** (NDCG@K).
# 
# > **Tip:** If you do not have the Netflix Prize files locally, the notebook will automatically download **MovieLens 100K** as a lightweight, classroom-friendly proxy dataset.
# 


import sys, numpy, surprise
print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("Surprise:", surprise.__version__)


# 
# ## 0. Setup
# Run the cell below **once** to install required packages in your environment (skip if already installed).
# 


# If you haven't installed packages in your conda env, do this once:
# !pip install numpy==1.26.4 scikit-surprise==1.1.4 pandas matplotlib torch torchvision torchaudio networkx

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from surprise import Dataset, SVDpp
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse


# 
# ## 1. Imports & Configuration
# 


# Use the lightweight MovieLens-100K dataset
data = Dataset.load_builtin('ml-100k')

# Split into train and test
trainset, testset = train_test_split(data, test_size=0.2)


# 
# ## 2. Data: Netflix Prize (if available) or MovieLens 100K (auto-download)
# - If you already have Netflix Prize data locally, set `USE_NETFLIX = True` and point to the path.
# - Otherwise, we will **auto-download MovieLens 100K**.
# 



USE_NETFLIX = False  # Set to True if you have the Netflix Prize data locally
NETFLIX_DIR = Path('./netflix-prize-data')  # change this to your local path if you have it

DATA_DIR = Path('./data')
DATA_DIR.mkdir(exist_ok=True, parents=True)

def load_movielens_100k():
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    zip_path = DATA_DIR / 'ml-100k.zip'
    extract_dir = DATA_DIR / 'ml-100k'
    if not extract_dir.exists():
        import requests
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(DATA_DIR)
    ratings_path = extract_dir / 'u.data'
    movies_path = extract_dir / 'u.item'
    # u.data columns: user_id, item_id, rating, timestamp (tab separated)
    ratings = pd.read_csv(ratings_path, sep='\t', names=['userId','movieId','rating','timestamp'], engine='python')
    # u.item columns are pipe-separated; include title & year where available
    movies = pd.read_csv(movies_path, sep='\|', header=None, encoding='latin-1')
    movies = movies[[0,1,2]]  # movieId, title, release_date
    movies.columns = ['movieId','title','release_date']
    return ratings, movies

def load_netflix_local(netflix_dir: Path):
    # Netflix Prize layout has multiple files like 'combined_data_1.txt'...
    # Each file contains lines of "movieId:" then user ratings rows "userId,rating,timestamp"
    # We'll parse a subset for demo (first ~1M ratings) to keep it light.
    import csv
    files = [netflix_dir / f'combined_data_{i}.txt' for i in range(1,5)]
    data = []
    current_movie = None
    for fp in files:
        if not fp.exists():
            continue
        with open(fp, 'r') as f:
            for line in f:
                line = line.strip()
                if line.endswith(':'):
                    current_movie = int(line[:-1])
                else:
                    if not current_movie:
                        continue
                    parts = line.split(',')
                    if len(parts) == 3:
                        uid, r, ts = parts
                        data.append((int(uid), int(current_movie), int(r), int(ts)))
                if len(data) >= 1_000_000:  # cap for demo speed
                    break
        if len(data) >= 1_000_000:
            break
    ratings = pd.DataFrame(data, columns=['userId','movieId','rating','timestamp'])
    movies = pd.DataFrame(columns=['movieId','title','release_date'])  # titles not included in prize files
    return ratings, movies

if USE_NETFLIX and NETFLIX_DIR.exists():
    print('Loading Netflix Prize subset from', NETFLIX_DIR.resolve())
    ratings_df, movies_df = load_netflix_local(NETFLIX_DIR)
else:
    print('Downloading MovieLens 100K (first time only) ...')
    ratings_df, movies_df = load_movielens_100k()

print('Ratings shape:', ratings_df.shape)
print(ratings_df.head())


# 
# ## 3. Preprocessing
# - Map `userId` and `movieId` to contiguous indices.
# - Sort interactions per user by `timestamp` (needed for temporal modeling).
# - Train/Validation/Test split by **time per user** (last interactions to test).
# 



# Map IDs to contiguous indices
uid_map = {u:i for i,u in enumerate(ratings_df['userId'].unique())}
iid_map = {i:j for j,i in enumerate(ratings_df['movieId'].unique())}

ratings_df['uid'] = ratings_df['userId'].map(uid_map)
ratings_df['iid'] = ratings_df['movieId'].map(iid_map)

# Normalize timestamps to seconds (ML-100K is already epoch seconds)
if 'timestamp' not in ratings_df.columns:
    ratings_df['timestamp'] = 0

# Sort by user then time
ratings_df = ratings_df.sort_values(['uid','timestamp']).reset_index(drop=True)

n_users = ratings_df['uid'].nunique()
n_items = ratings_df['iid'].nunique()
print(f'Users: {n_users}, Items: {n_items}, Interactions: {len(ratings_df)}')

# Time-based split per user: last 2 interactions -> test, previous 2 -> val (if available)
def train_val_test_split_per_user(df):
    train_idx, val_idx, test_idx = [], [], []
    for u, g in df.groupby('uid'):
        idxs = g.index.tolist()
        if len(idxs) <= 4:
            # small histories: 1 test, 1 val if possible
            if len(idxs) >= 1: test_idx.append(idxs[-1])
            if len(idxs) >= 2: val_idx.append(idxs[-2])
            train_idx += idxs[:-2] if len(idxs) > 2 else []
        else:
            test_idx += idxs[-2:]
            val_idx += idxs[-4:-2]
            train_idx += idxs[:-4]
    return train_idx, val_idx, test_idx

train_idx, val_idx, test_idx = train_val_test_split_per_user(ratings_df)

train_df = ratings_df.loc[train_idx].reset_index(drop=True)
val_df   = ratings_df.loc[val_idx].reset_index(drop=True)
test_df  = ratings_df.loc[test_idx].reset_index(drop=True)

print('Split sizes:', len(train_df), len(val_df), len(test_df))


# 
# ## 4. 2006-Style Baseline: SVD++ (Matrix Factorization with Implicit Feedback)
# 
# This baseline mirrors the Netflix Prize era approach (matrix factorization with biases and implicit feedback).
# We evaluate on our **time-based** validation/test splits for fair comparison.
# 



# Surprise expects columns: user, item, rating (strings/ids), so we pass original IDs as strings.
reader = Reader(rating_scale=(1,5))

# Build trainset with our train_df
train_surprise = Dataset.load_from_df(train_df[['userId','movieId','rating']].astype(str), reader).build_full_trainset()

# Build val/test "anti" sets to compute predictions for their pairs
val_pairs = list(zip(val_df['userId'].astype(str), val_df['movieId'].astype(str), val_df['rating'].astype(float)))
test_pairs = list(zip(test_df['userId'].astype(str), test_df['movieId'].astype(str), test_df['rating'].astype(float)))

algo = SVDpp(n_factors=50, n_epochs=20, reg_all=0.02, lr_all=0.005, random_state=42)
algo.fit(train_surprise)

def eval_pairs(algo, pairs):
    y_true, y_pred = [], []
    for u,i,r in pairs:
        y_true.append(r)
        y_pred.append(algo.predict(u,i).est)
    return math.sqrt(mean_squared_error(y_true, y_pred))

svdpp_val_rmse = eval_pairs(algo, val_pairs)
svdpp_test_rmse = eval_pairs(algo, test_pairs)

print(f"SVD++  Val RMSE: {svdpp_val_rmse:.4f}")
print(f"SVD++ Test RMSE: {svdpp_test_rmse:.4f}")


# 
# ## 5. Item-Item Graph (Co-occurrence) for Graph Regularization
# We compute a lightweight **item-item similarity** from co-ratings and keep **top-K neighbors** per item.  
# This provides a graph-regularization term encouraging similar items to have similar embeddings.
# 



from collections import defaultdict

K_NEIGHBORS = 20  # top-K neighbors per item

# Build item -> set(users) map from *train* only
item_users = defaultdict(set)
for row in train_df[['uid','iid']].itertuples(index=False):
    item_users[row.iid].add(row.uid)

# Compute Jaccard similarities (sparse, only for items that share users)
neighbors = {}
items = list(item_users.keys())
for i in tqdm(items, desc='Compute item neighbors'):
    sims = []
    A = item_users[i]
    if not A:
        neighbors[i] = []
        continue
    # Sample to limit runtime on larger sets
    for j in items:
        if j == i: 
            continue
        B = item_users[j]
        if not B:
            continue
        inter = len(A & B)
        if inter == 0:
            continue
        union = len(A | B)
        sim = inter / union
        if sim > 0:
            sims.append((j, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    neighbors[i] = sims[:K_NEIGHBORS]

sum_edges = sum(len(v) for v in neighbors.values())
print(f"Total neighbor edges (directed): {sum_edges}")


# 
# ## 6. Build Temporal Sequences per User
# We will create **user sequences** (ordered by time) and sample (sequence, target item, rating) tuples for training the **Transformer** model.
# 



MAX_SEQ_LEN = 20  # keep recent history
RATING_MEAN = train_df['rating'].mean()

def make_user_histories(df):
    histories = {}
    for uid, g in df.groupby('uid'):
        seq = list(zip(g['iid'].tolist(), g['rating'].astype(float).tolist(), g['timestamp'].tolist()))
        histories[uid] = seq
    return histories

train_hist = make_user_histories(train_df)
val_hist   = make_user_histories(val_df)
test_hist  = make_user_histories(test_df)

class SeqRatingDataset(TorchDataset):
    def __init__(self, histories, max_len=20):
        self.samples = []
        for uid, seq in histories.items():
            if len(seq) < 2:
                continue
            for t in range(1, len(seq)):
                past = seq[max(0, t-max_len):t]
                target_iid, target_rating, target_ts = seq[t]
                past_iids = [p[0] for p in past]
                past_ts   = [p[2] for p in past]
                self.samples.append((uid, past_iids, past_ts, target_iid, target_rating))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    # pad sequences with -1
    uids, seq_iids, seq_ts, target_iids, target_r = [], [], [], [], []
    max_len = max(len(x[1]) for x in batch) if batch else 0
    for (uid, past_iids, past_ts, tgt_iid, tgt_r) in batch:
        pad_len = max_len - len(past_iids)
        uids.append(uid)
        seq_iids.append(past_iids + [-1]*pad_len)
        seq_ts.append(past_ts + [0]*pad_len)
        target_iids.append(tgt_iid)
        target_r.append(tgt_r)
    return (
        torch.tensor(uids, dtype=torch.long),
        torch.tensor(seq_iids, dtype=torch.long),
        torch.tensor(seq_ts, dtype=torch.long),
        torch.tensor(target_iids, dtype=torch.long),
        torch.tensor(target_r, dtype=torch.float32)
    )

train_ds = SeqRatingDataset(train_hist, MAX_SEQ_LEN)
val_ds   = SeqRatingDataset(val_hist, MAX_SEQ_LEN)
test_ds  = SeqRatingDataset(test_hist, MAX_SEQ_LEN)

train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn)
val_dl   = DataLoader(val_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)
test_dl  = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)

len(train_ds), len(val_ds), len(test_ds)


# 
# ## 7. 2025-Style Model: **HG-TNR (Hybrid Graph + Temporal Transformer) – Lite**
# 
# **Components:**
# - **Embeddings:** `user_emb`, `item_emb`, plus bias terms.
# - **Temporal Encoder:** Transformer encoder over the user's historical item sequence.
# - **Graph Regularization:** Encourage similar items (based on co-occurrence) to have similar embeddings.
# - **Predictor:** Combines **contextual user state** with target item embedding to **predict rating** (1–5).
# 
# > This is intentionally compact and runs on CPU in class. You can scale dimensions/epochs for better scores later.
# 



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]

class HGTNRLite(nn.Module):
    def __init__(self, n_users, n_items, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.d_model = d_model

        self.user_emb = nn.Embedding(n_users, d_model)
        self.item_emb = nn.Embedding(n_items, d_model)
        self.item_bias = nn.Embedding(n_items, 1)
        self.user_bias = nn.Embedding(n_users, 1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.posenc = PositionalEncoding(d_model)

        self.dropout = nn.Dropout(dropout)

        # Prediction head
        self.out = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        # init
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, uid, seq_iids, target_iid):
        # Embeddings
        seq_mask = (seq_iids >= 0).float()   # 1 for valid, 0 for pad
        seq_iids_clamped = seq_iids.clamp(min=0)
        seq_e = self.item_emb(seq_iids_clamped) * seq_mask.unsqueeze(-1)

        seq_e = self.posenc(seq_e)
        # Generate key padding mask: True for PADs
        key_padding_mask = (seq_iids < 0)
        h = self.encoder(seq_e, src_key_padding_mask=key_padding_mask)  # (B,T,D)

        # Take pooled representation (mean over valid positions)
        denom = seq_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        user_ctx = (h * seq_mask.unsqueeze(-1)).sum(dim=1) / denom

        # Combine with static user embedding (gated)
        u_static = self.user_emb(uid)
        user_repr = self.dropout(torch.cat([user_ctx, u_static], dim=-1))  # (B, 2D)

        item_repr = self.item_emb(target_iid)
        x = torch.cat([user_ctx, item_repr], dim=-1)  # contextual user + target item

        pred = self.out(x).squeeze(-1)
        # Add biases and clamp to 1..5 during eval (not during training to allow gradient flow)
        pred = pred + self.item_bias(target_iid).squeeze(-1) + self.user_bias(uid).squeeze(-1)
        return pred

def graph_reg_loss(item_emb_table, neighbors, lam=1e-4, device='cpu'):
    # Sum lam * w_ij * ||e_i - e_j||^2 over neighbor pairs
    loss = torch.tensor(0.0, device=device)
    for i, neighs in neighbors.items():
        if not neighs:
            continue
        e_i = item_emb_table.weight[i]
        for j, w in neighs:
            e_j = item_emb_table.weight[j]
            loss = loss + lam * w * torch.sum((e_i - e_j) ** 2)
    return loss


# 
# ## 8. Train the HG-TNR Model
# We optimize **MSE (for ratings)** + **Graph Regularization**. We report **RMSE** on validation/test sets.
# 



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = HGTNRLite(n_users=n_users, n_items=n_items, d_model=64, n_heads=4, n_layers=2, dropout=0.1).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

EPOCHS = 5  # keep small for classroom demo; increase later (e.g., 15-30) for better results
LAMBDA_GRAPH = 1e-5

def run_epoch(dl, train=True):
    model.train(train)
    losses = []
    for (uids, seq_iids, seq_ts, tgt_iids, tgt_r) in dl:
        uids = uids.to(device)
        seq_iids = seq_iids.to(device)
        tgt_iids = tgt_iids.to(device)
        tgt_r = tgt_r.to(device)

        preds = model(uids, seq_iids, tgt_iids)
        mse = F.mse_loss(preds, tgt_r)

        loss = mse
        if train and LAMBDA_GRAPH > 0:
            loss = loss + graph_reg_loss(model.item_emb, neighbors, lam=LAMBDA_GRAPH, device=device)

        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        losses.append(loss.item())
    return float(np.mean(losses))

def evaluate_rmse(dl):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for (uids, seq_iids, seq_ts, tgt_iids, tgt_r) in dl:
            uids = uids.to(device)
            seq_iids = seq_iids.to(device)
            tgt_iids = tgt_iids.to(device)
            preds = model(uids, seq_iids, tgt_iids)
            # Clamp predictions to rating range for RMSE eval
            preds = preds.clamp(1.0, 5.0)
            y_true.append(tgt_r.numpy())
            y_pred.append(preds.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return math.sqrt(mean_squared_error(y_true, y_pred))

history = {'train_loss':[], 'val_rmse':[]}
best_val = 1e9
best_state = None

for ep in range(1, EPOCHS+1):
    tl = run_epoch(train_dl, train=True)
    vr = evaluate_rmse(val_dl)
    history['train_loss'].append(tl)
    history['val_rmse'].append(vr)
    print(f'Epoch {ep:02d}: train_loss={tl:.4f}  val_RMSE={vr:.4f}')
    if vr < best_val:
        best_val = vr
        best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

# Load best and evaluate on test
if best_state is not None:
    model.load_state_dict(best_state)

test_rmse = evaluate_rmse(test_dl)
print(f"HG-TNR Test RMSE: {test_rmse:.4f}")


# 
# ## 9. Compare Baseline vs HG-TNR & Plot Training
# 



print(f"SVD++ Test RMSE: {svdpp_test_rmse:.4f}")
print(f"HG-TNR Test RMSE: {test_rmse:.4f}")

plt.figure()
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_rmse'], label='Val RMSE')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('HG-TNR Training')
plt.legend()
plt.show()


# 
# ## 10. (Optional) Simple Ranking Metric @10
# For a small random user set, compute **Hit@10** by ranking items by predicted rating using the user's context.
# 



def hit_at_k(model, user_ids, k=10, sample_items=500):
    model.eval()
    hits = []
    rng = np.random.default_rng(42)
    with torch.no_grad():
        for u in user_ids:
            hist = train_hist.get(u, [])
            if len(hist) < 1:
                continue
            # Build a recent sequence
            seq = hist[-MAX_SEQ_LEN:]
            seq_iids = [x[0] for x in seq]
            # Pick a target from val/test for this user if available
            target_pool = (val_hist.get(u, []) + test_hist.get(u, []))
            if not target_pool:
                continue
            tgt_iid = target_pool[-1][0]

            # Negative sampling (random items)
            cand = set([tgt_iid])
            while len(cand) < sample_items:
                cand.add(rng.integers(0, n_items))
            cand = list(cand)

            # Score all candidates
            uid_t = torch.tensor([u]*len(cand)).to(device)
            seq_pad = torch.tensor([seq_iids + [-1]*(MAX_SEQ_LEN-len(seq_iids))]*len(cand)).to(device)
            cand_t = torch.tensor(cand).to(device)
            scores = model(uid_t, seq_pad, cand_t).cpu().numpy()

            # Rank and check hit
            topk = np.argsort(-scores)[:k]
            top_items = [cand[i] for i in topk]
            hits.append(1.0 if tgt_iid in top_items else 0.0)
    if hits:
        return float(np.mean(hits))
    return float('nan')

sample_users = list(train_hist.keys())[:200]
h10 = hit_at_k(model, sample_users, k=10, sample_items=500)
print(f"HG-TNR Hit@10 (sample users): {h10:.3f}")


# 
# ## 11. Save Artifacts (Optional)
# Save the trained HG-TNR model state dict and metrics.
# 



ART_DIR = Path('./artifacts')
ART_DIR.mkdir(exist_ok=True, parents=True)

torch.save(model.state_dict(), ART_DIR / 'hgtnr_lite_state_dict.pt')
with open(ART_DIR / 'metrics.json','w') as f:
    json.dump({
        'svdpp_val_rmse': svdpp_val_rmse,
        'svdpp_test_rmse': svdpp_test_rmse,
        'hgtnr_test_rmse': test_rmse
    }, f, indent=2)

print('Saved artifacts to', ART_DIR.resolve())


# 
# ---
# 
# ### What to demonstrate live
# 1. Show the **baseline SVD++** cell and its RMSE (connect to 2006 Netflix Prize era).
# 2. Explain the **graph neighbors** computation (why it helps long-tail similarity).
# 3. Walk through the **HG-TNR model** (Transformer for temporal context + graph reg for item geometry).
# 4. Train for a few epochs (already set short) and show **val/test RMSE** and the training plot.
# 5. If time permits, run **Hit@10** for a ranking flavor.
# 
# > You can later scale embedding sizes, epochs, and graph K to approach better accuracy.
#