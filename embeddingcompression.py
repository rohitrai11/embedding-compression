import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from typing import List, Dict, Any

# --- Configuration ---
CONFIG = {
    "model_name": "bert-base-uncased",
    "latent_dim": 50,
    "epochs": 40,
    "batch_size": 512,
    "lr": 1e-3,
    "lambda_contrast": 2.5,
    "temperatures": [0.1],
    "knn_pos": 20,
    "knn_centroid": 5,
    "eval_subset": 10000,
    "seed": 42
}

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
device = setup_device()

def load_data_split(split="train"):
    print(f"Loading dataset (split={split})...")
    # Using the dataset specified in your data.py
    dataset_hf = load_dataset("embedding-data/sentence-compression", split=split)
    text_column = dataset_hf.column_names[0]
    return dataset_hf[text_column]

def get_embeddings(sentences, model_name, device, batch_size=64):
    print(f"Encoding {len(sentences)} sentences with {model_name}...")
    model_st = SentenceTransformer(model_name, device=device)
    embeddings = model_st.encode(
        sentences,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=batch_size
    )
    return embeddings.cpu()

def get_nearest_neighbors_gpu(embeddings, k, batch_size, device):
    """Computes exact k-nearest neighbors using cosine similarity on GPU."""
    print(f"Computing {k}-NN on GPU in batches...")

    num_samples = embeddings.size(0)

    # Move to GPU for calculation
    embeddings_gpu = embeddings.to(device)
    if not isinstance(embeddings_gpu, torch.Tensor):
        embeddings_gpu = torch.tensor(embeddings_gpu).to(device)

    embeddings_gpu = F.normalize(embeddings_gpu, p=2, dim=1)

    indices_list = []

    # Process queries in batches
    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        batch = embeddings_gpu[i:end]

        # Sim: (Batch, Total)
        sim_matrix = torch.matmul(batch, embeddings_gpu.T)
        vals, inds = torch.topk(sim_matrix, k=k, dim=1)

        indices_list.append(inds.cpu())

    # Free GPU memory
    del embeddings_gpu
    torch.cuda.empty_cache()

    return torch.cat(indices_list, dim=0).numpy()

if os.path.exists("embeddings_bert.pt"):
    print("Found existing embeddings_bert.pt, loading...")
    base_embeddings = torch.load("embeddings_bert.pt", map_location=device)
else:
    print("Generating new BERT embeddings...")
    sentences = load_data_split(split="train")
    # You might want to limit sentences for debugging if dataset is huge
    # sentences = sentences[:20000]

    base_embeddings = get_embeddings(sentences, CONFIG["model_name"], device, batch_size=CONFIG["batch_size"])
    torch.save(base_embeddings, "embeddings_bert.pt")
    print(f"Saved embeddings_bert.pt with shape {base_embeddings.shape}")

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=50):
        super().__init__()

        # 1. Encoder (Compress: 768 -> 50)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ELU(),

            nn.Linear(512, latent_dim)
        )

        # 2. Decoder (Reconstruct: 50 -> 768)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ELU(),

            nn.Linear(512, input_dim)
        )

    def forward(self, x):
      z = self.encoder(x)
      z = F.normalize(z, p=2, dim=1) # Vital for Cosine Similarity
      x_hat = self.decoder(z)
      return z, x_hat

mse_loss_fn = nn.MSELoss()

def info_nce(z, z_pos, temperature):
    z = F.normalize(z, dim=1)
    z_pos = F.normalize(z_pos, dim=1)

    # Cosine similarity matrix: (Batch, Batch)
    logits = torch.matmul(z, z_pos.T) / temperature

    # Labels are diagonal (0, 1, 2... batch_size-1)
    labels = torch.arange(z.size(0)).to(z.device)

    return F.cross_entropy(logits, labels)

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import notebook_launcher
from accelerate import Accelerator
import torch.optim as optim

def distributed_training_wrapper():
    # 1. Initialize Accelerator inside the function
    accelerator = Accelerator()
    device = accelerator.device

    # 2. Instantiate Model, Optimizer, and DataLoader inside the function
    # (Creating them here ensures each GPU gets its own clean copy)
    model = AutoEncoder(input_dim=768, latent_dim=CONFIG["latent_dim"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # Standard PyTorch DataLoader (Accelerate will handle the distributed sampling)
    indices = torch.arange(len(base_embeddings))
    dataset = TensorDataset(base_embeddings, indices)
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # 3. THE KEY STEP: Prepare all objects
    # This automatically handles multi-GPU sharding
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # 4. Training Loop
    model.train()
    for epoch in range(CONFIG["epochs"]):
        for x, idx in loader:
            optimizer.zero_grad()
            z, x_hat = model(x)

            # Loss Calculation (Logic from your notebook)
            loss_rec = mse_loss_fn(x_hat, x)
            # ... add your info_nce loss here ...
            loss = loss_rec # + CONFIG["lambda_contrast"] * loss_con

            # 5. USE THIS instead of loss.backward()
            accelerator.backward(loss)
            optimizer.step()

        # Optional: Only print from the main process to avoid messy logs
        if accelerator.is_main_process:
            print(f"Epoch {epoch} complete.")

# 6. LAUNCHER: This starts the multi-GPU processes
# num_processes=2 for Kaggle T4x2, or use torch.cuda.device_count() for auto-detection
num_gpus = torch.cuda.device_count()
notebook_launcher(distributed_training_wrapper, num_processes=num_gpus)

def train_one_temperature(tau, model, X, nn_indices, config, device):
    print(f"\n===== Training: Retrieval-Optimized Mode | τ = {tau} =====")
    indices = torch.arange(len(X))
    loader = DataLoader(TensorDataset(X, indices), batch_size=config["batch_size"], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    model.train()
    for epoch in range(config["epochs"]):
        total_loss, total_rec, total_con = 0.0, 0.0, 0.0

        for x, idx in loader:
            x = x.to(device)
            optimizer.zero_grad()

            # 1. Forward Pass (Normalization is handled inside model.forward)
            z, x_hat = model(x)

            # 2. Positive Pair (1st Nearest Neighbor)
            pos_idx = nn_indices[idx.numpy(), 1]
            z_pos, _ = model(X[pos_idx].to(device))

            # 3. Hard Negative (10th Nearest Neighbor)
            # This forces the model to learn fine-grained semantic differences
            neg_idx = nn_indices[idx.numpy(), 10]
            z_neg, _ = model(X[neg_idx].to(device))

            # 4. Multi-term Loss
            loss_rec = mse_loss_fn(x_hat, x)

            # Standard InfoNCE
            loss_nce = info_nce(z, z_pos, tau)

            # Hard Negative Penalty: (1 - similarity) should be maximized
            # We want dot product (z * z_neg) to be low
            loss_hard = torch.mean(torch.clamp(torch.sum(z * z_neg, dim=1), min=0))

            # Combine: prioritize semantic separation (loss_nce + loss_hard)
            loss = loss_rec + config["lambda_contrast"] * (loss_nce + 0.5 * loss_hard)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_rec += loss_rec.item() * x.size(0)
            total_con += (loss_nce + loss_hard).item() * x.size(0)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {total_loss/len(X):.4f} | Con: {total_con/len(X):.4f}")

    return model

def encode_latent(model, embeddings, batch_size, device):
    model.eval()
    compressed_list = []
    tensor_embeddings = embeddings
    if not isinstance(tensor_embeddings, torch.Tensor):
        tensor_embeddings = torch.tensor(tensor_embeddings, dtype=torch.float32)

    with torch.no_grad():
        for i in range(0, len(tensor_embeddings), batch_size):
            batch = tensor_embeddings[i : i + batch_size].to(device)
            z = model.encoder(batch)
            compressed_list.append(z.cpu())

    return torch.cat(compressed_list, dim=0)

print("Computing nearest neighbors for contrastive positive pairs...")
# Ensure base_embeddings is tensor
if isinstance(base_embeddings, np.ndarray):
    base_embeddings = torch.from_numpy(base_embeddings)

nn_indices = get_nearest_neighbors_gpu(base_embeddings, k=CONFIG["knn_pos"], batch_size=CONFIG["batch_size"], device=device)

for tau in CONFIG["temperatures"]:
    print(f"\n>>> Compressing with Temperature tau={tau} <<<")

    # Initialize AutoEncoder
    model = AutoEncoder(input_dim=768, latent_dim=CONFIG["latent_dim"]).to(device)

    # Train
    model = train_one_temperature(tau, model, base_embeddings, nn_indices, CONFIG, device)

    # Save Model
    save_name = f"model_compressed_bert_tau_{tau}.pt"
    torch.save(model.state_dict(), save_name)
    print(f"Saved model to {save_name}")

    # Generate/Encode Compressed Embeddings
    z = encode_latent(model, base_embeddings, CONFIG["batch_size"], device)

    z_save_name = f"embeddings_compressed_bert_tau_{tau}.pt"
    torch.save(z, z_save_name)
    print(f"Saved compressed embeddings to {z_save_name}")

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from ragas import evaluate
from datasets import Dataset
from langchain_community.chat_models import ChatOllama

# Hardware check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class EvaluationWrapper:
    def __init__(self, model_name: str, device_in):
        self.device = device_in
        print(f"Loading model {model_name}...")
        self.base_model = SentenceTransformer(model_name, device=self.device)
        self.pca_model = None
        self.compressor = None

    def encode_base(self, sentences: List[str], batch_size: int = 128):
        return self.base_model.encode(
sentences,
convert_to_numpy=True,
batch_size=batch_size,
show_progress_bar=True
)

    def train_pca(self, embeddings: np.ndarray, n_components: int = 50):
        print(f"Training PCA to {n_components} dims...")
        self.pca_model = PCA(n_components=n_components)
        self.pca_model.fit(embeddings)

    def load_compressor(self, model_path: str, input_dim: int = 768, latent_dim: int = 50):
        # Ensure 'model.py' is available in your workspace
        # from model import AutoEncoder # This line is removed as AutoEncoder is defined in the notebook
        self.compressor = AutoEncoder(input_dim, latent_dim).to(self.device)
        self.compressor.load_state_dict(torch.load(model_path, map_location=self.device))
        self.compressor.eval()

    def encode_pca(self, embeddings: np.ndarray):
        return self.pca_model.transform(embeddings)

    def encode_compressed(self, embeddings: np.ndarray):
        compressed_list = []
        tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, len(embeddings), 512):
                batch = tensor_embeddings[i : i + 512].to(self.device)
                z = self.compressor.encoder(batch)
                compressed_list.append(z.cpu().numpy())
        return np.concatenate(compressed_list, axis=0)

class RagasEmbeddingsWrapper:
    def __init__(self, func): self.func = func
    def embed_documents(self, texts): return self.func(texts).tolist()
    def embed_query(self, text): return self.func([text])[0].tolist()

dataset_name = "hotpotqa"
data_path = f"datasets/{dataset_name}"

if not os.path.exists(os.path.join(data_path, "qrels")):
    from beir import util
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    util.download_and_unzip(url, "datasets")

corpus, queries, qrels = GenericDataLoader(data_path).load(split="dev")

# Create subset (Gold + 10k Distractors)
np.random.seed(42)
gold_doc_ids = set()
for qid in qrels:
    gold_doc_ids.update(qrels[qid].keys())

remaining_ids = list(set(corpus.keys()) - gold_doc_ids)
distractors = np.random.choice(remaining_ids, min(len(remaining_ids), 10000), replace=False).tolist()
final_doc_ids = list(gold_doc_ids) + distractors

subset_corpus = {did: corpus[did] for did in final_doc_ids}
docs_text = [subset_corpus[did]['title'] + " " + subset_corpus[did]['text'] for did in final_doc_ids]

print(f"Dataset Ready: {len(docs_text)} documents in retrieval pool.")

# Initialize Evaluators
bert_eval = EvaluationWrapper("bert-base-uncased", device)
distil_eval = EvaluationWrapper("distilbert-base-uncased", device)

# 1. Base Encodings (Heavy Operation)
print("Encoding corpus with BERT-base...")
base_doc_embs = bert_eval.encode_base(docs_text)
print("Encoding corpus with DistilBERT...")
distil_doc_embs = distil_eval.encode_base(docs_text)

# 2. PCA Training
bert_eval.train_pca(base_doc_embs, n_components=50)

# 3. Contrastive Setup (if file exists)
contrastive_ready = False
model_files = glob.glob("model_compressed_bert_tau_*.pt")
if model_files:
    selected_model = next((m for m in model_files if "0.05" in m), model_files[0])
    bert_eval.load_compressor(selected_model)
    contrastive_ready = True

# # 4. Prepare Queries (Subset for speed)
# #num_queries = 100
# query_ids = list(queries.keys())#[:num_queries]
# queries_text = [queries[qid] for qid in query_ids]

# base_query_embs = bert_eval.encode_base(queries_text)
# distil_query_embs = distil_eval.encode_base(queries_text)

# # Package all variations for testing
# variations = {
#     "BERT-Original (768D)": (base_query_embs, base_doc_embs),
#     "DistilBERT (768D)": (distil_query_embs, distil_doc_embs),
#     "BERT-PCA (50D)": (bert_eval.encode_pca(base_query_embs), bert_eval.encode_pca(base_doc_embs))
# }

# if contrastive_ready:
#     variations["BERT-Contrastive (50D)"] = (bert_eval.encode_compressed(base_query_embs), bert_eval.encode_compressed(base_doc_embs))

# print(f"All {len(variations)} variations encoded and ready for evaluation.")

# ============================================================
#  FULL DATASET EVALUATION (PURE PYTHON)
# ============================================================

# 1. Prepare the full list of queries from the loaded split
# This will evaluate all 7,405 queries in the dev split automatically
query_ids = list(queries.keys())
queries_text = [queries[qid] for qid in query_ids]

# Re-encode query embeddings for the full set (if not already done)
# This ensures we have embeddings for every query in the full set
print(f"Re-encoding {len(query_ids)} queries for full evaluation...")
base_query_embs = bert_eval.encode_base(queries_text)
distil_query_embs = distil_eval.encode_base(queries_text)

variations = {}

# Base
variations["BERT-Original (768D)"] = (
    base_query_embs,
    base_doc_embs
)

# Distil
variations["DistilBERT (768D)"] = (
    distil_query_embs,
    distil_doc_embs
)

# PCA
variations["BERT-PCA (50D)"] = (
    bert_eval.encode_pca(base_query_embs),
    bert_eval.encode_pca(base_doc_embs)
)

# Contrastive
if contrastive_ready:
    variations["BERT-Contrastive (50D)"] = (
        bert_eval.encode_compressed(base_query_embs),
        bert_eval.encode_compressed(base_doc_embs)
    )

'''
variations = {
     "BERT-Original (768D)": (base_query_embs, base_doc_embs),
     "DistilBERT (768D)": (distil_query_embs, distil_doc_embs),
     "BERT-PCA (50D)": (bert_eval.encode_pca(base_query_embs), bert_eval.encode_pca(base_doc_embs))
}

# Update the variations dictionary with the full-set query embeddings
variations["BERT-Original (768D)"] = (base_query_embs, variations["BERT-Original (768D)"][1])
variations["DistilBERT (768D)"] = (distil_query_embs, variations["DistilBERT (768D)"][1])
variations["BERT-PCA (50D)"] = (bert_eval.encode_pca(base_query_embs), variations["BERT-PCA (50D)"][1])

if contrastive_ready:
    variations["BERT-Contrastive (50D)"] = (bert_eval.encode_compressed(base_query_embs), variations["BERT-Contrastive (50D)"][1])
'''
results_dict = {}

print("\n" + "="*70)
print(f"STARTING FULL EVALUATION ({len(query_ids)} QUERIES)")
print("="*70)

for name, (q_embs, d_embs) in variations.items():
    print(f"Evaluating: {name}...")

    # 1. Cosine Similarity Retrieval
    d_embs_t = F.normalize(torch.tensor(d_embs).to(device), p=2, dim=1)
    q_embs_t = F.normalize(torch.tensor(q_embs).to(device), p=2, dim=1)

    # Use batches if memory is an issue for 7k+ queries
    scores = torch.matmul(q_embs_t, d_embs_t.T)
    top_scores, top_inds = torch.topk(scores, k=10)

    # 2. Prepare BEIR format
    beir_results = {}
    for i, qid in enumerate(query_ids):
        indices = top_inds[i].cpu().numpy()
        retrieved_ids = [final_doc_ids[idx] for idx in indices]
        beir_results[qid] = {did: float(s) for did, s in zip(retrieved_ids, top_scores[i].cpu().numpy())}

    # 3. Calculate BEIR Metrics
    retriever_eval = EvaluateRetrieval(k_values=[10])
    subset_qrels = {qid: qrels[qid] for qid in query_ids}

    ndcg, _, recall, _ = retriever_eval.evaluate(subset_qrels, beir_results, [10])
    mrr = retriever_eval.evaluate_custom(subset_qrels, beir_results, [10], metric="mrr")

    # Store results
    for m_name, val in [("Recall@10", recall["Recall@10"]),
                       ("NDCG@10", ndcg["NDCG@10"]),
                       ("MRR@10", mrr["MRR@10"])]:
        if m_name not in results_dict:
            results_dict[m_name] = {}
        results_dict[m_name][name] = val

# ============================================================
# PRINT RESULTS AS A FORMATTED TABLE
# ============================================================
models = list(variations.keys())
metrics = list(results_dict.keys())
metric_col_width, model_col_width = 15, 25

header = f"{'Metric':<{metric_col_width}}" + "".join([f"{m:<{model_col_width}}" for m in models])
print("\n" + "=" * len(header))
print(f"FINAL FULL BENCHMARK RESULTS (N={len(query_ids)})")
print("-" * len(header))
print(header)
print("-" * len(header))

for metric in metrics:
    row = f"{metric:<{metric_col_width}}"
    for model in models:
        score = results_dict[metric].get(model, 0.0)
        row += f"{score:<{model_col_width}.4f}"
    print(row)
print("=" * len(header))