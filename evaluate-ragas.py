"""
Standalone RAGAS evaluation with local Ollama judge (ragas==0.4.3)

- Loads BEIR hotpotqa dev split
- Builds retrieval pool: all gold docs + N distractors
- Builds model variations:
    1) BERT-Original (768D)
    2) DistilBERT (768D)
    3) BERT-PCA (50D)
    4) (Optional) BERT-AE (50D) if you provide --ae_ckpt path
- Retrieves top-k contexts per query with cosine similarity
- Runs RAGAS: context_recall and context_precision using Ollama as LLM judge
- Uses truncation and RunConfig(max_workers=1) for stability

Run:
  python evaluate-ragas.py

Optional:
  python evaluate-ragas.py --num_queries 200 --top_k 10 --distractors 10000
  python evaluate-ragas.py --ae_ckpt model_compressed_bert_tau_0.1.pt
"""

import os
import argparse
import math
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from beir.datasets.data_loader import GenericDataLoader
from beir import util

from ragas import evaluate
from ragas.metrics import context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings


# ----------------------------
# AutoEncoder (optional AE variation)
# ----------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=50):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z, p=2, dim=1)
        x_hat = self.decoder(z)
        return z, x_hat


# ----------------------------
# Helpers
# ----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clip_text(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else (s[:n] + " ...")


def download_beir_dataset(dataset_name: str, root: str) -> str:
    out_dir = os.path.join(root, dataset_name)
    if os.path.exists(os.path.join(out_dir, "qrels")):
        return out_dir
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    util.download_and_unzip(url, root)
    return out_dir


def build_subset_pool(
    corpus: Dict[str, Dict[str, str]],
    qrels: Dict[str, Dict[str, int]],
    num_distractors: int,
    seed: int,
) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    gold_doc_ids = set()
    for qid in qrels:
        gold_doc_ids.update(qrels[qid].keys())

    remaining = list(set(corpus.keys()) - gold_doc_ids)
    rng = np.random.default_rng(seed)

    distractors = []
    if remaining and num_distractors > 0:
        distractors = rng.choice(
            remaining, size=min(num_distractors, len(remaining)), replace=False
        ).tolist()

    final_doc_ids = list(gold_doc_ids) + distractors
    subset_corpus = {did: corpus[did] for did in final_doc_ids}
    return final_doc_ids, subset_corpus


def docs_to_texts(doc_ids: List[str], corpus: Dict[str, Dict[str, str]]) -> List[str]:
    texts = []
    for did in doc_ids:
        title = corpus[did].get("title", "") or ""
        text = corpus[did].get("text", "") or ""
        texts.append((title + " " + text).strip())
    return texts


def st_encode(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


@torch.no_grad()
def topk_retrieve(q_embs: np.ndarray, d_embs: np.ndarray, top_k: int, device: torch.device) -> np.ndarray:
    q = torch.tensor(q_embs, dtype=torch.float32, device=device)
    d = torch.tensor(d_embs, dtype=torch.float32, device=device)
    q = F.normalize(q, p=2, dim=1)
    d = F.normalize(d, p=2, dim=1)
    scores = q @ d.T
    _, inds = torch.topk(scores, k=top_k, dim=1)
    return inds.cpu().numpy()


def build_ragas_dataset(
    query_ids: List[str],
    queries: Dict[str, str],
    doc_ids: List[str],
    subset_corpus: Dict[str, Dict[str, str]],
    corpus_full: Dict[str, Dict[str, str]],
    qrels: Dict[str, Dict[str, int]],
    top_inds: np.ndarray,
    max_context_chars: int,
    max_gt_chars: int,
) -> Dataset:
    questions = [queries[qid] for qid in query_ids]
    contexts, ground_truths = [], []

    for i, qid in enumerate(query_ids):
        inds = top_inds[i]
        ctx = []
        for j in inds:
            did = doc_ids[int(j)]
            ctx.append(clip_text(subset_corpus[did].get("text", ""), max_context_chars))
        contexts.append(ctx)

        rel = qrels.get(qid, {})
        gt_texts = []
        for did in rel.keys():
            if did in corpus_full:
                gt_texts.append(corpus_full[did].get("text", "") or "")
        ground_truths.append(clip_text(" ".join(gt_texts).strip(), max_gt_chars))

    return Dataset.from_dict({"question": questions, "contexts": contexts, "ground_truth": ground_truths})


@torch.no_grad()
def ae_encode(encoder: nn.Module, embs: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    x = torch.tensor(embs, dtype=torch.float32)
    out = []
    for i in range(0, len(x), batch_size):
        z = encoder(x[i:i+batch_size].to(device)).cpu()
        out.append(z)
    return torch.cat(out, dim=0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="hotpotqa")
    ap.add_argument("--split", type=str, default="dev")
    ap.add_argument("--data_root", type=str, default="datasets")
    ap.add_argument("--distractors", type=int, default=10000)
    ap.add_argument("--num_queries", type=int, default=50)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_context_chars", type=int, default=600)
    ap.add_argument("--max_gt_chars", type=int, default=1200)

    ap.add_argument("--ollama_model", type=str, default="llama3.1")
    ap.add_argument("--ollama_url", type=str, default="http://localhost:11434")

    ap.add_argument("--ragas_embed_model", type=str, default="BAAI/bge-small-en-v1.5")

    ap.add_argument("--bert_model", type=str, default="bert-base-uncased")
    ap.add_argument("--distil_model", type=str, default="distilbert-base-uncased")
    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument("--pca_dim", type=int, default=50)

    ap.add_argument("--ae_ckpt", type=str, default="")  # optional
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    # 1) Load BEIR
    ds_path = download_beir_dataset(args.dataset, args.data_root)
    corpus, queries, qrels = GenericDataLoader(ds_path).load(split=args.split)

    # 2) Build pool
    final_doc_ids, subset_corpus = build_subset_pool(corpus, qrels, args.distractors, args.seed)
    docs_text = docs_to_texts(final_doc_ids, subset_corpus)
    print(f"Retrieval pool docs: {len(docs_text)} (gold + up to {args.distractors} distractors)")

    # 3) Query subset
    all_qids = list(queries.keys())
    num_q = min(args.num_queries, len(all_qids))
    query_ids = all_qids[:num_q]
    query_text = [queries[qid] for qid in query_ids]
    print(f"Evaluating queries: {num_q} | top_k={args.top_k}")

    # 4) Build retrieval embeddings for doc+query using SentenceTransformer
    print(f"Loading retriever ST model: {args.bert_model}")
    bert_st = SentenceTransformer(args.bert_model, device=str(device))

    print("Encoding docs (BERT ST)...")
    bert_doc_embs = st_encode(bert_st, docs_text, args.batch_size)
    print("Encoding queries (BERT ST)...")
    bert_query_embs = st_encode(bert_st, query_text, args.batch_size)

    print(f"Loading retriever ST model: {args.distil_model}")
    distil_st = SentenceTransformer(args.distil_model, device=str(device))

    print("Encoding docs (Distil ST)...")
    distil_doc_embs = st_encode(distil_st, docs_text, args.batch_size)
    print("Encoding queries (Distil ST)...")
    distil_query_embs = st_encode(distil_st, query_text, args.batch_size)

    # 5) PCA on BERT docs
    print(f"Training PCA to {args.pca_dim} dims on BERT doc embeddings...")
    pca = PCA(n_components=args.pca_dim, random_state=args.seed)
    pca.fit(bert_doc_embs)

    bert_doc_pca = pca.transform(bert_doc_embs)
    bert_query_pca = pca.transform(bert_query_embs)

    # 6) Optional AE compressed
    ae_ready = False
    if args.ae_ckpt and os.path.exists(args.ae_ckpt):
        print(f"Loading AE checkpoint: {args.ae_ckpt}")
        ae = AutoEncoder(input_dim=bert_doc_embs.shape[1], latent_dim=args.pca_dim).to(device)
        ae.load_state_dict(torch.load(args.ae_ckpt, map_location=device))
        ae.eval()
        bert_doc_ae = ae_encode(ae.encoder, bert_doc_embs, device)
        bert_query_ae = ae_encode(ae.encoder, bert_query_embs, device)
        ae_ready = True
    elif args.ae_ckpt:
        print(f"AE checkpoint not found: {args.ae_ckpt} (skipping AE variation)")

    # 7) Variations
    variations = {
        "BERT-Original (768D)": (bert_query_embs, bert_doc_embs),
        "DistilBERT (768D)": (distil_query_embs, distil_doc_embs),
        f"BERT-PCA ({args.pca_dim}D)": (bert_query_pca, bert_doc_pca),
    }
    if ae_ready:
        variations[f"BERT-AE ({args.pca_dim}D)"] = (bert_query_ae, bert_doc_ae)

    # 8) Setup local Ollama judge
    print("Connecting to local Ollama...")
    judge_llm = ChatOllama(model=args.ollama_model, temperature=0, base_url=args.ollama_url)
    ragas_llm = LangchainLLMWrapper(judge_llm)

    # RAGAS embeddings
    hf_embeddings = HuggingFaceEmbeddings(model_name=args.ragas_embed_model)
    ragas_emb = LangchainEmbeddingsWrapper(hf_embeddings)

    run_config = RunConfig(max_workers=1, timeout=240, max_retries=3)

    # 9) Evaluate each variation
    results = {}
    for name, (q_embs, d_embs) in variations.items():
        print(f"\n=== RAGAS evaluating: {name} ===")
        top_inds = topk_retrieve(q_embs, d_embs, args.top_k, device)

        ragas_ds = build_ragas_dataset(
            query_ids=query_ids,
            queries=queries,
            doc_ids=final_doc_ids,
            subset_corpus=subset_corpus,
            corpus_full=corpus,
            qrels=qrels,
            top_inds=top_inds,
            max_context_chars=args.max_context_chars,
            max_gt_chars=args.max_gt_chars,
        )

        try:
            res = evaluate(
                ragas_ds,
                metrics=[context_recall, context_precision],
                llm=ragas_llm,
                embeddings=ragas_emb,
                run_config=run_config,
            )

            def to_scalar(x):
                if isinstance(x, list):
                    x = [v for v in x if v is not None]
                    return float(sum(x) / max(len(x), 1))
                return float(x)

            results[name] = {
                "Context Recall": to_scalar(res["context_recall"]),
                "Context Precision": to_scalar(res["context_precision"]),
            }
            print("✓ Done.")

        except Exception as e:
            print(f"⚠ Failed: {e}")
            results[name] = {"Context Recall": 0.0, "Context Precision": 0.0}


    # 10) Report
    print("\n" + "=" * 90)
    print("RAGAS CONTEXT METRICS (Ollama Judge)")
    print("-" * 90)
    print(f"{'Variation':<30} {'Context Recall':<20} {'Context Precision':<20}")
    print("-" * 90)
    for k, v in results.items():
        print(f"{k:<30} {v['Context Recall']:<20.4f} {v['Context Precision']:<20.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()

