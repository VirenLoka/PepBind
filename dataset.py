

import csv
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf


# ── Labels ────────────────────────────────────────────────────────────────────

LABEL_POSITIVE = 1   # active peptide
LABEL_NEGATIVE = 0   # inactive peptide


# ── I/O helpers ───────────────────────────────────────────────────────────────

def txt_to_sequences(txt_path: str | Path) -> List[str]:
    """Read a plain-text file and return one cleaned sequence per line."""
    sequences = []
    with open(txt_path, "r") as fh:
        for line in fh:
            seq = line.strip().upper()
            if seq:
                sequences.append(seq)
    return sequences


def txt_to_csv(txt_path: str, csv_path: str, column_name: str = "Sequence") -> None:
    """Convert a TXT file (one peptide per line) to a single-column CSV."""
    with open(txt_path, "r") as txt_file, open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([column_name])
        for line in txt_file:
            clean = line.strip().upper()
            if clean:
                writer.writerow([clean])
    print(f"Converted {txt_path} → {csv_path}")


def load_raw_data(cfg: DictConfig) -> Tuple[List[str], List[int]]:
    """Load positive and negative peptide TXT files → (sequences, labels)."""
    pos_seqs = txt_to_sequences(cfg.data.pos_file)
    neg_seqs = txt_to_sequences(cfg.data.neg_file)
    print(f"Loaded {len(pos_seqs):>6} positive sequences from '{cfg.data.pos_file}'")
    print(f"Loaded {len(neg_seqs):>6} negative sequences from '{cfg.data.neg_file}'")
    sequences = pos_seqs + neg_seqs
    labels    = [LABEL_POSITIVE] * len(pos_seqs) + [LABEL_NEGATIVE] * len(neg_seqs)
    return sequences, labels


# ── ESM2 embedding generation ─────────────────────────────────────────────────

def gen_embed(cfg: DictConfig) -> None:
    """
    Generate ESM2 embeddings for all peptides and cache them to a pickle file.

    The pickle is skipped if the file already exists. Delete the file manually
    or pass a fresh path in config.yaml to force regeneration.

    Pickle schema
    -------------
    {
        "<sequence>": {"pep_id": str, "embedding": np.ndarray, "label": int},
        ...
    }

    Parameters
    ----------
    cfg : OmegaConf DictConfig loaded from config.yaml.
    """
    pkl_path = Path(cfg.data.embeddings_file)

    # ── Cache check ───────────────────────────────────────────────────────────
    if pkl_path.exists():
        print(f"[gen_embed] Pickle already exists at '{pkl_path}'. Skipping generation.")
        return

    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ESM2 ─────────────────────────────────────────────────────────────
    try:
        import esm
    except ImportError as e:
        raise ImportError(
            "The 'esm' package is required. Install it with:\n"
            "  pip install fair-esm"
        ) from e

    device = torch.device(cfg.esm.device if torch.cuda.is_available() else "cpu")
    print(f"[gen_embed] Loading ESM2 model '{cfg.esm.model_name}' on {device} …")

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    # ESM2 t33_650M has 33 transformer layers; embeddings come from the last one
    repr_layer = 33

    # ── Load sequences ────────────────────────────────────────────────────────
    sequences, labels = load_raw_data(cfg)
    n_total = len(sequences)
    print(f"[gen_embed] Embedding {n_total} sequences …")

    embeddings_store: Dict[str, Dict] = {}
    esm_batch_size = cfg.esm.batch_size

    # ── Batch inference ───────────────────────────────────────────────────────
    for batch_start in range(0, n_total, esm_batch_size):
        batch_end   = min(batch_start + esm_batch_size, n_total)
        batch_seqs  = sequences[batch_start:batch_end]
        batch_lbls  = labels[batch_start:batch_end]

        # ESM expects a list of (label, sequence) tuples
        esm_input = [(str(i), seq) for i, seq in enumerate(batch_seqs)]

        batch_labels_esm, batch_strs, batch_tokens = batch_converter(esm_input)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(
                batch_tokens,
                repr_layers=[repr_layer],
                return_contacts=False,
            )

        token_representations = results["representations"][repr_layer]
        # token_representations shape: (batch, seq_len+2, embed_dim)
        # Index 0 = BOS token, index -1 = EOS token → slice [1:-1] for residues

        for i, (seq, lbl) in enumerate(zip(batch_seqs, batch_lbls)):
            global_idx = batch_start + i          # unique index across all batches
            pep_id     = f"PEP_{global_idx + 1:06d}"   # e.g. PEP_000001

            n_residues = len(seq)
            # Mean-pool over actual residue tokens only (exclude BOS/EOS)
            mean_embedding = (
                token_representations[i, 1 : n_residues + 1]
                .mean(dim=0)
                .cpu()
                .numpy()
            )  # shape: (embed_dim,)

            embeddings_store[pep_id] = {
                "pep_id":    pep_id,
                "embedding": mean_embedding,
                'pep_seq':  seq,
                "label":     lbl,
            }

        print(
            f"[gen_embed]  {min(batch_end, n_total):>6} / {n_total} sequences embedded",
            end="\r",
        )

    print(f"\n[gen_embed] Done. Saving pickle to '{pkl_path}' …")
    with open(pkl_path, "wb") as f:
        pickle.dump(embeddings_store, f)
    print(f"[gen_embed] Pickle saved ({pkl_path.stat().st_size / 1e6:.1f} MB).")


# ── Dataset ───────────────────────────────────────────────────────────────────

class PeptideDataset(Dataset):
    """
    Reads pre-computed ESM2 embeddings from a pickle file.

    Parameters
    ----------
    records : List of dicts with keys 'embedding' (np.ndarray) and 'label' (int).
    """

    def __init__(self, records: List[Dict]) -> None:
        self.pep_ids    = [r["pep_id"]    for r in records]
        self.embeddings = [r["embedding"] for r in records]
        self.labels     = [r["label"]     for r in records]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        x      = torch.from_numpy(self.embeddings[idx]).float()
        y      = torch.tensor(self.labels[idx], dtype=torch.long)
        pep_id = self.pep_ids[idx]
        return x, y, pep_id

    @property
    def embedding_dim(self) -> int:
        return self.embeddings[0].shape[0]

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights — pass to nn.CrossEntropyLoss(weight=…)."""
        counts  = np.bincount(self.labels, minlength=2).astype(np.float32)
        weights = 1.0 / (counts + 1e-8)
        return torch.from_numpy(weights / weights.sum())

    def summary(self) -> None:
        pos = sum(self.labels)
        neg = len(self.labels) - pos
        print(
            f"  Total    : {len(self.labels):>6}\n"
            f"  Active   : {pos:>6}  ({100 * pos / len(self.labels):.1f} %)\n"
            f"  Inactive : {neg:>6}  ({100 * neg / len(self.labels):.1f} %)\n"
            f"  Embed dim: {self.embedding_dim}"
        )


# ── Splitting ─────────────────────────────────────────────────────────────────

def _split_records(
    records: List[Dict],
    cfg: DictConfig,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Stratified split of flat record list → (train, val, test)."""
    rng = random.Random(cfg.split.random_seed)

    pos = [r for r in records if r["label"] == LABEL_POSITIVE]
    neg = [r for r in records if r["label"] == LABEL_NEGATIVE]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split_class(items: list):
        n      = len(items)
        n_test = max(1, int(n * cfg.split.test_size))
        n_val  = max(1, int((n - n_test) * cfg.split.val_size))
        return (
            items[n_test + n_val:],          # train
            items[n_test: n_test + n_val],   # val
            items[:n_test],                  # test
        )

    pos_tr, pos_va, pos_te = split_class(pos)
    neg_tr, neg_va, neg_te = split_class(neg)

    train = pos_tr + neg_tr
    val   = pos_va + neg_va
    test  = pos_te + neg_te

    rng.shuffle(train)   # interleave pos/neg for training
    return train, val, test


# ── Public API ────────────────────────────────────────────────────────────────

def build_datasets(cfg: DictConfig) -> Tuple[PeptideDataset, PeptideDataset, PeptideDataset]:
    """
    Load the embedding pickle and return (train_dataset, val_dataset, test_dataset).

    Raises FileNotFoundError if the pickle does not exist — run gen_embed(cfg) first.
    """
    pkl_path = Path(cfg.data.embeddings_file)
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Embedding pickle not found at '{pkl_path}'.\n"
            "Run gen_embed(cfg) first to generate it."
        )

    print(f"[build_datasets] Loading embeddings from '{pkl_path}' …")
    with open(pkl_path, "rb") as f:
        store: Dict[str, Dict] = pickle.load(f)

    records = list(store.values())   # [{"embedding": ..., "label": ...}, ...]
    print(f"[build_datasets] Loaded {len(records)} records.")

    train_recs, val_recs, test_recs = _split_records(records, cfg)

    train_ds = PeptideDataset(train_recs)
    val_ds   = PeptideDataset(val_recs)
    test_ds  = PeptideDataset(test_recs)

    print("\n── Dataset splits ───────────────────────────────────")
    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        print(f"\n{name}:")
        ds.summary()

    return train_ds, val_ds, test_ds


def build_dataloaders(
    cfg: DictConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) ready for model training."""
    train_ds, val_ds, test_ds = build_datasets(cfg)
    dl = cfg.dataloader

    train_loader = DataLoader(
        train_ds, batch_size=dl.batch_size,
        shuffle=dl.shuffle_train, num_workers=dl.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=dl.batch_size,
        shuffle=False, num_workers=dl.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=dl.batch_size,
        shuffle=False, num_workers=dl.num_workers,
    )
    return train_loader, val_loader, test_loader


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ESM2 embeddings and inspect the dataset.")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--export-csv", action="store_true",
        help="Export pos/neg TXT files to CSV as a side effect",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.export_csv:
        txt_to_csv(cfg.data.pos_file, str(Path(cfg.data.pos_file).with_suffix(".csv")))
        txt_to_csv(cfg.data.neg_file, str(Path(cfg.data.neg_file).with_suffix(".csv")))

    # Step 1 — embed (no-op if pickle already exists)
    gen_embed(cfg)

    # Step 2 — inspect
    train_ds, val_ds, test_ds = build_datasets(cfg)

    print("\n── Sample ───────────────────────────────────────────")
    x, y, pep_id = train_ds[0]
    print(f"  pep_id  : {pep_id}")
    print(f"  x shape : {x.shape}")
    print(f"  y       : {y.item()}  ({'active' if y.item() == 1 else 'inactive'})")
    print(f"  class weights: {train_ds.class_weights}")