import os
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
from rdkit import Chem
from transformers import BertModel, BertTokenizer
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROTBERT_MODEL = "Rostlab/prot_bert"
CACHE_PATH = "processed/protbert_cache.pt"
BATCH_SIZE = 16

tokenizer = BertTokenizer.from_pretrained(PROTBERT_MODEL, do_lower_case=False)
protbert = BertModel.from_pretrained(PROTBERT_MODEL)
protbert.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
protbert = protbert.to(device)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            atom.GetImplicitValence(),
            int(atom.GetIsAromatic()),
        ])
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append([
            int(bond.GetBondTypeAsDouble()),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ])
        edge_attr.append([
            int(bond.GetBondTypeAsDouble()),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ])
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return x, edge_index, edge_attr

def collect_unique_sequences(dfs):
    seq_set = set()
    for df in dfs:
        for seq in df["target_sequence"]:
            if isinstance(seq, str) and pd.notna(seq):
                seq_set.add(seq)
    return list(seq_set)

def batch_protbert_embeddings(sequences, batch_size=BATCH_SIZE):
    embeddings = {}
    if not isinstance(sequences, list) or len(sequences) == 0:
        return embeddings
    total = len(sequences)
    for i in tqdm(range(0, total, batch_size), desc="ProtBERT batching"):
        batch_seqs = sequences[i:i+batch_size]
        if not batch_seqs:
            continue
        batch_inputs = []
        batch_keys = []
        for seq in batch_seqs:
            seq_spaced = " ".join(list(seq))
            batch_inputs.append(seq_spaced)
            batch_keys.append(seq)
        ids = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        ids = {k: v.to(device) for k, v in ids.items()}
        with torch.no_grad():
            out = protbert(**ids)
            batch_emb = out.last_hidden_state[:, 0, :].cpu()
        for key, emb in zip(batch_keys, batch_emb):
            embeddings[key] = emb
    return embeddings

def load_protbert_cache():
    if os.path.exists(CACHE_PATH):
        cache = torch.load(CACHE_PATH)
        print(f"Loaded ProtBERT cache with {len(cache)} entries.")
        return cache
    else:
        return {}

def save_protbert_cache(cache):
    torch.save(cache, CACHE_PATH)
    print(f"Saved ProtBERT cache with {len(cache)} entries.")

def process_dataset(df, protbert_cache, log_file):
    data_list = []
    skipped = 0
    with open(log_file, "w") as flog:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row["compound_iso_smiles"]
            seq = row["target_sequence"]
            pkd = row["pKd"]
            try:
                graph = smiles_to_graph(smiles)
                if graph is None:
                    raise ValueError("Invalid SMILES")
                x, edge_index, edge_attr = graph
            except Exception as e:
                flog.write(f"SMILES error at idx {idx}: {smiles} | {e}\n")
                skipped += 1
                continue
            try:
                if not isinstance(seq, str) or pd.isna(seq):
                    raise ValueError("Invalid or missing protein sequence")
                if seq not in protbert_cache:
                    raise ValueError("Protein embedding not found in cache")
                protein_emb = protbert_cache[seq]
            except Exception as e:
                flog.write(f"Protein error at idx {idx}: {str(seq)[:30]}... | {e}\n")
                skipped += 1
                continue
            try:
                y = torch.tensor([float(pkd)], dtype=torch.float)
            except Exception as e:
                flog.write(f"pKd error at idx {idx}: {pkd} | {e}\n")
                skipped += 1
                continue
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                protein=protein_emb,
                y=y
            )
            data_list.append(data)
    print(f"Skipped {skipped} samples due to errors. See {log_file} for details.")
    return data_list

def split_bindingdb(df):
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_train = int(0.8 * len(df))
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    return train_df, test_df

def main():
    os.makedirs("processed", exist_ok=True)
    # Load datasets
    bindingdb = pd.read_csv(os.path.join("data", "bindingdb.csv"))
    davis = pd.read_csv(os.path.join("data", "davis.csv"))
    train_df, test_df = split_bindingdb(bindingdb)
    # Collect unique sequences
    unique_seqs = collect_unique_sequences([train_df, test_df, davis])
    protbert_cache = load_protbert_cache()
    uncached_seqs = [seq for seq in unique_seqs if seq not in protbert_cache]
    if uncached_seqs:
        print(f"Computing embeddings for {len(uncached_seqs)} new protein sequences...")
        new_embs = batch_protbert_embeddings(uncached_seqs, batch_size=BATCH_SIZE)
        protbert_cache.update(new_embs)
        save_protbert_cache(protbert_cache)
    # Process datasets
    print("Processing BindingDB train set...")
    train_data = process_dataset(train_df, protbert_cache, "processed/bindingdb_train.log")
    torch.save(train_data, "processed/bindingdb_train.pt")
    print("Processing BindingDB test set...")
    test_data = process_dataset(test_df, protbert_cache, "processed/bindingdb_test.log")
    torch.save(test_data, "processed/bindingdb_test.pt")
    print("Processing Davis...")
    davis_data = process_dataset(davis, protbert_cache, "processed/davis.log")
    torch.save(davis_data, "processed/davis.pt")
    print("All datasets processed and saved in ./processed/")

if __name__ == "__main__":
    main()