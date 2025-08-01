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

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    # GraphDTA-style atom features (one-hot + aromaticity)
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
            'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()]
    )

def bond_features(bond):
    bt = bond.GetBondType()
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_of_k_encoding_unk(bt, bond_types)
    return np.array(
        bond_type_enc +
        [bond.GetIsConjugated()] +
        [bond.IsInRing()]
    )

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_features(atom))
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        bf = bond_features(bond)
        edge_attr.append(bf)
        edge_attr.append(bf)
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
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

def process_dataset(df, protbert_cache, log_file, label_col="pKd"):
    data_list = []
    skipped = 0
    with open(log_file, "w") as flog:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row["compound_iso_smiles"]
            seq = row["target_sequence"]
            label = row[label_col]
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
                if pd.isna(label) or not np.isfinite(label):
                    flog.write(f"Label error at idx {idx}: {label} | Not finite\n")
                    skipped += 1
                    continue
                label = float(label)
                y = torch.tensor([label], dtype=torch.float)
            except Exception as e:
                flog.write(f"Label error at idx {idx}: {label} | {e}\n")
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
    bindingdb = pd.read_csv(os.path.join("data", "bindingdb_highconf_pKd_sample.csv"))
    davis = pd.read_csv(os.path.join("data", "davis.csv"))
    train_df, test_df = split_bindingdb(bindingdb)
    unique_seqs = collect_unique_sequences([train_df, test_df, davis])
    protbert_cache_exists = os.path.exists("processed/protbert_cache.pt")
    if protbert_cache_exists:
        protbert_cache = load_protbert_cache()
    else:
        protbert_cache = {}

    uncached_seqs = [seq for seq in unique_seqs if seq not in protbert_cache]
    if uncached_seqs:
        print(f"Computing embeddings for {len(uncached_seqs)} new protein sequences...")
        new_embs = batch_protbert_embeddings(uncached_seqs, batch_size=BATCH_SIZE)
        protbert_cache.update(new_embs)
        save_protbert_cache(protbert_cache)
    else:
        print("All protein embeddings are already cached.")

    if os.path.exists("processed/bindingdb_train.pt"):
        print("bindingdb_train.pt already exists, skipping.")
    else:
        print("Processing BindingDB train set...")
        train_data = process_dataset(train_df, protbert_cache, "processed/bindingdb_train.log", label_col="affinity")
        torch.save(train_data, "processed/bindingdb_train.pt")

    if os.path.exists("processed/bindingdb_test.pt"):
        print("bindingdb_test.pt already exists, skipping.")
    else:
        print("Processing BindingDB test set...")
        test_data = process_dataset(test_df, protbert_cache, "processed/bindingdb_test.log", label_col="affinity")
        torch.save(test_data, "processed/bindingdb_test.pt")

    if os.path.exists("processed/davis.pt"):
        print("davis.pt already exists, skipping.")
    else:
        print("Processing Davis...")
        davis_data = process_dataset(davis, protbert_cache, "processed/davis.log", label_col="affinity")
        torch.save(davis_data, "processed/davis.pt")

    print("All datasets processed and saved in ./processed/")

if __name__ == "__main__":
    main()
