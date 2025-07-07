import os
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import BertModel, BertTokenizer
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


PROTBERT_MODEL = "Rostlab/prot_bert"
tokenizer = BertTokenizer.from_pretrained(PROTBERT_MODEL, do_lower_case=False)
protbert = BertModel.from_pretrained(PROTBERT_MODEL)
protbert.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
protbert = protbert.to(device)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
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

    # Edge index and edge features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        # Bond features: bond type as int, conjugation, ring
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
        # Handle molecules with no bonds (e.g., single atom)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return x, edge_index, edge_attr

def seq_to_protbert_embedding(sequence):
    # ProtBERT expects spaces between amino acids
    seq = " ".join(list(sequence))
    ids = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
    ids = {k: v.to(device) for k, v in ids.items()}
    with torch.no_grad():
        out = protbert(**ids)
        # CLS token embedding: shape (1, hidden_size)
        emb = out.last_hidden_state[:, 0, :]  # [CLS] token
    return emb.squeeze(0).cpu()

def process_dataset(df, log_file):
    data_list = []
    skipped = 0
    with open(log_file, "w") as flog:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row["compound_iso_smiles"]
            seq = row["target_sequence"]
            pkd = row["pKd"]
            # Drug graph
            try:
                graph = smiles_to_graph(smiles)
                if graph is None:
                    raise ValueError("Invalid SMILES")
                x, edge_index, edge_attr = graph
            except Exception as e:
                flog.write(f"SMILES error at idx {idx}: {smiles} | {e}\n")
                skipped += 1
                continue
            # Protein embedding
            try:
                # Check for valid protein sequence
                if not isinstance(seq, str) or pd.isna(seq):
                    raise ValueError("Invalid or missing protein sequence")
                protein_emb = seq_to_protbert_embedding(seq)
            except Exception as e:
                flog.write(f"Protein error at idx {idx}: {str(seq)[:30]}... | {e}\n")
                skipped += 1
                continue
            # Label
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
    # Shuffle and split
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_train = int(0.8 * len(df))
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    return train_df, test_df

def main():
    os.makedirs("processed", exist_ok=True)

    # --- BindingDB ---
    print("Processing BindingDB...")
    bindingdb_path = os.path.join("data", "bindingdb.csv")
    bindingdb = pd.read_csv(bindingdb_path)
    train_df, test_df = split_bindingdb(bindingdb)

    print("Processing BindingDB train set...")
    train_data = process_dataset(train_df, "processed/bindingdb_train.log")
    torch.save(train_data, "processed/bindingdb_train.pt")

    print("Processing BindingDB test set...")
    test_data = process_dataset(test_df, "processed/bindingdb_test.log")
    torch.save(test_data, "processed/bindingdb_test.pt")

    # --- Davis ---
    print("Processing Davis...")
    davis_path = os.path.join("data", "davis.csv")
    davis = pd.read_csv(davis_path)
    davis_data = process_dataset(davis, "processed/davis.log")
    torch.save(davis_data, "processed/davis.pt")

    print("All datasets processed and saved in ./processed/")

if __name__ == "__main__":
    main()