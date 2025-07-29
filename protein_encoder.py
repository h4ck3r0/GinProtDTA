import torch
import torch.nn as nn

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}  
VOCAB_SIZE = len(AMINO_ACIDS) + 1  

class ProteinBiLSTMEncoder(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=1024, num_layers=3, fc_dim=512, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, fc_dim)
        self.output_dim = fc_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, seqs):
      
        max_len = max(len(seq) for seq in seqs)
        idxs = []
        for seq in seqs:
            idx_seq = [AA_TO_IDX.get(aa, 0) for aa in seq]
            idxs.append(idx_seq + [0] * (max_len - len(idx_seq)))
        idxs = torch.tensor(idxs, dtype=torch.long, device=next(self.parameters()).device)
        emb = self.embedding(idxs)
        out, _ = self.lstm(emb)
        mask = (idxs != 0).float().unsqueeze(-1)
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        normed = self.layer_norm(pooled)
        fc_out = self.fc(normed)
        return self.dropout(fc_out)
