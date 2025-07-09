import torch
from transformers import BertTokenizer, BertModel

class ProteinBERTEncoder(torch.nn.Module):
    def __init__(self, model_name="Rostlab/prot_bert", device="cpu", finetune=True, fc_dim=512, dropout=0.1):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.bert = BertModel.from_pretrained(model_name)
        self.bert.to(device)
        self.finetune = finetune
        self.device = device
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, fc_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(fc_dim, self.bert.config.hidden_size)
        )
        # Set requires_grad according to finetune flag
        for param in self.bert.parameters():
            param.requires_grad = finetune

    def forward(self, seqs):
        spaced = [" ".join(list(seq)) for seq in seqs]
        tokens = self.tokenizer(
            spaced,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        out = self.bert(**tokens)
        emb = out.last_hidden_state[:, 0, :]
        emb = self.dropout(emb)
        emb = self.fc(emb)
        return emb