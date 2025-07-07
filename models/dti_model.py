import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch_geometric.nn import GINEConv, BatchNorm, global_mean_pool, global_max_pool, global_add_pool, GlobalAttention, Set2Set

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            nn.init.xavier_uniform_(layers[-1].weight)
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class MLPConv(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, num_layers=2, dropout=0.2, shared_weights=None):
        super().__init__()
        if shared_weights is not None:
            self.mlp = shared_weights
        else:
            hidden_dim = hidden_dim if hidden_dim is not None else out_dim
            self.mlp = MLP(in_dim, hidden_dim, out_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, x):
        return self.mlp(x)

class DrugEncoder(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2,
        residual=True,
        pooling='mean',
        share_gine_mlps=True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.pooling = pooling

        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.shared_mlps = {}

        for i in range(num_layers):
            in_dim = node_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            shared = None
            if share_gine_mlps and in_dim == out_dim:
                key = f"{in_dim}_{out_dim}"
                if key not in self.shared_mlps:
                    self.shared_mlps[key] = MLPConv(in_dim, out_dim, hidden_dim, num_layers=2, dropout=dropout)
                shared = self.shared_mlps[key].mlp
            mlp_conv = MLPConv(in_dim, out_dim, hidden_dim, num_layers=2, dropout=dropout, shared_weights=shared)
            conv = GINEConv(mlp_conv, edge_dim=edge_dim)
            self.gnn_layers.append(conv)
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        if pooling == 'attention':
            self.pool = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_dim, 1)
                )
            )
        elif pooling == 'set2set':
            self.pool = Set2Set(hidden_dim, processing_steps=2)
        else:
            self.pool = None

        # For pooling benchmarking
        self.pooling_fns = {
            'mean': global_mean_pool,
            'max': global_max_pool,
            'sum': global_add_pool,
            'attention': lambda h, batch: self.pool(h, batch),
            'set2set': lambda h, batch: self.pool(h, batch)
        }

    def forward(self, x, edge_index, edge_attr, batch, profile_pooling=False):
        h = x
        for i in range(self.num_layers):
            h_in = h
            h = self.gnn_layers[i](h, edge_index, edge_attr)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual and h_in.shape == h.shape:
                h = h + h_in

        if profile_pooling:
            pool_times = {}
            for name, fn in self.pooling_fns.items():
                start = time.time()
                _ = fn(h, batch)
                pool_times[name] = time.time() - start
            slowest = max(pool_times, key=pool_times.get)
            if pool_times.get('set2set', 0) > 0.1:
                print(f"Warning: Set2Set pooling is slow ({pool_times['set2set']:.3f}s). Consider using mean/max/sum for real-time.")
            print("Pooling times (s):", pool_times)

        if self.pooling == 'mean':
            hg = global_mean_pool(h, batch)
        elif self.pooling == 'max':
            hg = global_max_pool(h, batch)
        elif self.pooling == 'sum':
            hg = global_add_pool(h, batch)
        elif self.pooling == 'attention':
            hg = self.pool(h, batch)
        elif self.pooling == 'set2set':
            hg = self.pool(h, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return hg

class DTIModel(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        protein_dim=1024,
        gnn_hidden_dim=128,
        gnn_layers=3,
        gnn_dropout=0.2,
        gnn_residual=True,
        gnn_pooling='mean',
        mlp_hidden_dim=128,
        mlp_layers=2,
        mlp_dropout=0.2,
        share_gine_mlps=True
    ):
        super().__init__()
        self.drug_encoder = DrugEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            dropout=gnn_dropout,
            residual=gnn_residual,
            pooling=gnn_pooling,
            share_gine_mlps=share_gine_mlps
        )
        fusion_dim = gnn_hidden_dim if gnn_pooling != 'set2set' else 2 * gnn_hidden_dim
        self.reg_head = MLP(
            input_dim=fusion_dim + protein_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=1,
            num_layers=mlp_layers,
            dropout=mlp_dropout
        )

    def forward(self, batch, protein_emb, profile_pooling=False):
        drug_repr = self.drug_encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch, profile_pooling=profile_pooling)
        fusion = torch.cat([drug_repr, protein_emb], dim=1)
        out = self.reg_head(fusion)
        return out.view(-1)

    def optimize_for_inference(self, use_channels_last=True, use_fp16=False, compile_model=True):
        if use_channels_last:
            self.to(memory_format=torch.channels_last)
        if use_fp16:
            # Warn if BatchNorm is present
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    print("Warning: BatchNorm1d detected. FP16 inference may be unstable. Consider replacing with LayerNorm.")
            self.half()
        if compile_model and hasattr(torch, "compile"):
            self = torch.compile(self, dynamic=True)
        return self