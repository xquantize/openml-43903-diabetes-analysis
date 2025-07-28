import torch
import torch.nn as nn


class TabularTransformer(nn.Module):
    def __init__(self, cardinalities, n_numeric, cfg):
        super().__init__()
        d_model   = cfg['model']['d_model']
        n_heads   = cfg['model']['n_heads']
        n_layers  = cfg['model']['n_layers']
        dim_ff    = cfg['model']['dim_ff']
        dropout   = cfg['model']['dropout']

        self.cat_embed = nn.ModuleDict({
            col: nn.Embedding(card, d_model)
            for col, card in cardinalities.items()
        })

        self.num_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_numeric)])

        self.seq_len = len(cardinalities) + n_numeric
        self.pos_emb = nn.Parameter(torch.randn(self.seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )


    def forward(self, x_cat, x_num):
        """
        x_cat: (batch, n_categorical) tensor of ints
        x_num: (batch, n_numeric)   tensor of floats
        """
        tokens = []

        for i, col in enumerate(self.cat_embed):
            tokens.append(self.cat_embed[col](x_cat[:, i]))

        for j, proj in enumerate(self.num_proj):
            tokens.append(proj(x_num[:, j].unsqueeze(1)))

        x = torch.stack(tokens, dim=0)
        x = x + self.pos_emb.unsqueeze(1)
        x = self.transformer(x)

        pooled = x.mean(dim=0)  # (batch, d_model)

        logits = self.classifier(pooled).squeeze(1)

        return logits
