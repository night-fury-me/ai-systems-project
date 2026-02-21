import torch # type: ignore
import torch.nn as nn # type: ignore

class FormulaTransformer(nn.Module):
    def __init__(self, vocab_size, formula_max_length=200, d_model=256, n_heads=4, ff_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.formula_max_length = formula_max_length
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(self.formula_max_length + 1, d_model)  # self.formula_max_length + CLS #positions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # (1, 1, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size = x.size(0)
        device = next(self.parameters()).device
        
        # 1. Embed input tokens
        x = x.long().to(device)
        x_embedded = self.embedding(x)  # (batch_size, seq_len, d_model)
        
        # 2. Add CLS token (learnable parameter)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x_embedded], dim=1)  # (batch, seq_len+1, d_model)
        
        # 3. Add positional embeddings
        positions = torch.arange(x.size(1), device=device).long()
        x = x + self.pos_encoder(positions)
        x = self.dropout(x)
        # 4. Process through transformer
        x = self.transformer(x)
        
        return x[:, 0, :]  # Return CLS token's representation

class DocumentTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=4, ff_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.pos_encoder = nn.Embedding(10 + 1, d_model)  # Max formulas=10 and + 1 for CLS token
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, N, d_model)
        batch_size = x.size(0)
        # print(f"{x.shape}")
        device = next(self.parameters()).device
        x = x.to(device)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        positions = torch.arange(x.size(1), device=device).long()
        # print(f"{positions=}")
        x = x + self.pos_encoder(positions)
        x = self.dropout(x)
        x = self.transformer(x)
        return x[:, 0, :]  # Document embedding from [CLS]

class MathDocClassifier(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.form_config    = configs["FormulaTransformer"]
        self.doc_config     = configs["DocumentTransformer"]
        
        self.doc_embed_size = self.doc_config["d_model"]
        self.n_classes      = configs["n_classes"]
        self.vocab_size     = configs["vocab_size"]
        self.dropout        = configs["classifier"]["dropout"]
        self.formula_max_length = configs["formula_max_length"]

        self.formula_encoder = FormulaTransformer(
            self.vocab_size, 
            self.formula_max_length,
            self.form_config["d_model"],
            self.form_config["n_heads"],
            self.form_config["ff_dim"],
            self.form_config["num_layers"],
            self.form_config["dropout"]
        )

        self.doc_encoder = DocumentTransformer(
            self.doc_config["d_model"],
            self.doc_config["n_heads"],
            self.doc_config["ff_dim"],
            self.doc_config["num_layers"],
            self.doc_config["dropout"]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.doc_embed_size),
            nn.Linear(self.doc_embed_size, self.doc_embed_size), 
            nn.GELU(), 
            nn.Dropout(self.dropout),
            nn.Linear(self.doc_embed_size, self.n_classes)
        )
        
    def forward(self, formulas):
        # formulas: (batch_size, N, L)
        batch_size, N, L = formulas.shape
        device = next(self.parameters()).device 
        formulas = formulas.to(device)

        formula_embeds = []
        for i in range(N):
            embed = self.formula_encoder(formulas[:, i, :])
            formula_embeds.append(embed)
        formula_embeds = torch.stack(formula_embeds, dim=1)
        doc_embed = self.doc_encoder(formula_embeds)
        return self.classifier(doc_embed)