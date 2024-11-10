import torch
import torch.nn as nn

class SliceTransformer(nn.Module):
    def __init__(self, emb_dim=256, num_layers=4, num_heads=8, dropout=0.1):
        super(SliceTransformer, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, num_slices, emb_dim)
        """
        x = x.permute(1, 0, 2) # (num_slices, B, emb_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2) # (B, num_slices, emb_dim)
        
        return x