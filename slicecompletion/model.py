import torch
import torch.nn as nn
import torch.functional as F
from slicecompletion.encoder import SliceEncoder
from slicecompletion.transformer import SliceTransformer

        

class SliceCompletion(nn.Module):
    def __init__(self, emb_dim=256, code_dim=128, num_layers=4, num_heads=8, dropout=0.1): # make these part of config (OmegaConf)
        super(SliceCompletion, self).__init__()
        self.emb_dim = emb_dim
        self.code_dim = code_dim
        
        self.slice_encoder = SliceEncoder(emb_dim=emb_dim)
        self.slice_transformer = SliceTransformer(num_layers=num_layers, num_heads=num_heads, dropout=dropout)
        self.fc_out = nn.Linear(emb_dim, code_dim)

    def forward(self, slices):
        slice_embeddings = [self.slice_encoder(s) for s in slices]  # each (batch_size, emb_dim)
        slice_seq = torch.stack(slice_embeddings, dim=1)  # (batch_size, num_slices, emb_dim)
        enc_seq = self.slice_transformer(slice_seq) # (batch_size, num_slices, out_emb_dim)
        agg_enc = enc_seq.mean(dim=1)  # (batch_size, emb_dim)
        output = self.fc_out(agg_enc)
        
        return output
        
        
