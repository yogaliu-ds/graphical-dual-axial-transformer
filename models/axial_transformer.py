import torch
from torch import nn

# --- Utility blocks ---
class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim_heads = dim // heads if dim_heads is None else dim_heads
        dim_hidden = self.heads * self.dim_heads
        self.to_qkv = nn.Linear(dim, dim_hidden * 3, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, L, self.heads, self.dim_heads).transpose(1, 2) for t in qkv]
        dots = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_heads ** 0.5)
        if mask is not None:
            dots = dots.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.to_out(out)

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()
        N, T = shape
        self.pos_emb_N = nn.Parameter(torch.randn(1, N, 1, dim))
        self.pos_emb_T = nn.Parameter(torch.randn(1, 1, T, dim))
    def forward(self, x):
        return x + self.pos_emb_N + self.pos_emb_T

class GeneralAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_heads=None, attention_dim=2, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.attention_dim = attention_dim
        self.attn = SelfAttention(dim, heads, dim_heads, dropout)
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x_norm = self.layer_norm(x)
        B, N, T, D = x_norm.shape

        mask_flat = None
        if self.attention_dim == 1:  # N-axis attention
            x_perm = x_norm.permute(0, 2, 1, 3)
            x_flat = x_perm.reshape(B * T, N, D)
            if mask is not None:
                mask_flat = mask.unsqueeze(0).expand(B * T, -1, -1).unsqueeze(1)
                mask_flat = mask_flat.to(x_flat.device)
            else:
                mask_flat = None
            x_attn = self.attn(x_flat, mask=mask_flat)
            x_attn = x_attn.reshape(B, T, N, D)
            x_out = x_attn.permute(0, 2, 1, 3)

        elif self.attention_dim == 2:  # T-axis attention
            x_flat = x_norm.reshape(B * N, T, D)
            if mask is not None:
                mask_flat = mask.unsqueeze(1)
                mask_flat = mask_flat.to(x_flat.device)
            else:
                mask_flat = None
            x_attn = self.attn(x_flat, mask=mask_flat)
            x_out = x_attn.reshape(B, N, T, D)

        else:
            raise ValueError("attention_dim must be 1 (for N-axis) or 2 (for T-axis)")
        
        x_out = self.dense(x_out)
        x_out = self.dropout(x_out)
        return x + x_out

def shift_right(x):
    return torch.roll(x, shifts=1, dims=1)

# --- Encoder/Decoder ---
class Encoder(nn.Module):
    """Multi-layer N-axis attention block: h -> [T-axis -> N-axis] x L_encoder (supports N_mask)"""
    def __init__(self, hidden_dim, L_encoder=2, heads=1, dim_heads=None, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                # GeneralAttentionBlock(hidden_dim, heads, dim_heads, attention_dim=2, dropout=dropout),  # T-axis
                GeneralAttentionBlock(hidden_dim, heads, dim_heads, attention_dim=1, dropout=dropout)   # N-axis
            ) for _ in range(L_encoder)
        ])
    def forward(self, h, N_mask=None):
        for layer in self.layers:
            h = layer[0](h, mask=N_mask)  # pass N_mask to N-axis attention
        return h

class Decoder(nn.Module):
    """Multi-layer masked T-axis attention block: h -> [masked T-axis] x L_decoder (uses only T_mask)"""
    def __init__(self, hidden_dim, L_decoder=2, heads=1, dim_heads=None, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GeneralAttentionBlock(hidden_dim, heads, dim_heads, attention_dim=2, dropout=dropout)
            for _ in range(L_decoder)
        ])
    def forward(self, h, T_mask=None):
        for layer in self.layers:
            h = layer(h, mask=T_mask)
        return h

# --- Main module ---
class AxialTransformer(nn.Module):
    """Main model: handles embedding, positional embedding, shift, and encoder/decoder composition"""
    # def __init__(self, 
    #              input_dim, 
    #              hidden_dim=64, L_encoder=2, L_decoder=4, heads=8, dim_heads=None, axial_pos_emb_shape=None, dropout=0.1):
    def __init__(self, 
                 input_dim, 
                 hidden_dim=64, 
                 L_encoder=1, 
                 L_decoder=1, 
                 heads=2, 
                 dim_heads=None, 
                 axial_pos_emb_shape=None, 
                 dropout=0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        )
        self.pos_emb = AxialPositionalEmbedding(hidden_dim, axial_pos_emb_shape)
        self.encoder = Encoder(hidden_dim, L_encoder, heads, dim_heads, dropout)
        self.decoder = Decoder(hidden_dim, L_decoder, heads, dim_heads, dropout)
        self.output_proj = nn.Linear(hidden_dim, 1)
        self.dropout_layer = nn.Dropout(dropout)
    def forward(self, x, N_mask=None, T_mask=None):
        B, N, T, E = x.shape
        x_reshaped = x.view(B * N, E, T)
        h = self.embedding(x_reshaped)
        h = h.view(B, N, T, -1)
        u = h + self.pos_emb(h)
        u = self.encoder(u, N_mask=N_mask)  # encoder uses N_mask
        # h = u + shift_right(h) + self.pos_emb(h)
        h = u
        
        # --- Automatically generate the causal mask (T-axis) ---
        if T_mask is None:
            device = x.device
            causal_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
            causal_mask = causal_mask.expand(B, N, T, T).reshape(B * N, T, T)  # [B*N, T, T]
        else:
            causal_mask = T_mask
        h = self.decoder(h, T_mask=causal_mask)  # decoder uses T_mask
        h_agg = h[:, :, -1, :]  # [B, N, D]
        h_agg = self.dropout_layer(h_agg)
        scores = self.output_proj(h_agg) # [B, N, 1]
        return scores.squeeze(-1)  # [B, N]




