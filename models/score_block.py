from models.axial_transformer import AxialTransformer
import torch.nn as nn

# Graphical Dual Axial Transformer (GDAT)
class ScoreBlockGDAT(nn.Module):
    """Axial Transformer Score Block"""
    def __init__(self, input_dim, N=238, T=21, hidden_dim=64, L_encoder=1, L_decoder=1, heads=2, dim_heads=None, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.axial_transformer = AxialTransformer(
        input_dim=self.input_dim,
        hidden_dim=hidden_dim,
        L_encoder=L_encoder,
        L_decoder=L_decoder,
        heads=heads,
        dim_heads=dim_heads,
        axial_pos_emb_shape=(N, T),
        dropout=dropout
        )

    def forward(self, x, N_mask=None, T_mask=None, **kwargs):
        return self.axial_transformer(x, N_mask=N_mask, T_mask=T_mask)


# Model selection function
def get_score_block(model_name, **kwargs):
    """
    Factory function to create score block models

    Available models:
    - 'GDAT': Graphical Dual Axial Transformer with dual-axis attention
    """
    model_classes = {
        'GDAT': ScoreBlockGDAT,
    }

    if model_name not in model_classes:
        available_models = list(model_classes.keys())
        raise ValueError(f"Unknown score block model: {model_name}. Available models: {available_models}")

    return model_classes[model_name](**kwargs)
