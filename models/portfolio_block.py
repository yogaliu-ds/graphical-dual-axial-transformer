import torch
import torch.nn as nn
import torch.nn.functional as F

class PortfolioBlockSoftmax(nn.Module):
    """Constraint 1: Long-only (softmax)"""
    def __init__(self):
        super().__init__()
    def forward(self, scores):
        return F.softmax(scores, dim=-1)

class PortfolioBlockAbsMax(nn.Module):
    """Constraint 2: Maximum and Abs(w)=1"""
    def __init__(self, u=1.0):
        super().__init__()
        self.u = u
    def forward(self, scores):
        return constrained_weight(scores, self.u)
    
def constrained_weight(scores: torch.Tensor, u: float = 1.0) -> torch.Tensor:
    """
    Constraint 2: Maximum and Abs(w)=1
    Args:
        scores: [B, N] tensor of scores (can be positive or negative)
        u: upper bound per asset (e.g., 0.1 or 1.0)
    Returns:
        weights: [B, N] tensor of constrained weights, satisfying:
                 - sum_i |w_i| = 1
                 - |w_i| <= u
    """
    B, N = scores.shape
    a = (1 - u) / (N * u - 1)
    abs_scores = torch.abs(scores)
    phi = (a + 1) / (1 + torch.exp(-abs_scores))  # shape [B, N]
    signed_phi = torch.sign(scores) * phi
    weights = signed_phi / torch.sum(phi, dim=1, keepdim=True)
    return weights

# # Corrected version
# class PortfolioBlockAbsMax(nn.Module):
#     """Constraint: L1=1 and |w_i| <= u"""
#     def __init__(self, u=0.1):
#         super().__init__()
#         self.u = u

#     def forward(self, scores):
#         return constrained_weight(scores, self.u)
# def constrained_weight(scores: torch.Tensor, u: float = 1.0) -> torch.Tensor:
#     """
#     Project scores onto feasible set:
#         sum |w_i| = 1
#         |w_i| <= u
#     """
#     B, N = scores.shape
#     sign = torch.sign(scores)
#     abs_scores = torch.abs(scores) + 1e-8  # avoid all zero

#     # First normalize so that sum=1
#     w = abs_scores / abs_scores.sum(dim=1, keepdim=True)

#     # Project onto the box: |w_i| <= u
#     w = torch.clamp(w, 0, u)

#     # If sum < 1, renormalize
#     s = w.sum(dim=1, keepdim=True)
#     # Ensure sum=1
#     w = w / (s + 1e-8)

#     # Restore the sign
#     return sign * w


class PortfolioBlockGeneralizedSoftmax(nn.Module):
    """Constraint 3: Generalized Softmax"""
    def __init__(self):
        super().__init__()
        self.generalized_softmax = GeneralizedSoftmaxTransform()
    def forward(self, scores):
        return self.generalized_softmax(scores)

class GeneralizedSoftmaxTransform(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        B, N = x.shape
        exp_rest = torch.exp(x[:, 1:])  # [B, N-1]
        denom = 1 + exp_rest.sum(dim=-1, keepdim=True)  # [B, 1]
        q = torch.empty_like(x)
        q[:, 0] = 1. / denom.squeeze(-1)              # q1
        q[:, 1:] = exp_rest / denom                   # q2 ... qN
        p = (N + 1) * q - 1
        return p
    


class PortfolioBlockLongShort(nn.Module):
    """Constraint 4: separate long & short softmax, leverage control"""
    def __init__(self, leverage=1.0):
        super().__init__()
        self.leverage = leverage  # controls total leverage, e.g., 1.5 means 150%
    def forward(self, scores):
        longs = F.softmax(scores, dim=-1)   # long side
        shorts = F.softmax(-scores, dim=-1) # short side
        weights = (longs - shorts) * (self.leverage / 2)
        return weights
