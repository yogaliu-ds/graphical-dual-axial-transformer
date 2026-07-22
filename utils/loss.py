import torch
import torch.nn as nn
import inspect
from typing import Type

# --- Constants ---
TRADING_DAYS_PER_YEAR = 252
EPSILON = 1e-6

# --- Base Classes for Code Reuse ---

class _BaseLoss(nn.Module):
    """Base class for portfolio losses that handles common parameters."""
    def __init__(self, periods_per_year=TRADING_DAYS_PER_YEAR, **kwargs):
        super().__init__()
        self.periods_per_year = periods_per_year

    def _calculate_objective_return(self, y_returns, weights, mode):
        """
        Calculates the annualized return for the objective function based on the mode.
        It assumes y_returns is always provided.
        """
        if y_returns is None:
            raise ValueError("y_returns must be provided to calculate the objective return.")

        # Calculates annualized return with mode-dependent logic (t+1 vs horizon avg)
        annualized_return = self._get_annualized_return(y_returns, weights, mode)
        return annualized_return


    def _get_annualized_return(self, returns_data, weights, mode='eval'):
        """
        Calculates annualized return from a simple returns tensor.
        The returns_data can be historical (x) or future (y).
        Shape: [B, N, T] or [B, N, H].
        In 'train' mode, uses only the first step of the horizon (t+1).
        In 'eval' mode, uses the average of the entire horizon.
        """
        if mode == 'train':
            # Use only the first step of the horizon, shape [B, N]
            returns_at_t1 = returns_data[:, :, 0]
            # Calculate portfolio return for that single step, shape [B]
            portfolio_daily_return = torch.sum(returns_at_t1 * weights, dim=1)
        else:  # 'eval' mode or if mode is not specified
            # Use the full horizon, shape [B, N, H]
            # Calculate returns for each step in the horizon, shape [B, H]
            portfolio_returns_over_period = torch.sum(returns_data * weights.unsqueeze(2), dim=1)
            # Average the returns over the horizon, shape [B]
            portfolio_daily_return = portfolio_returns_over_period.mean(dim=1)

        annualized_return = portfolio_daily_return * self.periods_per_year
        return annualized_return

class _BaseCovarianceLoss(_BaseLoss):
    """Base class for losses using a shrinked covariance matrix from long-term historical returns."""
    def __init__(self, shrinkage=0.7, burn_in_period=0, **kwargs):
        super().__init__(**kwargs)
        self.shrinkage = shrinkage
        self.burn_in_period = burn_in_period

    def _calculate_covariance_variance(self, x_cov, weights):
        """Calculates annualized portfolio variance via the covariance matrix method."""
        if x_cov is None:
            raise ValueError("x_cov is required for covariance-based loss calculation.")
        
        B, N, T = x_cov.shape

        # Covariance estimation (shrinked) from long-term history
        mean = x_cov.mean(dim=2, keepdim=True)
        X_centered = x_cov - mean
        cov_matrix = torch.matmul(X_centered, X_centered.transpose(1, 2)) / (T - 1)
        
        # Shrinkage
        trace = torch.diagonal(cov_matrix, dim1=1, dim2=2).sum(dim=1, keepdim=True)
        mu = trace / N
        I = torch.eye(N, device=x_cov.device).unsqueeze(0)
        shrinked_cov = (1 - self.shrinkage) * cov_matrix + self.shrinkage * mu.view(B, 1, 1) * I
        
        annualized_shrinked_cov = shrinked_cov * self.periods_per_year

        w = weights.unsqueeze(1)
        portfolio_var = torch.bmm(w, annualized_shrinked_cov).bmm(w.transpose(1, 2)).squeeze()

        return portfolio_var

class _BaseHistoricalSimLoss(_BaseLoss):
    """Base class for losses using historical simulation from long-term historical returns."""
    def __init__(self, burn_in_period=0, **kwargs):
        super().__init__(**kwargs)
        self.burn_in_period = burn_in_period

    def _get_historical_portfolio_returns(self, returns_data, weights, x_cov=None):
        """
        Calculates a simulated portfolio return series from the provided returns data.
        Prioritizes x_cov if provided for risk calculation, otherwise uses returns_data.
        The returns_data can be historical (x) or future (y).
        """
        data_for_sim = x_cov if x_cov is not None else returns_data
        
        # data_for_sim is [B, N, T], weights is [B, N] -> result is [B, T]
        historical_portfolio_returns = torch.sum(data_for_sim * weights.unsqueeze(2), dim=1)
        return historical_portfolio_returns

# --- Refactored Loss Implementations ---

class Cov_MeanVarianceLoss(_BaseCovarianceLoss):
    """Calculates loss based on mean-variance optimization, using a shrinked covariance matrix."""
    def __init__(self, risk_aversion=1.0, **kwargs):
        super().__init__(**kwargs)
        self.risk_aversion = risk_aversion

    def forward(self, weights, batch_x, y_returns=None, x_cov=None, mode='train'):
        # Risk is always calculated from historical data x_cov
        portfolio_var = self._calculate_covariance_variance(x_cov, weights)
        
        # Return is now calculated via the base class helper method
        annualized_return = self._calculate_objective_return(y_returns, weights, mode)
        
        loss = -(annualized_return - self.risk_aversion * portfolio_var)
        return loss.mean()

class Cov_SharpeRatioLoss(_BaseCovarianceLoss):
    """Calculates loss based on the Sharpe Ratio, using a shrinked covariance matrix."""
    def __init__(self, risk_free_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.risk_free_rate = risk_free_rate

    def forward(self, weights, batch_x, y_returns=None, x_cov=None, mode='train'):
        # Risk is always calculated from historical data x_cov
        portfolio_var = self._calculate_covariance_variance(x_cov, weights)
        annualized_std = torch.sqrt(portfolio_var + EPSILON)
        
        # Return is now calculated via the base class helper method
        annualized_return = self._calculate_objective_return(y_returns, weights, mode)

        excess_return = annualized_return - self.risk_free_rate
        sharpe = excess_return / annualized_std
        return -sharpe.mean()

class HistoricalVariance_SharpeRatioLoss(_BaseHistoricalSimLoss):
    """Calculates loss based on the Sharpe Ratio, using historical simulation on long-term data."""
    def __init__(self, risk_free_rate=0.0, mode='softplus', **kwargs):
        super().__init__(**kwargs)
        self.risk_free_rate = risk_free_rate
        self.mode = mode

    def forward(self, weights, batch_x, y_returns=None, x_cov=None, mode='train'):
        # Extract the returns from the feature tensor batch_x. Returns are the first feature.
        historical_returns = batch_x[:, :, :, 0]

        # Risk is always calculated from historical data, using x_cov if provided, else historical_returns
        historical_portfolio_returns_risk = self._get_historical_portfolio_returns(historical_returns, weights, x_cov=x_cov)
        portfolio_std = historical_portfolio_returns_risk.std(dim=1, unbiased=True)
        annualized_std = portfolio_std * torch.sqrt(torch.tensor(self.periods_per_year, device=portfolio_std.device))
        
        # Return is now calculated via the base class helper method
        annualized_return = self._calculate_objective_return(y_returns, weights, mode)
        
        excess_return = annualized_return - self.risk_free_rate
        sharpe = excess_return / (annualized_std + EPSILON)
        
        if self.mode == 'softplus':
            return -torch.nn.functional.softplus(sharpe, beta=1).mean()
        elif self.mode == 'linear':
            return -sharpe.mean()
        else:
            raise ValueError(f"Unknown HistoricalVariance_SharpeRatioLoss mode: {self.mode}")

class ExpectedReturnLoss(_BaseLoss):
    """Calculates loss to maximize returns, with an optional L2 penalty on weights."""
    def __init__(self, penalty_weight=0.01, **kwargs):
        super().__init__(**kwargs)
        self.penalty_weight = penalty_weight

    def forward(self, weights, batch_x, y_returns=None, mode='train'):
        # Return is now calculated via the base class helper method
        annualized_return = self._calculate_objective_return(y_returns, weights, mode)
        l2_penalty = self.penalty_weight * torch.sum(weights ** 2, dim=1)
        loss = -annualized_return + l2_penalty
        return loss.mean()

class HistoricalVariance_MeanVarianceLoss(_BaseHistoricalSimLoss):
    """Calculates loss based on mean-variance optimization, using historical simulation."""
    def __init__(self, risk_aversion=1.0, **kwargs):
        super().__init__(**kwargs)
        self.risk_aversion = risk_aversion

    def forward(self, weights, batch_x, y_returns=None, x_cov=None, mode='train'):
        # Extract the returns from the feature tensor batch_x. Returns are the first feature.
        historical_returns = batch_x[:, :, :, 0]
        
        # Risk is always calculated from historical data, using x_cov if provided, else historical_returns
        historical_portfolio_returns_risk = self._get_historical_portfolio_returns(historical_returns, weights, x_cov=x_cov)
        portfolio_var = historical_portfolio_returns_risk.var(dim=1, unbiased=True)
        annualized_variance = portfolio_var * self.periods_per_year
        
        # Return is now calculated via the base class helper method
        annualized_return = self._calculate_objective_return(y_returns, weights, mode)

        utility = annualized_return - (self.risk_aversion / 2) * annualized_variance
        return -utility.mean()

class HistoricalVariance_SortinoRatioLoss(_BaseHistoricalSimLoss):
    """Calculates loss based on the Sortino Ratio, using historical simulation."""
    def __init__(self, risk_free_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.risk_free_rate = risk_free_rate

    def forward(self, weights, batch_x, y_returns=None, x_cov=None, mode='train'):
        # Extract the returns from the feature tensor batch_x. Returns are the first feature.
        historical_returns = batch_x[:, :, :, 0]

        # Risk is always calculated from historical data (downside deviation), using x_cov if provided, else historical_returns
        historical_portfolio_returns_risk = self._get_historical_portfolio_returns(historical_returns, weights, x_cov=x_cov)
        target_return_per_step = self.risk_free_rate / self.periods_per_year
        downside_returns = torch.minimum(historical_portfolio_returns_risk - target_return_per_step, torch.tensor(0.0, device=batch_x.device)) ** 2
        downside_std = torch.sqrt(downside_returns.mean(dim=1))
        annualized_downside = downside_std * torch.sqrt(torch.tensor(self.periods_per_year, device=downside_std.device))
        
        # Return is now calculated via the base class helper method
        annualized_return = self._calculate_objective_return(y_returns, weights, mode)
        
        excess_return = annualized_return - self.risk_free_rate
        sortino = excess_return / (annualized_downside + EPSILON)
        return -torch.nn.functional.softplus(sortino, beta=1).mean()

class HistoricalVariance_CVaRLoss(_BaseHistoricalSimLoss):
    """
    Calculates loss based on Conditional Value at Risk (CVaR).
    In train mode, it maximizes historical CVaR.
    In eval mode, it maximizes CVaR on future returns.
    """
    def __init__(self, alpha=0.95, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def forward(self, weights, batch_x, y_returns=None, x_cov=None, mode='train'):
        # For CVaR, risk calculation MUST use the long-term historical data from x_cov.
        if x_cov is None:
            raise ValueError("x_cov must be provided for HistoricalVariance_CVaRLoss.")

        # Always use x_cov for the historical simulation to calculate CVaR.
        # The `mode` or `y_returns` availability does not affect the risk calculation basis.
        historical_portfolio_returns = self._get_historical_portfolio_returns(returns_data=None, weights=weights, x_cov=x_cov)

        sorted_returns, _ = torch.sort(historical_portfolio_returns, dim=1)
        tail_size = max(int((1 - self.alpha) * sorted_returns.shape[1]), 1)
        cvar = sorted_returns[:, :tail_size].mean(dim=1)
        
        # This is a rate of return, so we should annualize it
        annualized_cvar = cvar * self.periods_per_year
        
        # The goal is to maximize CVaR (make it less negative), so we negate it for the loss
        return -annualized_cvar.mean()

# --- Loss Factory ---

LOSS_REGISTRY = {
    "Cov_MeanVariance": Cov_MeanVarianceLoss,
    "Cov_SharpeRatio": Cov_SharpeRatioLoss,
    "ExpectedReturn": ExpectedReturnLoss,
    "HistoricalVariance_SharpeRatio": HistoricalVariance_SharpeRatioLoss,
    "HistoricalVariance_MeanVariance": HistoricalVariance_MeanVarianceLoss,
    "HistoricalVariance_SortinoRatio": HistoricalVariance_SortinoRatioLoss,
    "HistoricalVariance_CVaR": HistoricalVariance_CVaRLoss,
}

def get_loss_function(loss_name: str, **kwargs) -> Type[nn.Module]:
    """Factory function to get a loss function class by name."""
    loss_class = LOSS_REGISTRY.get(loss_name)
    if not loss_class:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    sig = inspect.signature(loss_class.__init__)
    valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters or 'kwargs' in sig.parameters}
    
    return loss_class(**valid_args)