import torch
from functools import cached_property
import config

def _calculate_portfolio_returns(weights, y_returns):
    """
    Calculates the time-series of portfolio returns.
    This is a standalone utility function.

    Args:
        weights (torch.Tensor): Portfolio weights. Shape [B, N] or [B, N, H].
        y_returns (torch.Tensor): Future returns. Shape [B, N, H].

    Returns:
        torch.Tensor: Time-series of portfolio returns. Shape [B, H].
    """
    # Ensure weights have the same number of dimensions as y_returns for broadcasting
    # If weights are [B, N] and y_returns are [B, N, H], unsqueeze weights to [B, N, 1]
    if weights.dim() == 2 and y_returns.dim() == 3:
        weights = weights.unsqueeze(-1)
    
    # Element-wise product and sum over assets (N)
    # Resulting shape will be [B, H]
    return torch.sum(weights * y_returns, dim=1)


class PortfolioMetrics:
    """
    Portfolio evaluation metrics organized by calculation method.

    This class is designed to be instantiated with all necessary data (weights, historical data `x`, and future data `y`).
    It uses cached properties to avoid redundant calculations. It is a post-hoc backtesting tool.
    """

    def __init__(self, weights, y=None, x=None, x_cov=None,
                 horizon=21, risk_free_rate=0.0,
                 shrinkage=0.7, risk_aversion=1.0, burn_in_period=0):
        """
        Initializes the PortfolioMetrics evaluator for backtesting.

        Args:
            weights (torch.Tensor): Portfolio weights. Shape [B, N].
            y (torch.Tensor): Time-series of *future* returns for each asset. Used for realized return calculations. Shape [B, N, T_future].
            x (torch.Tensor): Time-series of *historical* returns for each asset (from model input). Used for risk calculations (e.g., HV). Shape [B, N, T_hist].
            x_cov (torch.Tensor, optional): Time-series of *historical* returns for covariance calculation. Shape [B, N, T_cov]. Defaults to None.
            horizon (int): Number of periods for future returns.
            risk_free_rate (float): Annual risk-free rate.
            shrinkage (float): Shrinkage factor for covariance estimation.
            risk_aversion (float): Risk aversion coefficient for mean-variance utility.
            burn_in_period (int): The burn in period used.
        """
        self.weights = weights
        self.y = y
        self.x = x
        self.x_cov = x_cov
        self.horizon = horizon
        self.risk_free_rate = risk_free_rate
        self.shrinkage = shrinkage
        self.risk_aversion = risk_aversion
        self.burn_in_period = burn_in_period

        # Device for creating new tensors
        self.device = weights.device

        # Pre-calculate annualization factors
        self.ann_factor_return = 252 / self.horizon
        self.ann_factor_std = (252 / self.horizon)**0.5

    # =====================================
    # Cached Intermediate Calculations
    # =====================================

    @cached_property
    def portfolio_returns_over_time(self):
        """Time-series of portfolio returns over the evaluation period. Based on `y` (future returns)."""
        if self.y is None:
            raise ValueError("`y` (future returns) is required for sample-based metrics.")
        # weights [B, N] -> [B, N, 1] | y [B, N, T_future]
        return torch.sum(self.y * self.weights.unsqueeze(2), dim=1)  # Result shape: [B, T_future]

    @cached_property
    def _historical_portfolio_returns(self):
        """
        Time-series of portfolio returns for historical risk metrics.
        Prioritizes `x_cov` if provided, otherwise uses `x`.
        """
        if self.x_cov is not None:
            data_for_hv_risk = self.x_cov
        elif self.x is not None:
            data_for_hv_risk = self.x
        else:
            raise ValueError("`x_cov` or `x` is required for HV risk calculation.")
            
        # weights [B, N] -> [B, N, 1] | data_for_hv_risk [B, N, T_hist]
        return torch.sum(data_for_hv_risk * self.weights.unsqueeze(2), dim=1)  # Result shape: [B, T_hist]

    @cached_property
    def _raw_covariance_matrix(self):
        """Raw covariance matrix from historical data (no shrinkage)."""
        if self.x_cov is not None:
            data_for_cov = self.x_cov
        elif self.x is not None:
            data_for_cov = self.x
        else:
            raise ValueError("`x_cov` or `x` is required for covariance-based metrics.")
            
        B, N, T = data_for_cov.shape
        mean = data_for_cov.mean(dim=2, keepdim=True)  # [B, N, 1]
        X_centered = data_for_cov - mean  # [B, N, T]
        return torch.matmul(X_centered, X_centered.transpose(1, 2)) / (T - 1)  # [B, N, N]

    @cached_property
    def covariance_matrix(self):
        """Shrinkage-adjusted covariance matrix from historical data `x`."""
        raw_cov = self._raw_covariance_matrix
        B, N, _ = raw_cov.shape
        trace = torch.diagonal(raw_cov, dim1=1, dim2=2).sum(dim=1, keepdim=True)
        avg_var = trace / N
        I = torch.eye(N, device=self.device).unsqueeze(0).expand(B, N, N)
        return (1 - self.shrinkage) * raw_cov + self.shrinkage * avg_var.view(B, 1, 1) * I

    # =====================================
    # 1. BASIC METRICS (Based on Realized Returns `y`)
    # =====================================

    def er(self):
        """Annualized Realized Portfolio Return (based on `y`). Shape: [B]"""
        return self.portfolio_returns_over_time.mean(dim=1) * self.ann_factor_return

    def mdd(self):
        """
        Calculates the Maximum Drawdown (MDD) for the entire evaluation period.
        MDD is the maximum loss from a peak to a trough of the portfolio's equity curve.
        """
        # Step 0: Get portfolio returns. The shape is [T, H] where T is the number of time steps.
        # We are interested in the daily return, so we take the first element of the horizon.
        returns = self.portfolio_returns_over_time[:, 0] # Shape -> [T]
        
        if returns.shape[0] <= 1:
            return torch.tensor(0.0, device=self.device)

        # Step 1: Calculate the cumulative wealth (equity curve) over the time axis (dim=0).
        initial_wealth = torch.ones(1, device=self.device)
        cumulative_wealth = torch.cat([initial_wealth, (1 + returns).cumprod(dim=0)], dim=0)

        # Step 2: Calculate the running maximum (peak) of the equity curve.
        running_max = torch.cummax(cumulative_wealth, dim=0)[0]

        # Step 3: Calculate the drawdown from the running maximum.
        drawdown = (cumulative_wealth - running_max) / running_max

        # Step 4: Find the maximum (most negative) drawdown for the entire period.
        # The result is a single scalar value. MDD is typically reported as a positive value.
        return torch.min(drawdown).abs()

    def positive_ratio(self):
        """Percentage of periods with positive returns, averaged across the batch. Returns a scalar."""
        positive_counts = (self.portfolio_returns_over_time > 0).sum(dim=1).float()
        ratios = positive_counts / self.portfolio_returns_over_time.shape[1]
        return ratios.mean() * 100

    def turnover(self):
        """Portfolio turnover (average weight change between periods). Returns a scalar."""
        if self.weights.shape[0] <= 1:
            return torch.tensor(0.0, device=self.device)
        weight_changes = torch.abs(self.weights[1:] - self.weights[:-1])
        return weight_changes.sum(dim=1).mean()

    # =====================================
    # 2. COVARIANCE-BASED METRICS (Risk from `x`, Return from `y`)
    # =====================================

    def cov_std(self):
        """Standard deviation using shrinkage covariance from `x`. Shape: [B]"""
        portfolio_var = torch.matmul(self.weights.unsqueeze(1), torch.matmul(self.covariance_matrix, self.weights.unsqueeze(2))).squeeze(-1).squeeze(-1)
        return torch.sqrt(portfolio_var * self.ann_factor_return + 1e-6)

    def nonshrink_cov_std(self):
        """Standard deviation using raw covariance from `x` (no shrinkage). Shape: [B]"""
        portfolio_var = torch.matmul(self.weights.unsqueeze(1), torch.matmul(self._raw_covariance_matrix, self.weights.unsqueeze(2))).squeeze(-1).squeeze(-1)
        return torch.sqrt(portfolio_var * self.ann_factor_return + 1e-6)

    def cov_sr(self):
        """Sharpe ratio (realized return `y` / cov-based risk `x`). Shape: [B]"""
        excess_return = self.er() - self.risk_free_rate
        return excess_return / (self.cov_std() + 1e-6)

    def nonshrink_cov_sr(self):
        """Sharpe ratio using raw covariance (no shrinkage). Shape: [B]"""
        excess_return = self.er() - self.risk_free_rate
        return excess_return / (self.nonshrink_cov_std() + 1e-6)

    def cov_sortino(self):
        """Sortino ratio using covariance-based downside deviation (Approximation). Shape: [B]"""
        std = self.cov_std()
        downside_std = std / torch.sqrt(torch.tensor(2.0, device=std.device))  # Approximation
        return (self.er() - self.risk_free_rate) / (downside_std + 1e-6)

    def cov_mv(self):
        """Mean-variance utility using realized return `y` and cov-based risk `x`. Shape: [B]"""
        variance = self.cov_std() ** 2
        return self.er() - (self.risk_aversion / 2) * variance

    def nonshrink_cov_mean_variance(self):
        """Mean-variance utility using raw covariance. Shape: [B]"""
        variance = self.nonshrink_cov_std() ** 2
        return self.er() - (self.risk_aversion / 2) * variance

    # =====================================
    # 3. SAMPLE-BASED (HV) METRICS (Return from `y`, Risk from `x`)
    # =====================================

    def hv_std(self):
        """Standard deviation using historical portfolio returns from `x`. Shape: [B]"""
        portfolio_std = self._historical_portfolio_returns.std(dim=1, unbiased=True)
        return portfolio_std * self.ann_factor_std

    def hv_sr(self):
        """Sharpe ratio using realized return from `y` and historical risk from `x`. Shape: [B]"""
        excess_return = self.er() - self.risk_free_rate
        return excess_return / (self.hv_std() + 1e-6)
    
    def hv_downside_std(self):
        """Calculates historical downside deviation for each sample. Shape: [B]"""
        daily_rf = self.risk_free_rate / 252
        historical_excess_returns = self._historical_portfolio_returns - daily_rf
        downside_sq = torch.minimum(historical_excess_returns, torch.tensor(0.0, device=self.device)) ** 2
        downside_var = torch.mean(downside_sq, dim=1)
        downside_std = torch.sqrt(downside_var + 1e-6)
        return downside_std * self.ann_factor_std

    def hv_sortino(self):
        """Sortino ratio using realized return from `y` and historical downside risk from `x`. Shape: [B]"""
        annual_return = self.er()
        annual_downside = self.hv_downside_std()
        return (annual_return - self.risk_free_rate) / (annual_downside + 1e-6)

    def hv_mv(self):
        """Mean-variance utility using realized return from `y` and historical variance from `x`. Shape: [B]"""
        annual_return = self.er()
        portfolio_var_x = self._historical_portfolio_returns.var(dim=1, unbiased=True)
        annual_variance = portfolio_var_x * self.ann_factor_return
        return annual_return - (self.risk_aversion / 2) * annual_variance

    # =====================================
    # 4. RISK METRICS (VaR from `y`, CVaR from `x`)
    # =====================================

    def var_95(self):
        """95% Value at Risk from `y` (realized returns). Shape: [B]"""
        sorted_returns, _ = torch.sort(self.portfolio_returns_over_time, dim=1)
        var_index = int(0.05 * self.portfolio_returns_over_time.shape[1])
        var_index = min(max(var_index, 0), self.portfolio_returns_over_time.shape[1] - 1)
        var = sorted_returns[:, var_index]
        return var * self.ann_factor_std

    def hv_cvar(self):
        """95% Conditional Value at Risk (Expected Shortfall) from `x` (historical returns). Shape: [B]"""
        sorted_returns, _ = torch.sort(self._historical_portfolio_returns, dim=1)
        tail_size = max(int(0.05 * self._historical_portfolio_returns.shape[1]), 1)
        cvar = sorted_returns[:, :tail_size].mean(dim=1)
        return cvar * self.ann_factor_std

    def es_ratio(self):
        """Expected Shortfall Ratio: (Return `y` - RF) / |CVaR `x`|. Shape: [B]"""
        excess_return = self.er() - self.risk_free_rate
        cvar = self.hv_cvar()
        return excess_return / (torch.abs(cvar) + 1e-6)

    @cached_property
    def stress_test_mdd(self):
        if self.x_cov is None or self.y is None: return torch.tensor(0.0, device=self.device)
        historical_returns = torch.sum(self.x_cov * self.weights.unsqueeze(2), dim=1)
        future_returns = self.portfolio_returns_over_time
        combined_returns = torch.cat([historical_returns, future_returns], dim=1)
        if combined_returns.shape[1] <= 1: return torch.tensor(0.0, device=self.device)
        equity_curve = torch.cumprod(1 + combined_returns, dim=1)
        ones = torch.ones(equity_curve.shape[0], 1, device=self.device)
        equity_curve = torch.cat([ones, equity_curve], dim=1)
        peak = torch.cummax(equity_curve, dim=1)[0]
        drawdown = (peak - equity_curve) / peak
        max_drawdown_per_sample = torch.max(drawdown, dim=1)[0]
        return torch.max(max_drawdown_per_sample)

    def get_all_metrics(self):
        """
        Computes and returns a dictionary of all metrics as scalar values.
        Switches calculation strategy based on config.PORTFOLIO_METRICS_MODE.
        - 'micro': Average of Ratios (e.g., mean of per-sample Sharpe Ratios).
        - 'macro': Ratio of Averages (e.g., ratio of mean-return over mean-volatility).
        """
        def _get_mean_metric(metric_fn):
            return metric_fn().mean()

        if config.PORTFOLIO_METRICS_MODE == "macro":
            avg_er = _get_mean_metric(self.er)
            avg_hv_std = _get_mean_metric(self.hv_std)
            avg_cov_std = _get_mean_metric(self.cov_std)
            avg_hv_downside_std = _get_mean_metric(self.hv_downside_std)
            avg_nonshrink_cov_std = _get_mean_metric(self.nonshrink_cov_std)
            avg_hv_cvar = _get_mean_metric(self.hv_cvar)
            
            avg_hv_var = _get_mean_metric(lambda: self.hv_std()**2)
            avg_cov_var = _get_mean_metric(lambda: self.cov_std()**2)
            avg_nonshrink_cov_var = _get_mean_metric(lambda: self.nonshrink_cov_std()**2)

            results = {
                'ER': avg_er, 'MDD': self.mdd(), 'Stress Test MDD': self.stress_test_mdd,
                'Positive Ratio': self.positive_ratio(), 'Turnover': self.turnover(),
                
                'HV SR': (avg_er - self.risk_free_rate) / (avg_hv_std + 1e-6),
                'HV Sortino': (avg_er - self.risk_free_rate) / (avg_hv_downside_std + 1e-6),
                'HV MV': avg_er - (self.risk_aversion / 2) * avg_hv_var,
                'HV Std': avg_hv_std,

                'Cov SR': (avg_er - self.risk_free_rate) / (avg_cov_std + 1e-6),
                'Cov Sortino': (avg_er - self.risk_free_rate) / ((avg_cov_std / (2**0.5)) + 1e-6),
                'Cov MV': avg_er - (self.risk_aversion / 2) * avg_cov_var,
                'Cov Std': avg_cov_std,

                'NonshrinkCov SR': (avg_er - self.risk_free_rate) / (avg_nonshrink_cov_std + 1e-6),
                'NonshrinkCov MeanVariance': avg_er - (self.risk_aversion / 2) * avg_nonshrink_cov_var,
                'NonshrinkCov Std': avg_nonshrink_cov_std,

                'VaR 95%': _get_mean_metric(self.var_95),
                'HV CVaR': avg_hv_cvar,
                'ES Ratio': (avg_er - self.risk_free_rate) / (torch.abs(avg_hv_cvar) + 1e-6),
            }
        else: # "micro" mode
            results = {
                'ER': _get_mean_metric(self.er), 'MDD': self.mdd(), 'Stress Test MDD': self.stress_test_mdd,
                'Positive Ratio': self.positive_ratio(), 'Turnover': self.turnover(),
                
                'HV SR': _get_mean_metric(self.hv_sr),
                'HV Sortino': _get_mean_metric(self.hv_sortino),
                'HV MV': _get_mean_metric(self.hv_mv), 'HV Std': _get_mean_metric(self.hv_std),

                'Cov SR': _get_mean_metric(self.cov_sr),
                'Cov Sortino': _get_mean_metric(self.cov_sortino),
                'Cov MV': _get_mean_metric(self.cov_mv), 'Cov Std': _get_mean_metric(self.cov_std),

                'NonshrinkCov SR': _get_mean_metric(self.nonshrink_cov_sr),
                'NonshrinkCov MeanVariance': _get_mean_metric(self.nonshrink_cov_mean_variance),
                'NonshrinkCov Std': _get_mean_metric(self.nonshrink_cov_std),

                'VaR 95%': _get_mean_metric(self.var_95),
                'HV CVaR': _get_mean_metric(self.hv_cvar),
                'ES Ratio': _get_mean_metric(self.es_ratio),
            }
        return {k: v.item() for k, v in results.items()}
