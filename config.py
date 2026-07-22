# config.py

# PORTFOLIO METRICS CALCULATION MODE
# "macro": Metrics are calculated on the aggregated (averaged) portfolio returns series across the entire batch.
# "micro": (Default in original implementation) Metrics are calculated for each sample in the batch and then averaged.
PORTFOLIO_METRICS_MODE = "micro"

# Additional extensible parameters

# GRID SEARCH CONFIGURATION
'''
--- Available Options ---
score_blocks: ['GDAT']
portfolio_blocks: ['softmax', 
                  'absmax', 
                  'generalized']
loss_types: ['Cov_SharpeRatio', 
              'HistoricalVariance_SharpeRatio', 
              'ExpectedReturn', 
              'Cov_MeanVariance', 
              'HistoricalVariance_MeanVariance', 
              'HistoricalVariance_SortinoRatio', 
              'HistoricalVariance_CVaR']
datasets: ['SP500', 
          'NIKKEI225', 
          'TW50', 
          'RUSSELL1000', 
          'FTSE100']
'''
# ------------------------------------
GRID_SEARCH_CONFIG = {
    "hidden_dims": [64],
    "L_encoders": [2],
    "L_decoders": [2],
    "heads_list": [4],
    "dropouts": [0.1],
    "seeds": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    # "seeds": [2],
    "lr_epochs_list": [(1e-4, 1000)],
    "early_stop_patience_list": [15],
    "risk_aversions": [2.0],
    "score_blocks": ['GDAT'],
    "portfolio_blocks": ['softmax'],
    "datasets": ['FTSE100'],
    "loss_types": ['HistoricalVariance_MeanVariance'], # Currently active selection
}

# Per-dataset settings
DATASETS = {
    'SP500': {
        'T0_date': '2015-12-31',  # Train End
        'T1_date': '2018-12-31',  # Val End
        'T2_date': '2025-06-27',  # Test End
        'price_path': 'data/SP500/SP500_Close.csv',
        'high_path': 'data/SP500/SP500_High.csv',
        'low_path': 'data/SP500/SP500_Low.csv',
        'open_path': 'data/SP500/SP500_Open.csv',
        'adjclose_path': 'data/SP500/SP500_AdjClose.csv',
        'volume_path': 'data/SP500/SP500_Volume.csv',
        'shrinkage': 0.7,  # Linear shrinkage parameter for covariance estimation
        'graphical_lasso': {
            'alpha': 0.4,  # 0.4
            'max_iter': 200,
            'tol': 1e-4,
            'mode': 'cd',
            'verbose': True
        }
    },
    'NIKKEI225': {
        'T0_date': '2015-12-31',
        'T1_date': '2018-12-31',
        'T2_date': '2025-06-27',
        'price_path': 'data/NIKKEI225/NIKKEI225_Close.csv',
        'high_path': 'data/NIKKEI225/NIKKEI225_High.csv',
        'low_path': 'data/NIKKEI225/NIKKEI225_Low.csv',
        'shrinkage': 0.5,  # 0.5
        'graphical_lasso': {
            'alpha': 0.5,  # 0.5
            'max_iter': 300,
            'tol': 1e-4,
            'mode': 'cd',
            'verbose': True
        }
    },
    'TW50': {
        'T0_date': '2015-12-31',
        'T1_date': '2018-12-31',
        'T2_date': '2025-06-27',
        'price_path': 'data/TW50/TW50_Close.csv',
        'high_path': 'data/TW50/TW50_High.csv',
        'low_path': 'data/TW50/TW50_Low.csv',
        'shrinkage': 0.1,  # 0.2
        'graphical_lasso': {
            'alpha': 0.5,
            'max_iter': 200,
            'tol': 1e-4,
            'mode': 'cd',
            'verbose': True
        }
    },
    'RUSSELL1000': {
        'T0_date': '2015-12-31',
        'T1_date': '2018-12-31',
        'T2_date': '2025-06-27',
        'price_path': 'data/RUSSELL1000/RUSSELL1000_Close.csv',
        'high_path': 'data/RUSSELL1000/RUSSELL1000_High.csv',
        'low_path': 'data/RUSSELL1000/RUSSELL1000_Low.csv',
        'shrinkage': 0.8,  # 0.9
        'graphical_lasso': {
            'alpha': 0.5,
            'max_iter': 200,
            'tol': 1e-4,
            'mode': 'cd',
            'verbose': True
        }
    },
    'FTSE100': {
        'T0_date': '2015-12-31',
        'T1_date': '2018-12-31',
        'T2_date': '2025-06-27',
        'price_path': 'data/FTSE100/FTSE100_Close.csv',
        'high_path': 'data/FTSE100/FTSE100_High.csv',
        'low_path': 'data/FTSE100/FTSE100_Low.csv',
        'shrinkage': 0.3,  # 0.2
        'graphical_lasso': {
            'alpha': 0.25,  # 0.4
            'max_iter': 200,
            'tol': 1e-4,
            'mode': 'cd',
            'verbose': True
        }
    },
}

# Default dataset
DEFAULT_DATASET = 'SP500'

CONFIG = {
    'data': DATASETS,
    'model': {
        'burn_in_period': 252, # Burn-in period for initial history before sampling
        'sequence_length': 21,    # Input sequence length (T dimension) for the model
        'horizon': 1, # Future Period for performance examination. More than 1 may be a kind of temporal data leak
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'risk_aversion': 1.0,
        'risk_free_rate': 0.0,
        # 'max_gap': 5,  # Removed: max_gap condition for early stopping
    },
    'training': {
        'early_stop_patience': 5,
        'min_train_loss': -20,
        'max_train_samples': None, # Default: None (use all samples)
        'sharpe_loss_mode': 'softplus', # 'softplus' or 'linear'
    }
}

