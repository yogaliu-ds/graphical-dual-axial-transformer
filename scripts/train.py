import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import copy
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from models.score_block import get_score_block
from utils.loss import get_loss_function
from utils.evaluation_metrics import PortfolioMetrics
from itertools import product
import csv
import sys
from models.portfolio_block import PortfolioBlockSoftmax, PortfolioBlockAbsMax, PortfolioBlockGeneralizedSoftmax
import config
from config import GRID_SEARCH_CONFIG
from utils.common import set_seed, get_device, load_data, get_nearest_date_idx, create_experiment_dirs, ExperimentLogger, load_adjacency_matrix

# Global device setting
device = get_device()

def extract_time_series_batches(features_tuple, date_index, start_idx, end_idx, sequence_length, horizon, batch_size, burn_in_period):
    """
    Extracts time-series samples of (X, Y, X_cov, dates) tuples in batches.

    Args:
        features_tuple (tuple): Tuple of feature arrays [T, N]. First feature must be returns.
        date_index (np.array): Array of date strings for the features.
        start_idx (int): Start index for sampling.
        end_idx (int): End index for sampling.
        sequence_length (int): Length of the input sequence for the model (X).
        horizon (int): Length of the future prediction period (Y).
        batch_size (int): Number of samples per batch.
        burn_in_period (int): Length of the historical period for covariance estimation (X_cov).

    Returns:
        list: A list of batches, where each batch is a tuple `(x_batch, y_batch, x_cov_batch, dates_batch)`.
        - x_batch (np.array): Model input features, shape [B, N, sequence_length, E].
        - y_batch (np.array): Future returns (target), shape [B, N, horizon].
        - x_cov_batch (np.array): Historical returns for covariance, shape [B, N, burn_in_period].
        - dates_batch (list): List of date strings for the prediction point, length [B].
    """
    batches = []
    returns_data = features_tuple[0]
    
    # The first sample requires a full history for all data slices.
    effective_start_idx = max(start_idx, sequence_length, burn_in_period)

    total_samples = end_idx - effective_start_idx - horizon + 1
    if total_samples <= 0:
        return batches
    
    all_x = []
    all_y = []
    all_x_cov = []
    all_dates = []

    for i in range(total_samples):
        t_end_x = effective_start_idx + i
        t_start_y = t_end_x + 1

        if t_start_y + horizon > end_idx:
            break

        # Extract X: past `sequence_length` days of all features
        feature_list = []
        for feature_data in features_tuple:
            feature_list.append(feature_data[t_end_x - sequence_length + 1 : t_end_x + 1, :].T)
        x_feat = np.stack(feature_list, axis=-1) # Shape: [N, sequence_length, E]

        # Extract Y: future `horizon` days of returns
        y_feat = returns_data[t_start_y : t_start_y + horizon, :].T # Shape: [N, horizon]
        
        # Extract X_cov: past `burn_in_period` days of returns
        x_cov_feat = returns_data[t_end_x - burn_in_period + 1 : t_end_x + 1, :].T # Shape: [N, burn_in_period]

        # Extract date for the prediction point
        # The date corresponds to the start of the prediction period (Y)
        date_feat = date_index[t_start_y]

        all_x.append(x_feat)
        all_y.append(y_feat)
        all_x_cov.append(x_cov_feat)
        all_dates.append(date_feat)

    # Batch the collected samples
    for i in range(0, len(all_x), batch_size):
        x_batch = np.array(all_x[i:i+batch_size])
        y_batch = np.array(all_y[i:i+batch_size])
        x_cov_batch = np.array(all_x_cov[i:i+batch_size])
        dates_batch = all_dates[i:i+batch_size]
        batches.append((x_batch, y_batch, x_cov_batch, dates_batch))

    return batches

class TimeSerieBatchDataLoader:
    """Custom DataLoader for time series batches."""

    def __init__(self, batches, shuffle=False):
        self.batches = batches
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.batches)))
        if self.shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            x_batch, y_batch, x_cov_batch, dates_batch = self.batches[idx]
            x_tensor = torch.from_numpy(x_batch).float()
            y_tensor = torch.from_numpy(y_batch).float()
            x_cov_tensor = torch.from_numpy(x_cov_batch).float()
            yield x_tensor, y_tensor, x_cov_tensor, dates_batch

    def __len__(self):
        return len(self.batches)

def prepare_data(features_tuple, date_index, burn_in_period, T0, T1, T2, batch_size, sequence_length, horizon, max_train_samples):
    # The burn_in_period determines the start of the first training sample.
    train_start_idx = burn_in_period
    
    print(f"Extracting training batches from index {train_start_idx} to {T0}...")
    train_batches = extract_time_series_batches(
        features_tuple, date_index, train_start_idx, T0, sequence_length, horizon, batch_size, burn_in_period
    )

    # Validation and Test sets should start from their respective split points.
    print(f"Extracting validation batches from index {T0} to {T1}...")
    val_batches = extract_time_series_batches(
        features_tuple, date_index, T0, T1, sequence_length, horizon, batch_size, burn_in_period
    )

    print(f"Extracting test batches from index {T1} to {T2}...")
    test_batches = extract_time_series_batches(
        features_tuple, date_index, T1, T2, sequence_length, horizon, 1, burn_in_period
    )

    # Apply max_train_samples limit if specified
    if max_train_samples is not None and len(train_batches) > 0:
        # This logic needs to be carefully implemented based on actual sample count
        pass # Placeholder for sample limiting logic

    # Create custom dataloaders
    train_dataloader = TimeSerieBatchDataLoader(train_batches, shuffle=True)
    val_dataloader = TimeSerieBatchDataLoader(val_batches, shuffle=False)
    test_dataloader = TimeSerieBatchDataLoader(test_batches, shuffle=False)

    # Calculate total number of samples
    train_size = sum(len(batch[0]) for batch in train_batches) if train_batches else 0
    val_size = sum(len(batch[0]) for batch in val_batches) if val_batches else 0
    test_size = sum(len(batch[0]) for batch in test_batches) if test_batches else 0

    print(f"Created {len(train_batches)} training batches with {train_size} total samples")
    print(f"Created {len(val_batches)} validation batches with {val_size} total samples")
    print(f"Created {len(test_batches)} test batches with {test_size} total samples")

    return train_dataloader, val_dataloader, test_dataloader, train_size, val_size, test_size

class PortfolioModel(nn.Module):
    def __init__(self, score_block, portfolio_block):
        super(PortfolioModel, self).__init__()
        self.score_block = score_block
        self.portfolio_block = portfolio_block
    def forward(self, x, x_cov=None, N_mask=None, T_mask=None):
        scores = self.score_block(x, x_cov=x_cov, N_mask=N_mask, T_mask=T_mask)
        weights = self.portfolio_block(scores)
        return weights

def train_epoch(model, train_dataloader, optimizer, loss_fn, device, N_mask=None):
    model.train()
    total_loss = 0
    for x, y, x_cov, _ in tqdm(train_dataloader, desc="Training"): # dates ignored
        x, y, x_cov = x.to(device), y.to(device), x_cov.to(device)
        weights = model(x, x_cov=x_cov, N_mask=N_mask)
        loss = loss_fn(weights=weights, batch_x=x, y_returns=y, x_cov=x_cov, mode='train')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_dataloader) if train_dataloader else 0

def validate(model, val_dataloader, loss_fn, device, N_mask=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y, x_cov, _ in tqdm(val_dataloader, desc="Validating"): # dates ignored
            x, y, x_cov = x.to(device), y.to(device), x_cov.to(device)
            weights = model(x, x_cov=x_cov, N_mask=N_mask)
            loss = loss_fn(weights=weights, batch_x=x, y_returns=y, x_cov=x_cov, mode='eval')
            total_loss += loss.item()
    return total_loss / len(val_dataloader) if val_dataloader else 0

def evaluate(model, test_dataloader, device, shrinkage, risk_free_rate, horizon, risk_aversion, burn_in_period, N_mask=None):
    model.eval()
    all_weights, all_y, all_x, all_x_cov, all_dates = [], [], [], [], []
    with torch.no_grad():
        for x, y, x_cov, dates in tqdm(test_dataloader, desc="Evaluating"):
            x, y, x_cov = x.to(device), y.to(device), x_cov.to(device)
            weights = model(x, x_cov=x_cov, N_mask=N_mask)
            all_weights.append(weights)
            all_y.append(y)
            all_x.append(x)
            all_x_cov.append(x_cov)
            all_dates.extend(dates)

    if not all_weights:
        print("No data to evaluate.")
        return {}, [], [], []
    
    all_weights = torch.cat(all_weights, dim=0)
    all_y = torch.cat(all_y, dim=0)
    all_x = torch.cat(all_x, dim=0)
    all_x_cov = torch.cat(all_x_cov, dim=0)
    
    # Instantiate PortfolioMetrics for a full out-of-sample backtest
    # It receives both historical `x` and realized future `y` to provide a complete picture
    
    # Prepare arguments for PortfolioMetrics
    historical_returns = all_x[:, :, :, 0] # Extract 3D returns from 4D features

    metrics_evaluator = PortfolioMetrics(
        weights=all_weights,
        y=all_y,
        x=historical_returns,
        x_cov=all_x_cov,
        horizon=horizon,
        risk_free_rate=risk_free_rate,
        shrinkage=shrinkage,
        risk_aversion=risk_aversion,
        burn_in_period=burn_in_period
    )
    
    # get_all_metrics will be implemented in the next step to return a dict of all metrics
    results = metrics_evaluator.get_all_metrics() 
    
    print("\n--- Backtest Evaluation Results ---")
    for metric_name, value in results.items():
        print(f"{metric_name:30}: {value:.4f}")
    print("-" * 40)
    
    return results, all_dates, all_weights, all_y

def run_experiment(config):
    """
    Runs a single, self-contained experiment based on the provided configuration.
    """
    # Unpack configuration
    dataset_name = config['dataset_name']
    score_block_name = config['score_block_name']
    portfolio_block_name = config['portfolio_block_name']
    loss_type = config['loss_type']
    seed = config['seed']
    
    # Data & Model Shape
    data_cfg = config['data_cfg']
    burn_in_period = config['burn_in_period']
    sequence_length = config['sequence_length']
    horizon = config['horizon']
    batch_size = config['batch_size']
    adj_matrix = config['adj_matrix']
    
    # Training Hyperparameters
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    early_stop_patience = config['early_stop_patience']
    min_train_loss = config['min_train_loss']
    
    # Model Hyperparameters
    hidden_dim = config['hidden_dim']
    L_encoder = config['L_encoder']
    L_decoder = config['L_decoder']
    heads = config['heads']
    dropout = config['dropout']
    
    # Financial & Loss Parameters
    risk_aversion = config['risk_aversion']
    risk_free_rate = config['risk_free_rate']
    shrinkage = data_cfg['shrinkage']

    set_seed(seed) 
    
    # --- Data Loading ---
    features = load_data(
        data_cfg['price_path'], data_cfg['high_path'], data_cfg['low_path']
    )
    df = pd.read_csv(data_cfg['price_path'], index_col=0)
    # The first row of features is lost due to pct_change, so dates must be offset by 1
    date_index_str = pd.to_datetime(df.index[1:], utc=True, errors='coerce').strftime('%Y-%m-%d').to_numpy()
    
    T2_date = data_cfg['T2_date']
    last_data_date = date_index_str[-1]
    if T2_date > last_data_date: T2_date = last_data_date
        
    T0 = get_nearest_date_idx(date_index_str, data_cfg['T0_date'])
    T1 = get_nearest_date_idx(date_index_str, data_cfg['T1_date'])
    T2 = get_nearest_date_idx(date_index_str, T2_date)

    train_dataloader, val_dataloader, test_dataloader, _, _, _ = prepare_data(
        features, date_index_str, burn_in_period, T0, T1, T2, batch_size, sequence_length, horizon, config['max_train_samples']
    )
    
    N = features[0].shape[1]
    input_dim = len(features)
    print(f"Number of assets (N): {N}, Input Dimensions (Features): {input_dim}")

    # --- Model Initialization ---
    score_block = get_score_block(
        model_name=score_block_name, input_dim=input_dim, N=N, T=sequence_length,
        hidden_dim=hidden_dim, L_encoder=L_encoder, L_decoder=L_decoder, heads=heads, dropout=dropout
    )
    
    portfolio_block_map = {'softmax': PortfolioBlockSoftmax(), 'absmax': PortfolioBlockAbsMax(), 'generalized': PortfolioBlockGeneralizedSoftmax()}
    portfolio_block = portfolio_block_map.get(portfolio_block_name)

    model = PortfolioModel(score_block=score_block, portfolio_block=portfolio_block).to(device)
    
    # --- Loss & Optimizer ---
    sharpe_loss_mode = config.get('sharpe_loss_mode', 'softplus')
    loss_params = {
        'risk_free_rate': risk_free_rate, 
        'shrinkage': shrinkage, 
        'risk_aversion': risk_aversion,
        'mode': sharpe_loss_mode,
        'burn_in_period': burn_in_period,
    }
    loss_fn = get_loss_function(loss_type, **loss_params).to(device)

    best_model = model
    best_epoch = 0

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Using Loss: {loss_type} ({loss_fn.__class__.__name__})")

    # --- Training Loop ---
    best_val_loss, epochs_no_improve = float('inf'), 0
    best_model_from_training = None

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_fn, device, N_mask=adj_matrix)
        val_loss = validate(model, val_dataloader, loss_fn, device, N_mask=adj_matrix)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if train_loss < min_train_loss:
            print(f"Train loss {train_loss:.4f} < {min_train_loss}, early stopping.")
            if best_model_from_training is None: best_model_from_training = copy.deepcopy(model)
            best_epoch = epoch + 1
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_from_training = copy.deepcopy(model)
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f"Best model updated at epoch {epoch+1}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Validation loss did not improve for {early_stop_patience} epochs. Early stopping.")
                if best_epoch == 0: best_epoch = epoch + 1
                break

    if best_model_from_training is not None:
        best_model = best_model_from_training
    elif best_epoch == 0:
        best_epoch = num_epochs

    # --- Evaluation ---
    results, all_dates, all_weights, all_y = evaluate(best_model, test_dataloader, device, shrinkage, risk_free_rate, horizon, risk_aversion, burn_in_period, N_mask=adj_matrix)

    # --- Result Aggregation & Portfolio Return Calculation ---
    results.update({
        'Dataset': dataset_name, 'ScoreBlock': score_block_name, 'PortfolioBlock': portfolio_block_name,
        'LossType': loss_type, 'Seed': seed, 'LearningRate': learning_rate, 'NumEpochs': num_epochs, 
        'BestEpoch': best_epoch, 'RiskAversion': risk_aversion, 'hidden_dim': hidden_dim, 
        'L_encoder': L_encoder, 'L_decoder': L_decoder, 'heads': heads, 'dropout': dropout,
        'EarlyStopPatience': early_stop_patience
    })

    if all_dates:
        # 1. Manually calculate the portfolio returns using the helper function
        from utils.evaluation_metrics import _calculate_portfolio_returns
        # y is [B, N, H], weights is [B, N, H]. We are interested in the daily return, so we take the first element of the horizon.
        portfolio_returns_series = _calculate_portfolio_returns(all_weights, all_y)[:, 0]

        # 2. Create the dictionary with dates as keys
        returns_by_date = {
            date: value.item() 
            for date, value in zip(all_dates, portfolio_returns_series)
        }

        # 3. Add this dictionary to the final results, placing these columns at the end.
        results.update(returns_by_date)

    return results

def main():
    experiment_dirs = create_experiment_dirs()
    logger = ExperimentLogger(f"{experiment_dirs['logs']}/experiment.log")
    sys.stdout = logger
    logger.log_config(config.CONFIG)
    
    # --- Grid Search Setup ---
    param_grid = {
        "datasets": GRID_SEARCH_CONFIG["datasets"],
        "score_blocks": GRID_SEARCH_CONFIG["score_blocks"],
        "lr_epochs_list": GRID_SEARCH_CONFIG["lr_epochs_list"],
        "seeds": GRID_SEARCH_CONFIG["seeds"],
        "portfolio_blocks": GRID_SEARCH_CONFIG["portfolio_blocks"],
        "loss_types": GRID_SEARCH_CONFIG["loss_types"],
        "risk_aversions": GRID_SEARCH_CONFIG["risk_aversions"],
        "early_stop_patience_list": GRID_SEARCH_CONFIG["early_stop_patience_list"],
        "model_params": list(product(
            GRID_SEARCH_CONFIG["hidden_dims"], GRID_SEARCH_CONFIG["L_encoders"],
            GRID_SEARCH_CONFIG["L_decoders"], GRID_SEARCH_CONFIG["heads_list"],
            GRID_SEARCH_CONFIG["dropouts"]
        ))
    }
    model_param_names = ['hidden_dim', 'L_encoder', 'L_decoder', 'heads', 'dropout']
    
    all_results = []
    
    for dataset_name in param_grid["datasets"]:
        adj_matrix_path = f"data/{dataset_name}/graphical_lasso/adj_matrix_{dataset_name}.npy"
        current_adj_matrix = torch.from_numpy(np.load(adj_matrix_path)).float().to(device)
        # current_adj_matrix=None

        grid_keys = [k for k in param_grid.keys() if k not in ['datasets', 'model_params']]
        
        # Combine all other grid search parameters
        grid_values = [param_grid[k] for k in grid_keys]
        grid_values.append(param_grid['model_params']) # Add model params to product
        grid_keys.append('model_params') # Also add the name

        combined_grid = list(product(*grid_values))
        
        for i, params in enumerate(combined_grid):
            param_dict = dict(zip(grid_keys, params))
            model_param_tuple = param_dict.pop('model_params')
            model_param_dict = dict(zip(model_param_names, model_param_tuple))
            
            lr, epochs = param_dict['lr_epochs_list']

            log_str = (f"Score={param_dict['score_blocks']}, Portfolio={param_dict['portfolio_blocks']}, "
                       f"Loss={param_dict['loss_types']}, LR={lr}, Seed={param_dict['seeds']}, "
                       f"RiskAversion={param_dict['risk_aversions']}, Patience={param_dict['early_stop_patience_list']}, "
                       + ", ".join([f"{k}={v}" for k, v in model_param_dict.items()]))
            print(f"\n--- Running Experiment {i+1}/{len(combined_grid)} for Dataset: {dataset_name} ---")
            print(log_str)
            
            # --- Build Config for This Run ---
            data_cfg = config.CONFIG['data'][dataset_name]
            model_cfg = config.CONFIG['model']
            training_cfg = config.CONFIG['training']
            
            exp_config = {
                'dataset_name': dataset_name,
                'score_block_name': param_dict['score_blocks'],
                'portfolio_block_name': param_dict['portfolio_blocks'],
                'loss_type': param_dict['loss_types'],
                'seed': param_dict['seeds'],
                'learning_rate': lr, 'num_epochs': epochs,
                'early_stop_patience': param_dict['early_stop_patience_list'], 
                'risk_aversion': param_dict['risk_aversions'],
                **model_param_dict,
                
                'data_cfg': data_cfg,
                'burn_in_period': model_cfg['burn_in_period'],
                'sequence_length': model_cfg['sequence_length'],
                'horizon': model_cfg['horizon'],
                'batch_size': model_cfg['batch_size'],
                'risk_free_rate': model_cfg['risk_free_rate'],
                'min_train_loss': training_cfg['min_train_loss'],
                'max_train_samples': training_cfg['max_train_samples'],
                'sharpe_loss_mode': training_cfg.get('sharpe_loss_mode', 'softplus'),
                'adj_matrix': current_adj_matrix,
            }

            result = run_experiment(exp_config)
            all_results.append(result)

    # --- Save All Results ---
    if all_results:
        # Print results row by row as a list of dictionaries
        print("\n--- Final Experiment Results Summary ---")
        for row in all_results:
            print(row)
        print("--------------------------------------\n")

        keys = all_results[0].keys()
        results_path = f"{experiment_dirs['results']}/experiment_results.csv"
        with open(results_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nAll experiments completed. Results saved to {results_path}")
    else:
        print("\nNo experiments were run.")

    logger.close()

if __name__ == "__main__":
    main()