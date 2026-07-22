import os
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso
import matplotlib.pyplot as plt
import networkx as nx
import config

for dataset_name, dataset_cfg in config.DATASETS.items():
    # Create dataset-specific output directory
    OUTPUT_DIR = f'data/{dataset_name}/graphical_lasso'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n=== Processing {dataset_name} ===")
    price_path = dataset_cfg['price_path']
    T0_date = dataset_cfg['T0_date']
    glasso_cfg = dataset_cfg['graphical_lasso']

    # Load data and split the training set based on T0_date
    price_df = pd.read_csv(price_path, index_col=0)
    train_df = price_df.loc[:T0_date]
    symbols = train_df.columns

    # Compute log returns
    returns = np.log(train_df).diff().dropna()
    # Standardize returns
    returns = (returns - returns.mean()) / returns.std()
    returns_np = returns.values

    # Detailed standardization check
    print("\n=== Post-standardization statistics check ===")
    print(f"Mean: {np.mean(returns_np):.8f} (should be close to 0)")
    print(f"Std: {np.std(returns_np):.8f} (should be close to 1)")
    print(f"Min: {np.min(returns_np):.4f}")
    print(f"Max: {np.max(returns_np):.4f}")
    print(f"Skewness: {np.mean(((returns_np - np.mean(returns_np)) / np.std(returns_np))**3):.4f}")
    print(f"Kurtosis: {np.mean(((returns_np - np.mean(returns_np)) / np.std(returns_np))**4):.4f}")

    # Data sanity check
    print(f"returns shape: {returns_np.shape}")
    print(f"returns mean: {np.mean(returns_np):.6f}, std: {np.std(returns_np):.6f}")
    if np.isnan(returns_np).any():
        print("[Warning] returns contains NaN!")
    if np.allclose(returns_np, 0):
        print("[Warning] returns are all zeros!")

    print("Any NaN in returns:", np.isnan(returns_np).any())
    print("Any Inf in returns:", np.isinf(returns_np).any())
    print("Correlation matrix mean/std:", np.mean(np.corrcoef(returns_np.T)), np.std(np.corrcoef(returns_np.T)))
    print("Current alpha:", glasso_cfg['alpha'])

    # Build and fit GraphicalLasso
    model = GraphicalLasso(
        alpha=glasso_cfg['alpha'],
        max_iter=glasso_cfg['max_iter'],
        tol=glasso_cfg['tol'],
        mode=glasso_cfg['mode'],
        verbose=glasso_cfg['verbose']
    )
    model.fit(returns_np)

    # Generate positive-edge and negative-edge adjacency matrices
    adj_matrix_pos = (model.precision_ > 0).astype(float)
    adj_matrix_neg = (model.precision_ < 0).astype(float)
    adj_matrix = (model.precision_ != 0).astype(float)  # In case all edges are needed

    # Save numpy files
    np.save(os.path.join(OUTPUT_DIR, f'adj_matrix_pos_{dataset_name}.npy'), adj_matrix_pos)
    np.save(os.path.join(OUTPUT_DIR, f'adj_matrix_neg_{dataset_name}.npy'), adj_matrix_neg)
    np.save(os.path.join(OUTPUT_DIR, f'adj_matrix_{dataset_name}.npy'), adj_matrix)

    # Save csv files
    adj_df_pos = pd.DataFrame(adj_matrix_pos, index=symbols, columns=symbols)
    adj_df_neg = pd.DataFrame(adj_matrix_neg, index=symbols, columns=symbols)
    adj_df = pd.DataFrame(adj_matrix, index=symbols, columns=symbols)
    adj_df_pos.to_csv(os.path.join(OUTPUT_DIR, f'adj_matrix_pos_{dataset_name}.csv'))
    adj_df_neg.to_csv(os.path.join(OUTPUT_DIR, f'adj_matrix_neg_{dataset_name}.csv'))
    adj_df.to_csv(os.path.join(OUTPUT_DIR, f'adj_matrix_{dataset_name}.csv'))

    # Precision matrix heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(model.precision_, cmap='RdBu')
    plt.colorbar()
    plt.title(f'Precision Matrix ({dataset_name})')
    plt.savefig(os.path.join(OUTPUT_DIR, f'precision_matrix_heatmap_{dataset_name}.png'))
    plt.close()

    # Network graph
    G = nx.Graph(adj_matrix)
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=500, font_size=10, font_weight='bold')
    plt.title(f'Asset Network Graph ({dataset_name})')
    plt.savefig(os.path.join(OUTPUT_DIR, f'network_graph_{dataset_name}.png'))
    plt.close()

    # Summary statistics
    print(f"Precision matrix shape: {model.precision_.shape}")

    print("\nPrecision Matrix Statistics:")
    print(f"Mean: {np.mean(model.precision_):.4f}")
    print(f"Std: {np.std(model.precision_):.4f}")
    print(f"Min: {np.min(model.precision_):.4f}")
    print(f"Max: {np.max(model.precision_):.4f}")

    # Count positive/negative edges in the precision matrix (excluding the diagonal)
    precision = model.precision_
    off_diag = ~np.eye(precision.shape[0], dtype=bool)
    num_pos_edges = np.sum((precision > 0) & off_diag)
    num_neg_edges = np.sum((precision < 0) & off_diag)
    print(f"Precision matrix positive edges (off-diagonal): {num_pos_edges}")
    print(f"Precision matrix negative edges (off-diagonal): {num_neg_edges}")

    print("\nAdjacency Matrix Statistics:")
    print(f"Non-zero elements: {np.sum(adj_matrix != 0)}")
    print(f"Total elements: {adj_matrix.size}")
    print(f"Sparsity: {1 - np.sum(adj_matrix != 0) / adj_matrix.size:.4f}")

    # Analyze each asset's connectivity degree distribution
    print(f"\n=== {dataset_name} asset connectivity degree distribution analysis ===")

    # Exclude diagonal elements (self-loops)
    np.fill_diagonal(adj_matrix, 0)
    np.fill_diagonal(adj_matrix_pos, 0)
    np.fill_diagonal(adj_matrix_neg, 0)

    # Compute each asset's connectivity degree
    degrees_total = np.sum(adj_matrix, axis=1)  # Total degree
    degrees_pos = np.sum(adj_matrix_pos, axis=1)  # Positive degree
    degrees_neg = np.sum(adj_matrix_neg, axis=1)  # Negative degree

    # Build a dataframe of per-asset connectivity degree
    connection_df = pd.DataFrame({
        'Asset': symbols,
        'Total_Connections': degrees_total.astype(int),
        'Positive_Connections': degrees_pos.astype(int),
        'Negative_Connections': degrees_neg.astype(int),
        'Connection_Ratio': degrees_total / (len(symbols) - 1),
        'Positive_Ratio': degrees_pos / (len(symbols) - 1),
        'Negative_Ratio': degrees_neg / (len(symbols) - 1)
    })

    # Sort by total number of connections
    connection_df = connection_df.sort_values('Total_Connections', ascending=False)

    print(f"Number of assets: {len(symbols)}")
    print(f"Total edges: {np.sum(adj_matrix) // 2}")
    print(f"Positive edges: {np.sum(adj_matrix_pos) // 2}")
    print(f"Negative edges: {np.sum(adj_matrix_neg) // 2}")
    print(f"Average total degree: {np.mean(degrees_total):.2f}")
    print(f"Average positive degree: {np.mean(degrees_pos):.2f}")
    print(f"Average negative degree: {np.mean(degrees_neg):.2f}")
    print(f"Degree standard deviation: {np.std(degrees_total):.2f}")
    print(f"Minimum degree: {np.min(degrees_total)}")
    print(f"Maximum degree: {np.max(degrees_total)}")

    # Degree distribution statistics
    from collections import Counter
    degree_counts = Counter(degrees_total)
    print(f"\nDegree distribution:")
    for degree in sorted(degree_counts.keys()):
        count = degree_counts[degree]
        percentage = count / len(symbols) * 100
        print(f"  Degree {degree}: {count} assets ({percentage:.1f}%)")

    # Network density
    total_possible_edges = len(symbols) * (len(symbols) - 1) / 2
    actual_edges = np.sum(adj_matrix) / 2
    network_density = actual_edges / total_possible_edges
    print(f"\nNetwork density: {network_density:.4f} ({actual_edges:.0f}/{total_possible_edges:.0f} edges)")

    # Number of isolated nodes
    isolated_nodes = np.sum(degrees_total == 0)
    print(f"Isolated nodes: {isolated_nodes} ({isolated_nodes/len(symbols)*100:.1f}%)")

    # Number of highly connected nodes
    high_connectivity_threshold = np.mean(degrees_total) + 2 * np.std(degrees_total)
    high_connectivity_nodes = np.sum(degrees_total > high_connectivity_threshold)
    print(f"High-connectivity nodes (>{high_connectivity_threshold:.1f}): {high_connectivity_nodes} ({high_connectivity_nodes/len(symbols)*100:.1f}%)")
