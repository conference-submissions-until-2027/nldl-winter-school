import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from pathlib import Path
import os
import pickle
import json
import torch
import matplotlib.pyplot as plt
from model.prototype_generator import generate_prototypes
import torch
import numpy as np
from model.utils import get_frechet_mean


def locate_files(path, pattern, csv_pattern = None):

    if csv_pattern is None:
        csv_pattern = 'training'
    
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{path} does not exist")
    subdirectories = [p for p in path.iterdir() if p.is_dir() and pattern in p.name]
    dir_to_file = {}
    for subdir in subdirectories:
        potential_csvs = [path for path in subdir.rglob("*.csv") if csv_pattern in str(path)]
        if len(potential_csvs) > 0:
            dir_to_file[subdir.name] = potential_csvs[0]

    return dir_to_file
    
    
class ExperimentVisualizer:
    def __init__(self, log_paths, exp_names):
        """
        Args:
            log_paths (list of str): Paths to the training_log.csv files.
            exp_names (list of str): Custom labels for each experiment (e.g., ['Pure Hyp', 'Mixed (alpha=0.5)']).
        """
        assert len(log_paths) == len(exp_names), "Must provide a name for each log path."
        
        self.experiments = {}
        self.metrics = set()
        
        for name, path in zip(exp_names, log_paths):
            if not os.path.exists(path):
                print(f"Warning: File not found -> {path}")
                continue
                
            df = pd.read_csv(path)
            self.experiments[name] = df
            
            # Extract plottable columns (exclude epoch, loss, and non-numerics)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            plottable = [c for c in numeric_cols if c not in ['epoch']]
            self.metrics.update(plottable)
            
        self.metrics = sorted(list(self.metrics))

    def _get_actual_col(self, df, base_metric, prefix):
        """Helper to resolve the column name with fallback."""
        if prefix and f"{prefix}{base_metric}" in df.columns:
            return f"{prefix}{base_metric}"
        elif base_metric in df.columns:
            return base_metric
        return ""

    def _get_best_epochs(self, target_metric='L2_Family_BalAcc', potential_prefix=None):
        """Finds the row yielding the highest target metric for each experiment."""
        best_data = {}
        for name, df in self.experiments.items():
            # Resolve target metric with prefix fallback
            actual_target = self._get_actual_col(df, target_metric, potential_prefix)
            
            # Fallback to standard Leaf_Acc if the specified target doesn't exist at all
            if not actual_target:
                actual_target = self._get_actual_col(df, 'Leaf_Acc', potential_prefix)
            
            if actual_target and actual_target in df.columns:
                best_idx = df[actual_target].idxmax()
                best_data[name] = df.loc[best_idx]
            else:
                best_data[name] = df.iloc[-1] # Fallback to last epoch
                
        return pd.DataFrame(best_data).T

    def plot(self, mode='curves', metrics_subset=None, target_metric='L2_Family_BalAcc', save_path=None, potential_prefix=None, title = None):
        """
        Generates a grid of plots for the specified metrics.
        
        Args:
            mode (str): 'curves' for line plots across all epochs, 'max' for bar charts at best epoch.
            metrics_subset (list): Specific column names to plot. If None, plots all shared metrics.
            target_metric (str): The metric used to determine the "best" epoch for 'max' mode.
            save_path (str): If provided, saves the figure to this path.
            potential_prefix (str): Prefix to prefer if it exists (e.g., "Forced_").
        """
        metrics_to_plot = metrics_subset if metrics_subset else self.metrics
        if not metrics_to_plot:
            print("No metrics available to plot.")
            return

        # Calculate Grid Layout
        n_metrics = len(metrics_to_plot)
        
        if n_metrics == 4:
            ncols = 2
        else:
            ncols = min(3, n_metrics)
            
        nrows = math.ceil(n_metrics / ncols)
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))
        if n_metrics == 1: axes = [axes]
        else: axes = axes.flatten()

        # Generate Plots
        if mode == 'curves':
            self._plot_curves(axes, metrics_to_plot, potential_prefix)
            if title:
                fig.suptitle(title, fontsize = 18, y = 1.01)
            else:
                fig.suptitle(f'Training Curves Comparison {"(Forced Preferred)" if potential_prefix else ""}', fontsize=18, y=1.02)
            
        elif mode == 'max':
            best_df = self._get_best_epochs(target_metric, potential_prefix)
            self._plot_bars(axes, metrics_to_plot, best_df, target_metric, potential_prefix)
            fig.suptitle(f'Peak Performance Comparison (Selected by Max {target_metric})', fontsize=18, y=1.02)
        else:
            raise ValueError("mode must be 'curves' or 'max'")

        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def _plot_curves(self, axes, metrics, potential_prefix):
        """Plots standard line charts over epochs with dynamic column resolution."""
        for ax, base_metric in zip(axes, metrics):
            for name, df in self.experiments.items():
                actual_col = self._get_actual_col(df, base_metric, potential_prefix)
                
                if actual_col:
                    # Append indicator to label if the forced prefix was used
                    label = f"{name} (F)" if (potential_prefix and actual_col.startswith(potential_prefix)) else name
                    ax.plot(df['epoch'], df[actual_col], marker='o', markersize=3, label=label)
            
            ax.set_title(base_metric.replace('_', ' '))
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score / Distance')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize=12)

    def _plot_bars(self, axes, metrics, best_df, target_metric, potential_prefix):
        """Plots grouped bar charts comparing the best epochs with dynamic column resolution."""
        exp_names = best_df.index.tolist()
        x = np.arange(len(exp_names))
        colors = plt.cm.get_cmap('tab10', len(exp_names))

        for ax, base_metric in zip(axes, metrics):
            values = []
            valid_plot = False
            
            for name in exp_names:
                df = self.experiments[name]
                actual_col = self._get_actual_col(df, base_metric, potential_prefix)
                
                # Fetch value from best_df if the column exists
                if actual_col and actual_col in best_df.columns:
                    val = best_df.loc[name, actual_col]
                    values.append(val if pd.notna(val) else 0.0)
                    valid_plot = True
                else:
                    values.append(0.0)

            if not valid_plot:
                ax.axis('off')
                continue

            bars = ax.bar(x, values, color=[colors(i) for i in range(len(exp_names))], edgecolor='black')
            
            # Annotate bars
            for bar, val in zip(bars, values):
                if val == 0.0: continue
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

            ax.set_title(base_metric.replace('_', ' '))
            ax.set_xticks(x)
            
            # Alter tick labels slightly if the forced version was used
            tick_labels = [f"{n} (F)" if (potential_prefix and self._get_actual_col(self.experiments[n], base_metric, potential_prefix).startswith(potential_prefix)) else n for n in exp_names]
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
            
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            if all(0<val<1 for val in values):
                ax.set_ylim(max(min(values)-0.05,0), min(max(values) + 0.05, 1))
            if base_metric == target_metric:
                ax.set_title(f"⭐ {base_metric.replace('_', ' ')}\n(Selection Target)")



# --- 2. Geodesic Calculator ---
def get_poincare_geodesic(u, v, num_points=50):
    """Computes the hyperbolic geodesic and projects it to the 2D Poincaré disk."""
    # Safety slice: Only use the first 3 dimensions (t, x, y) 
    # This prevents math errors if your embedding_dim > 3 (e.g., 10)
    u_3d = u[:3]
    v_3d = v[:3]
    
    # Minkowski Inner Product: -t1*t2 + x1*x2 + y1*y2
    minkowski_dot = -u_3d[0]*v_3d[0] + u_3d[1]*v_3d[1] + u_3d[2]*v_3d[2]
    minkowski_dot = torch.clamp(minkowski_dot, max=-1.0)
    
    d = torch.acosh(-minkowski_dot)
    
    # If nodes are virtually identical, return a single point
    if d < 1e-5:
        return [u_3d[1].item() / (u_3d[0].item() + 1.0)], [u_3d[2].item() / (u_3d[0].item() + 1.0)]
        
    lambda_vals = torch.linspace(0, 1, num_points)
    sinh_d = torch.sinh(d)
    
    xs, ys = [], []
    for lam in lambda_vals:
        c1 = torch.sinh((1 - lam) * d) / sinh_d
        c2 = torch.sinh(lam * d) / sinh_d
        
        pt = c1 * u_3d + c2 * v_3d
        
        # Stereographic Projection to 2D
        t, x, y = pt[0], pt[1], pt[2]
        px = x / (t + 1.0)
        py = y / (t + 1.0)
        
        xs.append(px.item())
        ys.append(py.item())
        
    return xs, ys


def read_checkpoint_path(path):

    prototype_path = os.path.join(path, 'prototypes.pkl')
    hierarchy_path = os.path.join(path, 'hierarchy_tree.json')
    embedding_path = os.path.join(path, 'evaluation_embeddings.pkl')
    return prototype_path, hierarchy_path, embedding_path


# --- 3. Main Plotting Function ---
def plot_generated_hyperbolic_tree(embedding_dim=10, target_radius=5.0, angular_alloc = 'log', title = "", outdir="./meta-data", save_path =""):
    """Executes the generator, loads the saved outputs, and plots the hierarchy."""
    print("\n--- Step 1: Generating Prototypes ---")
    # Call your existing generator function
    proto_path, tree_path = generate_prototypes(
        embedding_dim=embedding_dim, 
        target_radius=target_radius, 
        outdir=outdir,
        angular_allocation_metric = angular_alloc
    )
    
    print("\n--- Step 2: Loading Saved Data ---")
    with open(proto_path, 'rb') as f:
        prototypes = pickle.load(f)  # Dictionary of Name -> Tensor
        
    with open(tree_path, 'r') as f:
        hierarchy_tree = json.load(f) # Nested Dictionary
        
    # Extract string-based edges for drawing lines
    edges = _extract_string_edges(hierarchy_tree)
    
    print("--- Step 3: Plotting Poincaré Disk ---")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw Poincaré disk boundary
    boundary = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', zorder=1)
    ax.add_patch(boundary)
    
    # Draw Geodesic Edges
    for parent, child in edges:
        if parent in prototypes and child in prototypes:
            u = prototypes[parent]
            v = prototypes[child]
            arc_x, arc_y = get_poincare_geodesic(u, v)
            ax.plot(arc_x, arc_y, 'k-', alpha=0.4, linewidth=1.5, zorder=2)
            
    # Draw Nodes and Labels
    for name, vec in prototypes.items():
        t = vec[0].item()
        x = vec[1].item()
        y = vec[2].item()
        
        # Project node to 2D
        px = x / (t + 1.0)
        py = y / (t + 1.0)
        
        # Plot Node
        ax.scatter(px, py, c='dodgerblue', s=80, edgecolors='white', zorder=3)
        
        # Plot Label
        ax.text(px + 0.02, py + 0.02, name.title(), fontsize=9, 
                ha='left', va='bottom', zorder=4,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.3))
        
    # Formatting
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        plt.title(title, fontsize = 16, pad = 20)
    else:
        plt.title(f"Clinical Taxonomy - Poincaré Disk with Geodesics\nTarget Radius: {target_radius}", fontsize=14, pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format = save_path.split(".")[-1])
    plt.show()


def visualize_poincare_hierarchy(lorentz_prototypes, edges=None, labels=None):
    """
    Visualizes stationary Lorentz prototypes on the 2D Poincaré disk.
    
    Args:
        lorentz_prototypes: Tensor of shape (N, 3) representing the fixed nodes.
        edges: List of tuples [(parent_idx, child_idx), ...] defining the tree.
        labels: List of strings [N] for node names (e.g., 'Root', 'Melanoma').
    """
    # 1. Project from Lorentz to Poincaré
    # x0 is the time dimension, x1 and x2 are spatial
    x0 = lorentz_prototypes[:, 0]
    x_spatial = lorentz_prototypes[:, 1:]
    
    # Apply stereographic projection: p = x_spatial / (x0 + 1)
    denominator = (x0 + 1.0).unsqueeze(-1)
    poincare_coords = (x_spatial / denominator).detach().cpu().numpy()
    
    # 2. Setup the Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw the Poincaré disk boundary (Unit Circle)
    boundary = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--')
    ax.add_patch(boundary)
    
    # 3. Draw Edges (Geodesic Links)
    if edges:
        for parent_idx, child_idx in edges:
            # Safely check if indices are within bounds
            if parent_idx < len(lorentz_prototypes) and child_idx < len(lorentz_prototypes):
                # Get the 3D Lorentz tensors using integer indexing
                u = lorentz_prototypes[parent_idx]
                v = lorentz_prototypes[child_idx]
                
                # Compute the projected geodesic arc
                arc_x, arc_y = get_poincare_geodesic(u, v)
                
                # Plot the beautifully curved path
                ax.plot(arc_x, arc_y, 'k-', alpha=0.4, linewidth=1.5, zorder=2)
            
    # 4. Draw Nodes
    ax.scatter(poincare_coords[:, 0], poincare_coords[:, 1], 
               c='blue', s=100, zorder=5, edgecolors='white')
    
    # 5. Add Labels
    if labels:
        for i, (x, y) in enumerate(poincare_coords):
            # Offset labels slightly so they don't overlap the node
            ax.text(x + 0.03, y + 0.03, labels[i], fontsize=9, 
                    ha='left', va='bottom', zorder=10)
            
    # Formatting
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("Fixed Clinical Taxonomy (Poincaré Disk with Geodesics)")
    plt.tight_layout()
    plt.show()


# --- Helper Functions ---
def _extract_string_edges(tree_dict, parent=None, edges=None):
    if edges is None: edges = []
    for node_name, children in tree_dict.items():
        if parent is not None: edges.append((parent, node_name))
        if isinstance(children, dict) and children:
            _extract_string_edges(children, parent=node_name, edges=edges)
    return edges

def get_poincare_geodesic(u, v, num_points=50):
    u_3d, v_3d = u[:3], v[:3]
    minkowski_dot = -u_3d[0]*v_3d[0] + u_3d[1]*v_3d[1] + u_3d[2]*v_3d[2]
    minkowski_dot = torch.clamp(minkowski_dot, max=-1.0)
    d = torch.acosh(-minkowski_dot)
    
    if d < 1e-5:
        return [u_3d[1].item() / (u_3d[0].item() + 1.0)], [u_3d[2].item() / (u_3d[0].item() + 1.0)]
        
    lambda_vals = torch.linspace(0, 1, num_points)
    sinh_d = torch.sinh(d)
    xs, ys = [], []
    for lam in lambda_vals:
        c1 = torch.sinh((1 - lam) * d) / sinh_d
        c2 = torch.sinh(lam * d) / sinh_d
        pt = c1 * u_3d + c2 * v_3d
        px = pt[1] / (pt[0] + 1.0)
        py = pt[2] / (pt[0] + 1.0)
        xs.append(px.item())
        ys.append(py.item())
    return xs, ys



def plot_embeddings_with_fine_centroids(proto_path, tree_path, embeddings_path, save_path = "", title = ""):
    print("--- Loading Data ---")
    with open(proto_path, 'rb') as f:
        prototypes = pickle.load(f)
    with open(tree_path, 'r') as f:
        hierarchy_tree = json.load(f)
    with open(embeddings_path, 'rb') as f:
        val_data = pickle.load(f)

    embs_tensor = val_data['embeddings'] 
    targets = np.array(val_data['targets'])
    class_names = np.array(val_data['class_names']) # L2 Families
    class_name_mapping = val_data['class_name_mapping'] # Fine-grained mapping
    
    # Project Validation Embeddings to Poincaré Disk
    t = embs_tensor[:, 0]
    x = embs_tensor[:, 1]
    y = embs_tensor[:, 2]
    px = (x / (t + 1.0)).numpy()
    py = (y / (t + 1.0)).numpy()

    print("--- Plotting ---")
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # 1. Draw Poincaré disk boundary
    boundary = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', zorder=0)
    ax.add_patch(boundary)

    # 2. Process Families and Fine-Grained Centroids
    unique_families = np.unique(class_names)
    cmap = plt.get_cmap('tab10') 
    
    for i, fam in enumerate(unique_families):
        fam_mask = class_names == fam
        fam_color = cmap(i % 10)
        
        # Plot the raw cluster points (colored by L2 Family)
        ax.scatter(
            px[fam_mask], py[fam_mask], 
            color=fam_color, 
            label=fam.title(), 
            alpha=0.35,      
            s=15,            
            edgecolors='none', 
            zorder=1         
        )
        
        # --- FINE-GRAINED CENTROID LOGIC (>15% threshold) ---
        fam_targets = targets[fam_mask]
        total_fam_samples = len(fam_targets)
        
        # Get unique targets and their counts within this specific family
        unique_t, counts_t = np.unique(fam_targets, return_counts=True)
        
        for tgt, count in zip(unique_t, counts_t):
            proportion = count / total_fam_samples
            
            if proportion > 0.05:
                # Isolate the embeddings for this specific fine-grained class
                fine_mask = (targets == tgt)
                fine_embs = embs_tensor[fine_mask]
                
                # Calculate the exact Riemannian mean
                hyperbolic_centroid = get_frechet_mean(fine_embs)
                
                # Project to Poincaré
                cent_px = (hyperbolic_centroid[1] / (hyperbolic_centroid[0] + 1.0)).item()
                cent_py = (hyperbolic_centroid[2] / (hyperbolic_centroid[0] + 1.0)).item()
                
               
                # Plot the Sub-Cluster Centroid as a Star
                ax.scatter(cent_px, cent_py, marker='*', s=350, color=fam_color, 
                           edgecolors='black', linewidth=1.5, zorder=6)
                
                # Label it with the fine-grained name and its proportion
                fine_name = class_name_mapping[tgt]
                label_text = f"{fine_name.title()} ({proportion*100:.0f}%)"
                print(f"{fine_name} was placed at ({cent_px}, {cent_py})")
                ax.text(cent_px + 0.02, cent_py + 0.02, label_text, fontsize=9, fontweight='bold',
                        ha='left', va='bottom', zorder=7, color='black',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.8))

    # 3. Draw Clinical Tree Edges (Geodesics)
    edges = _extract_string_edges(hierarchy_tree)
    for parent, child in edges:
        if parent in prototypes and child in prototypes:
            u, v = prototypes[parent], prototypes[child]
            arc_x, arc_y = get_poincare_geodesic(u, v)
            ax.plot(arc_x, arc_y, color='black', alpha=0.7, linewidth=1.5, zorder=2)

    # 4. Draw Prototype Nodes (Bullseyes)
    for name, vec in prototypes.items():
        pt_t, pt_x, pt_y = vec[0].item(), vec[1].item(), vec[2].item()
        pt_px = pt_x / (pt_t + 1.0)
        pt_py = pt_y / (pt_t + 1.0)
        
        ax.scatter(pt_px, pt_py, c='white', s=100, edgecolors='black', linewidth=1.5, zorder=3)
        ax.scatter(pt_px, pt_py, c='red', s=30, zorder=4) 
        
        ax.text(pt_px + 0.02, pt_py - 0.02, name.title(), fontsize=9, fontweight='bold',
                ha='left', va='top', zorder=5, color='darkred',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.0))

    # Formatting & Legend
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect('equal')
    ax.axis('off')
    # Custom Legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Add manual legend entries for the Markers
    from matplotlib.lines import Line2D
    # star_marker = Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
    #                      markeredgecolor='black', markersize=15, label='Actual Centroid')
    bullseye_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                             markeredgecolor='black', markersize=10, label='Fixed Prototype')
    
    handles.extend([bullseye_marker])
    labels.extend(['Fixed Prototype'])
    leg = ax.legend(handles=handles, labels=labels, loc='upper right', 
                    bbox_to_anchor=(1.15, 1), title="L2 Families", markerscale=1.1)
    
    # Use legend_handles (snake_case) for modern Matplotlib
    for lh in leg.legend_handles[:-2]: 
        lh.set_alpha(1)

    if title:
        plt.title(title, fontsize = 16, pad = 20)
    else:
        plt.title("Validation Embeddings: Actual Centroids vs Fixed Prototypes", fontsize=16, pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format = save_path.split(".")[-1])
    
    plt.show()

if __name__ == '__main__':
    # 1. Define your experiments
    log_files = [
        './output_pure_hyp/training_log.csv',
        './output_pure_euc/training_log.csv',
        './output_mixed_alpha05/training_log.csv'
    ]
    experiment_names = [
        'Pure Hyperbolic (R=2.5)',
        'Pure Euclidean (Hypersphere)',
        'Mixed Geometry (Alpha=0.5)'
    ]

    # 2. Initialize Visualizer
    viz = ExperimentVisualizer(log_files, experiment_names)

    # 3. Define the key metrics you want to analyze (keeps the plot grid readable)
    key_metrics = [
        'Leaf_Acc', 'Leaf_HED', 'L1_Malignancy_BalAcc', 
        'L2_Family_BalAcc', 'Euc_Leaf_Acc', 'Avg_Radius'
    ]

    # 4. Plot Curves (How the models evolved)
    viz.plot(mode='curves', metrics_subset=key_metrics, save_path='./curves_comparison.png')

    # 5. Plot Bars (The ultimate side-by-side at their respective peaks)
    viz.plot(mode='max', metrics_subset=key_metrics, save_path='./max_bars_comparison.png')


