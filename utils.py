import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
import tqdm
import json



def read_tsv(path):
    df = pd.read_csv(path, sep=  "\t", low_memory=False)
    return df 

def read_csv(path):
    return pd.read_csv(path)

def read_table(path):

    if path.endswith('.tsv'):
        df = pd.read_csv(path, sep=  "\t", low_memory=False)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raise NotImplementedError("Path must be either csv or tsv")
    return df


def create_stratified_split(csv_path, val_ratio=0.2, seed=42):
    """
    Splits data while handling high imbalance and singleton classes.
    """
    df = pd.read_csv(csv_path)
    
    # Clean diagnosis names for grouping
    df['target'] = df['diagnosis_name'].str.lower().str.strip()
    
    # 1. Identify Singletons (Classes with only 1 sample)
    counts = df['target'].value_counts()
    singletons = counts[counts < 2].index.tolist()
    
    # 2. Separate Data
    df_singletons = df[df['target'].isin(singletons)].copy()
    df_main = df[~df['target'].isin(singletons)].copy()
    
    # 3. Stratified Split on the Main Data
    # using StratifiedKFold logic to get indices
    skf = StratifiedKFold(n_splits=int(1/val_ratio), shuffle=True, random_state=seed)
    
    # Just take the first fold
    train_idx, val_idx = next(skf.split(df_main, df_main['target']))
    
    train_main = df_main.iloc[train_idx]
    val_main = df_main.iloc[val_idx]
    
    # 4. Force Singletons into Train (otherwise we can't learn them)
    # Alternatively, you could exclude them, but for few-shot learning we keep them.
    train_final = pd.concat([train_main, df_singletons])
    val_final = val_main
    
    print(f"Split Statistics:")
    print(f"Train: {len(train_final)} (Inc. {len(df_singletons)} singletons)")
    print(f"Val:   {len(val_final)}")
    
    return train_final, val_final

def dataset_sanity_check(train_dataset, test_dataset):
    print("Running Dataset Sanity Check...")
    
    # 1. Check Label Maps (The Dictionary View)
    # This ensures Index 5 means 'Melanoma' in both sets
    train_map = train_dataset.label_map
    test_map = test_dataset.label_map
    
    assert train_map == test_map, \
        f"CRITICAL: Label maps differ!\nTrain keys: {list(train_map.keys())[:5]}\nTest keys: {list(test_map.keys())[:5]}"

    # 2. Check Prototype Tensors (The Model View)
    # get_prototypes() returns Tensors [N, 10]
    train_protos = train_dataset.get_prototypes()
    test_protos = test_dataset.get_prototypes()
    
    # Ensure they are on CPU for comparison
    if train_protos.device.type != 'cpu': train_protos = train_protos.cpu()
    if test_protos.device.type != 'cpu': test_protos = test_protos.cpu()

    # Iterate by Index (since maps are identical, indices align)
    for name, idx in train_map.items():
        vec_train = train_protos[idx]
        vec_test = test_protos[idx]
        
        # correct usage of all() and isclose()
        match = torch.all(torch.isclose(vec_train, vec_test, atol=1e-6))
        
        assert match.item(), f"Mismatch for class '{name}' (Index {idx})!\nTrain: {vec_train}\nTest: {vec_test}"

    print("Sanity Check Passed: Train and Test datasets are perfectly aligned.")

def generate_cascading_sampler(dataset):
    """
    Generates a WeightedRandomSampler that ensures perfect mathematical 
    class balance at every level of the hierarchy simultaneously.
    """
    print("--- Generating Cascading Hierarchical Sampler ---")
    
    # 1. Extract paths for all valid samples
    sample_paths = []
    for diagnosis in dataset.diagnoses:
        clean_diag = diagnosis.lower().strip()
        path = dataset.hierarchy_paths.get(clean_diag, {'L1': clean_diag, 'L2': None, 'L3': None})
        sample_paths.append(path)
        
    # 2. Build the exact tree structure (with self-loop prevention!)
    tree = {}
    node_counts = {} 
    
    for path in sample_paths:
        l1, l2, l3 = path.get('L1'), path.get('L2'), path.get('L3')
        
        terminal_node = l3 if l3 else (l2 if l2 else l1)
        node_counts[terminal_node] = node_counts.get(terminal_node, 0) + 1
        
        if l1:
            if l1 not in tree: tree[l1] = set()
            # Safety: Prevent L2 from being the same name as L1
            if l2 and l2 != l1:
                tree[l1].add(l2)
                if l2 not in tree: tree[l2] = set()
                # Safety: Prevent L3 from being the same name as L2 or L1
                if l3 and l3 != l2 and l3 != l1:
                    tree[l2].add(l3)

    # 3. Calculate the Cascading Probability Mass (Top-Down)
    target_mass = {}
    
    l1_nodes = [node for node in tree.keys() if not any(node in children for children in tree.values())]
    l1_mass = 1.0 / len(l1_nodes) if l1_nodes else 1.0
    
    # Safety: Added 'visited' set to break any remaining cyclic loops
    def cascade_mass(node, current_mass, visited=None):
        if visited is None: visited = set()
        if node in visited:
            return # Cycle detected, break out!
        visited.add(node)
        
        # Double check we don't process self as a child
        children = {c for c in tree.get(node, set()) if c != node}
        
        if not children:
            target_mass[node] = current_mass
        else:
            split_mass = current_mass / len(children)
            
            if node in node_counts:
                split_mass = current_mass / (len(children) + 1)
                target_mass[node] = split_mass
                
            for child in children:
                cascade_mass(child, split_mass, visited)

    for l1 in l1_nodes:
        cascade_mass(l1, l1_mass)

    # 4. Calculate individual sample weights (Mass / N_samples)
    sample_weights = []
    for path in sample_paths:
        l1, l2, l3 = path.get('L1'), path.get('L2'), path.get('L3')
        terminal_node = l3 if l3 else (l2 if l2 else l1)
        
        mass = target_mass[terminal_node]
        count = node_counts[terminal_node]
        
        weight = mass / count
        sample_weights.append(weight)

    sample_weights = torch.DoubleTensor(sample_weights)

    # Diagnostic Output
    print(f"Total Terminal Nodes Balanced: {len(target_mass)}")
    for node, mass in target_mass.items():
        count = node_counts.get(node, 0)
        print(f"  - {node:<25} | Samples: {count:<5} | Target Batch Mass: {mass*100:>5.1f}%")

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def generate_sampler(dataset, balance_factor=1.0, target_level='L1'):
    """
    Generates a WeightedRandomSampler directed at a specific hierarchy level.
    
    Args:
        dataset: The SkinLesionDataset object.
        balance_factor (float): 1.0 = Fully balanced, 0.0 = Natural distribution.
        target_level (str): 'L1' (Malign/Benign), 'L2' (Families), or 'L3'/'leaf'.
    """
    print(f"--- Generating Sampler (Target: {target_level}, Factor: {balance_factor}) ---")
    
    targets = []
    
    # 1. Extract the target label for every image based on the chosen level
    for diagnosis in dataset.diagnoses:
        clean_diag = diagnosis.lower().strip()
        
        if target_level == 'leaf':
            # Use exact diagnosis
            targets.append(clean_diag)
        else:
            # Look up the ancestor in our precomputed paths
            path = dataset.hierarchy_paths.get(clean_diag, {})
            ancestor = path.get(target_level)
            
            # Fallback to the exact diagnosis if the path is missing
            targets.append(ancestor if ancestor else clean_diag)
            
    # 2. Convert string targets to integer indices for frequency counting
    unique_targets = list(set(targets))
    target_to_idx = {name: i for i, name in enumerate(unique_targets)}
    target_indices = [target_to_idx[t] for t in targets]
    
    # 3. Calculate frequencies
    class_counts = np.bincount(target_indices)
    
    # 4. Calculate weights based on the balance_factor
    class_weights = [
        (1.0 / count) ** balance_factor if count > 0 else 0.0 
        for count in class_counts
    ]
    
    # 5. Assign a weight to every individual image
    sample_weights = [class_weights[t] for t in target_indices]
    sample_weights = torch.DoubleTensor(sample_weights)

    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Quick diagnostic print
    print(f"Unique {target_level} classes balanced: {len(unique_targets)}")
    for name, count, weight in zip(unique_targets, class_counts, class_weights):
        print(f"  - {name:<20} | Count: {count:<5} | Base Weight: {weight:.4f}")
    
    return sampler


def get_class_weights(dataset, balance_factor = 1.0):
    all_diagnosis = dataset.df['diagnosis_names']

    targets = []
    for diagnosis in all_diagnosis:
        diagnosis = diagnosis.lower().strip()
        label_idx = dataset.label_map.get(diagnosis, -1)
        targets.append(label_idx)
    
    class_counts = np.bincount(targets)
    
    # Calculate weights based on the balance_factor exponent
    class_weights = [
        (1.0 / count) ** balance_factor if count > 0 else 0.0 
        for count in class_counts
    ]
   
    sample_weights = torch.FloatTensor(class_weights)
    return sample_weights


def print_metrics_report(metrics):
    """
    Nicely formats the hierarchical metrics dictionary into a readable report.
    Dynamically supports Standard, Forced, and Mixed Euclidean configurations.
    """
    # 1. Detect Active Configurations based on presence of Leaf_Acc keys
    configs = []
    if 'Leaf_Acc' in metrics: 
        configs.append(('Hyp', ''))
    if 'Forced_Leaf_Acc' in metrics: 
        configs.append(('Hyp(F)', 'Forced_'))
    if 'Euc_Leaf_Acc' in metrics: 
        configs.append(('Euc', 'Euc_'))
    if 'Forced_Euc_Leaf_Acc' in metrics: 
        configs.append(('Euc(F)', 'Forced_Euc_'))
    
    # Fallback if standard keys are missing
    if not configs: configs = [('Model', '')]

    # Calculate dynamic widths
    n_cols = len(configs)
    col_w = 10 if n_cols <= 2 else 8
    label_w = 22
    width = label_w + 3 + ((col_w + 3) * n_cols)
    width = max(width, 70) # Set a minimum width for aesthetics
    
    # --- INJECT GEOMETRIC STATE HERE ---
    if 'Mean_radius' in metrics:
        # Format the string cleanly
        rad_str = f"Lorentz Radius -> Mean: {metrics['Mean_radius']:.2f}"
        if 'Max_radius' in metrics:
            rad_str += f"  |  Max: {metrics['Max_radius']:.2f}"
        
        # Print it centered, just like the title
        print(f"{rad_str:^{width}}")
        print("-" * width) # Add a light separator before the table headers
    
    print("\n" + "="*width)
    print(f"{'HIERARCHICAL PERFORMANCE REPORT':^{width}}")
    headers = " | ".join([f"{h:^{col_w}}" for h, _ in configs])
    print(f"{'':<{label_w}} | {headers}")
    print("="*width)

    # --- Helper Functions ---
    def fmt_val(key, is_pct=True):
        """Formats a row of values for the active configurations."""
        vals = []
        for _, prefix in configs:
            val = metrics.get(f"{prefix}{key}")
            if val is None:
                vals.append(f"{'-':^{col_w}}")
            else:
                s = f"{val:.2%}" if is_pct else f"{val:.4f}"
                vals.append(f"{s:>{col_w}}")
        return " | ".join(vals)

    def fmt_val_inline(key, is_pct=True):
        """Formats an inline string like 85% / 88% / 80% for tight tables."""
        vals = []
        for _, prefix in configs:
            val = metrics.get(f"{prefix}{key}")
            if val is None:
                vals.append("-")
            else:
                vals.append(f"{val*100:.0f}%" if is_pct else f"{val:.2f}")
        return "/".join(vals)

    def get_base_labels(target_substr):
        """Extracts unique class labels, stripping all active prefixes."""
        labels = set()
        for k in metrics.keys():
            base_k = k
            for _, p in configs:
                if p and base_k.startswith(p):
                    base_k = base_k[len(p):]
                    break # Only strip one prefix
            if base_k.startswith(target_substr):
                labels.add(base_k[len(target_substr):])
        return labels

    # --- SECTION 1: GLOBAL (LEAF) METRICS ---
    print(f"\n{'[ GLOBAL / LEAF LEVEL ]':<{width}}")
    print("-" * width)
    
    global_keys = ['Leaf_Acc', 'Leaf_HED', 'Avg_Radius']
    for k in global_keys:
        # Only print if at least one config has this metric
        if any(f"{p}{k}" in metrics for _, p in configs):
            is_pct = 'Acc' in k
            print(f"{k.replace('_', ' '):<{label_w}} | {fmt_val(k, is_pct)}")

    # Prepare inline headers for complex tables
    inline_hdr = "/".join([h.replace("yp", "").replace("uc", "") for h, _ in configs])
    inl_w = max(len(inline_hdr), n_cols * 4 + (n_cols - 1)) # Approx width of "99%/99%/99%"

    # --- SECTION 2: LEVEL 1 (MALIGNANCY) ---
    print(f"\n{'[ LEVEL 1: MALIGNANCY STATUS ]':<{width}}")
    print("-" * width)
    
    if any(f"{p}L1_Malignancy_BalAcc" in metrics for _, p in configs):
        print(f"{'Balanced Accuracy':<{label_w}} | {fmt_val('L1_Malignancy_BalAcc', True)}\n")
        
    l1_labels = sorted(get_base_labels('L1_Sens_') | get_base_labels('L1_Prec_') | get_base_labels('L1_Spec_'))

    if l1_labels:
        print(f"{'Malignancy Class':<18} | {'Sens '+inline_hdr:<{inl_w}} | {'Prec '+inline_hdr:<{inl_w}} | {'Spec '+inline_hdr:<{inl_w}}")
        print("-" * (27 + inl_w * 3))
        for label in l1_labels:
            sens = fmt_val_inline(f'L1_Sens_{label}')
            prec = fmt_val_inline(f'L1_Prec_{label}')
            spec = fmt_val_inline(f'L1_Spec_{label}')
            print(f"{label:<18} | {sens:<{inl_w}} | {prec:<{inl_w}} | {spec:<{inl_w}}")

    # --- SECTION 3: LEVEL 2 (DIAGNOSIS FAMILY) ---
    print(f"\n{'[ LEVEL 2: DIAGNOSIS FAMILY ]':<{width}}")
    print("-" * width)
    
    if any(f"{p}L2_Family_BalAcc" in metrics for _, p in configs):
        print(f"{'Balanced Accuracy':<{label_w}} | {fmt_val('L2_Family_BalAcc', True)}\n")
        
    l2_labels = sorted(get_base_labels('L2_Sens_') | get_base_labels('L2_Prec_') | get_base_labels('L2_Spec_'))

    if l2_labels:
        print(f"{'Diagnosis Class':<18} | {'Sens '+inline_hdr:<{inl_w}} | {'Prec '+inline_hdr:<{inl_w}} | {'Spec '+inline_hdr:<{inl_w}}")
        print("-" * (27 + inl_w * 3))
        for label in l2_labels:
            sens = fmt_val_inline(f'L2_Sens_{label}')
            prec = fmt_val_inline(f'L2_Prec_{label}')
            spec = fmt_val_inline(f'L2_Spec_{label}')
            print(f"{label:<18} | {sens:<{inl_w}} | {prec:<{inl_w}} | {spec:<{inl_w}}")

    # --- SECTION 4: REFUSAL RATES ---
    refusal_labels = sorted(get_base_labels('Refusal_Rate_'))
    if refusal_labels:
        print(f"\n{'[ REFUSAL RATES ]':<{width}}")
        print("-" * width)
        # Refusal rates only make sense for unforced configurations, but fmt_val handles missing/zeros safely
        for label in refusal_labels:
            print(f" > {label:<{label_w-3}} | {fmt_val(f'Refusal_Rate_{label}', True)}")

    print("\n" + "="*width + "\n")


def print_metrics_report_old(metrics):
    """
    Nicely formats the hierarchical metrics dictionary into a readable report.
    Automatically detects mixed-mode and prints Hyperbolic vs Euclidean side-by-side.
    """
    is_mixed = any(k.startswith('Euc_') for k in metrics.keys())
    width = 85 if is_mixed else 65
    
    print("\n" + "="*width)
    print(f"{'HIERARCHICAL PERFORMANCE REPORT':^{width}}")
    if is_mixed:
        print(f"{'( HYPERBOLIC vs EUCLIDEAN )':^{width}}")
    print("="*width)

    # Helper function to format dual columns
    def fmt_val(key, is_pct=True):
        val_h = metrics.get(key)
        val_e = metrics.get(f"Euc_{key}")
        
        str_h = (f"{val_h:.2%}" if is_pct else f"{val_h:.4f}") if val_h is not None else "-"
        str_e = (f"{val_e:.2%}" if is_pct else f"{val_e:.4f}") if val_e is not None else "-"
        
        if is_mixed:
            return f"{str_h:>7} | {str_e:<7}"
        return f"{str_h:<10}"

    col_w = 17 if is_mixed else 10

    # --- SECTION 1: GLOBAL (LEAF) METRICS ---
    print(f"\n{'[ GLOBAL / LEAF LEVEL ]':<{width}}")
    print("-" * width)
    
    if is_mixed:
        print(f"{'Metric':<25} | {'Hyp':>7} | {'Euc':<7}")
        print("-" * 45)
        
    global_keys = ['Leaf_Acc', 'Leaf_HED', 'Avg_Radius']
    for k in global_keys:
        if k in metrics or f"Euc_{k}" in metrics:
            is_pct = 'Acc' in k
            print(f"{k.replace('_', ' '):<25} | {fmt_val(k, is_pct)}")

    # --- Helper to extract labels stripping Euc_ prefix ---
    def get_labels(prefix):
        labels = set()
        for k in metrics.keys():
            base_k = k.replace('Euc_', '')
            if base_k.startswith(prefix): 
                labels.add(base_k.replace(prefix, ''))
        return labels # <-- Return the set directly, do not sort here

    # --- SECTION 2: LEVEL 1 (MALIGNANCY) ---
    print(f"\n{'[ LEVEL 1: MALIGNANCY STATUS ]':<{width}}")
    print("-" * width)
    
    if 'L1_Malignancy_BalAcc' in metrics or 'Euc_L1_Malignancy_BalAcc' in metrics:
        print(f"{'Balanced Accuracy':<25} | {fmt_val('L1_Malignancy_BalAcc', True)}\n")
        
    # Sort the combined sets here instead
    l1_labels = sorted(get_labels('L1_Sens_') | get_labels('L1_Prec_') | get_labels('L1_Spec_'))

    if l1_labels:
        s_hdr = "Sens (H | E)" if is_mixed else "Sens"
        p_hdr = "Prec (H | E)" if is_mixed else "Prec"
        sp_hdr = "Spec (H | E)" if is_mixed else "Spec"
        
        print(f"{'Malignancy Class':<20} | {s_hdr:<{col_w}} | {p_hdr:<{col_w}} | {sp_hdr:<{col_w}}")
        print("-" * (29 + col_w * 3))
        
        for label in sorted(l1_labels):
            sens = fmt_val(f'L1_Sens_{label}', True)
            prec = fmt_val(f'L1_Prec_{label}', True)
            spec = fmt_val(f'L1_Spec_{label}', True)
            print(f"{label:<20} | {sens:<{col_w}} | {prec:<{col_w}} | {spec:<{col_w}}")

    # --- SECTION 3: LEVEL 2 (DIAGNOSIS FAMILY) ---
    print(f"\n{'[ LEVEL 2: DIAGNOSIS FAMILY ]':<{width}}")
    print("-" * width)
    
    if 'L2_Family_BalAcc' in metrics or 'Euc_L2_Family_BalAcc' in metrics:
        print(f"{'Balanced Accuracy':<25} | {fmt_val('L2_Family_BalAcc', True)}\n")
        
    # Make sure to apply the same fix to Level 2!
    l2_labels = sorted(get_labels('L2_Sens_') | get_labels('L2_Prec_') | get_labels('L2_Spec_'))

    if l2_labels:
        s_hdr = "Sens (H | E)" if is_mixed else "Sens"
        p_hdr = "Prec (H | E)" if is_mixed else "Prec"
        sp_hdr = "Spec (H | E)" if is_mixed else "Spec"
        
        print(f"{'Diagnosis Class':<20} | {s_hdr:<{col_w}} | {p_hdr:<{col_w}} | {sp_hdr:<{col_w}}")
        print("-" * (29 + col_w * 3))
        
        for label in sorted(l2_labels):
            sens = fmt_val(f'L2_Sens_{label}', True)
            prec = fmt_val(f'L2_Prec_{label}', True)
            spec = fmt_val(f'L2_Spec_{label}', True)
            print(f"{label:<20} | {sens:<{col_w}} | {prec:<{col_w}} | {spec:<{col_w}}")

    # --- SECTION 4: REFUSAL RATES ---
    # We only want to find the base keys, ignoring the Euc_ prefix for iteration
    refusal_keys = sorted(set(
        k.replace('Euc_', '') for k in metrics.keys() if 'Refusal_Rate_' in k
    ))
    
    if refusal_keys:
        print(f"\n{'[ REFUSAL RATES ]':<{width}}")
        print("-" * width)
        if is_mixed:
            print(f"   > {'Node':<20} | {'Hyp':>7} | {'Euc':<7}")
            print("   " + "-" * 42)
            
        for k in refusal_keys:
            label = k.replace('Refusal_Rate_', '')
            print(f"   > {label:<20} | {fmt_val(k, True)}")

    print("\n" + "="*width + "\n")


def get_frechet_mean_unsafe(lorentz_points, num_iterations=15, lr=1.0):
    """
    Calculates the true Riemannian Center of Mass (Fréchet Mean) 
    using Riemannian Gradient Descent.
    """
    # 1. Start with the Einstein midpoint as a highly educated first guess
    S = torch.sum(lorentz_points, dim=0)
    mink_sq = -S[0]**2 + torch.sum(S[1:]**2)
    mu = S / torch.sqrt(torch.clamp(-mink_sq, min=1e-6))
    
    for _ in tqdm.tqdm(range(num_iterations), desc = 'Calculating Freched Mean', total=num_iterations):
        # 2. Minkowski Dot Product between mu and all points
        dots = -mu[0] * lorentz_points[:, 0] + torch.sum(mu[1:] * lorentz_points[:, 1:], dim=1)
        dots = torch.clamp(dots, max=-1.0 - 1e-6)
        
        # 3. Hyperbolic Distances
        dists = torch.acosh(-dots)
        
        # 4. Logarithmic Map: Project points onto the tangent space at mu
        # Tangent vector = x_i + <x_i, mu>_L * mu
        tangents = lorentz_points + dots.unsqueeze(1) * mu.unsqueeze(0)
        
        # Scale the tangent vectors by (d / sinh(d))
        sinh_d = torch.sinh(dists)
        scales = torch.where(dists < 1e-5, torch.ones_like(dists), dists / sinh_d)
        log_vecs = scales.unsqueeze(1) * tangents
        
        # 5. Average the tangent vectors to find the gradient direction
        grad = torch.mean(log_vecs, dim=0) * lr
        
        # 6. Exponential Map: Move mu along the gradient
        norm_grad_sq = -grad[0]**2 + torch.sum(grad[1:]**2)
        if norm_grad_sq < 1e-7:
            break # We have converged exactly on the center!
            
        norm_grad = torch.sqrt(norm_grad_sq)
        mu = torch.cosh(norm_grad) * mu + torch.sinh(norm_grad) * (grad / norm_grad)
        
    return mu


def get_frechet_mean(lorentz_points, num_iterations=15, lr=1.0):
    """
    Calculates the true Riemannian Center of Mass with float64 precision 
    to prevent catastrophic overflow at the manifold boundaries.
    """
    # SAFEGUARD 1: Trivial case of a single point
    if len(lorentz_points) == 1:
        return lorentz_points[0].clone()
        
    # SAFEGUARD 2: Cast to float64 for internal hyperbolic math
    # This prevents t1*t2 from overflowing float32's 10^38 limit
    pts = lorentz_points.to(torch.float64)
    
    # 1. Start with the Einstein midpoint as the first guess
    S = torch.sum(pts, dim=0)
    mink_sq = -S[0]**2 + torch.sum(S[1:]**2)
    mu = S / torch.sqrt(torch.clamp(-mink_sq, min=1e-12))
    
    for _ in range(num_iterations):
        # Calculate distances
        dots = -mu[0] * pts[:, 0] + torch.sum(mu[1:] * pts[:, 1:], dim=1)
        dots = torch.clamp(dots, max=-1.0 - 1e-12)
        dists = torch.acosh(-dots)
        
        # Logarithmic Map
        tangents = pts + dots.unsqueeze(1) * mu.unsqueeze(0)
        
        sinh_d = torch.sinh(dists)
        scales = torch.where(dists < 1e-7, torch.ones_like(dists), dists / sinh_d)
        log_vecs = scales.unsqueeze(1) * tangents
        
        # Average to get raw gradient
        grad = torch.mean(log_vecs, dim=0) * lr
        
        # Tangent Snap
        grad_dot_mu = -grad[0]*mu[0] + torch.sum(grad[1:]*mu[1:])
        grad = grad + grad_dot_mu * mu
        
        norm_grad_sq = -grad[0]**2 + torch.sum(grad[1:]**2)
        if norm_grad_sq < 1e-12:
            break # Converged
            
        # Exponential Map step
        norm_grad = torch.sqrt(torch.clamp(norm_grad_sq, min=1e-16))
        mu = torch.cosh(norm_grad) * mu + torch.sinh(norm_grad) * (grad / norm_grad)
        
        # Manifold Snap
        t_new = torch.sqrt(1.0 + torch.sum(mu[1:]**2)).unsqueeze(0)
        mu = torch.cat([t_new, mu[1:]])

    # SAFEGUARD 3: Fallback if math completely breaks down
    if torch.isnan(mu).any():
        print("Warning: Fréchet Mean returned NaN despite float64. Falling back to base point.")
        return lorentz_points[0] # Return a safe, valid point from the cluster
        
    # Cast back to float32 to match the rest of your pipeline
    return mu.to(torch.float32)


def generate_l1_mapping(class_names, hierarchy_tree):
    """
    Generates a mapping from fine-grained dataset class names to their 
    Level 1 diagnostic category (e.g., 'Malign' or 'Benign').
    
    Args:
        class_names (list or dict_keys): The exact class names from the dataset.
        hierarchy_tree (dict): The parsed JSON taxonomy tree.
        
    Returns:
        dict: A mapping { 'nodular melanoma': 'Malign', 'compound nevus': 'Benign', ... }
    """
    # 1. Handle the Root Node Wrapper (if it exists)
    root_keys = list(hierarchy_tree.keys())
    if len(root_keys) == 1 and root_keys[0].lower().strip() == 'skin lesion':
        l1_nodes = hierarchy_tree[root_keys[0]]
    else:
        l1_nodes = hierarchy_tree

    # 2. Build the Universal Translation Map
    translation_map = {}
    
    def map_descendants(node, target_label):
        if isinstance(node, dict):
            for key, children in node.items():
                clean_key = str(key).lower().strip()
                translation_map[clean_key] = target_label
                map_descendants(children, target_label)

    # Walk through the Level 1 categories (Malign, Benign, etc.)
    for l1_name, descendants in l1_nodes.items():
        clean_l1_name = str(l1_name).lower().strip()
        formatted_l1_label = clean_l1_name.capitalize() # e.g., 'malign' -> 'Malign'
        
        # Map the L1 node to itself
        translation_map[clean_l1_name] = formatted_l1_label
        
        # Recursively map all nested sub-classes to this L1 label
        map_descendants(descendants, formatted_l1_label)

    # 3. Filter for the requested Dataset Classes
    final_mapping = {}
    missing_classes = []
    
    for name in class_names:
        clean_name = str(name).lower().strip()
        if clean_name in translation_map:
            # We map the EXACT original string from the dataset to the L1 label
            final_mapping[name] = translation_map[clean_name]
        elif clean_name == 'skin lesion':
            final_mapping[name] = 'Benign'
        else:
            final_mapping[name] = 'Other'
            missing_classes.append(name)
    # Diagnostic warning if something didn't match
    if missing_classes:
        print(f"Warning: {len(missing_classes)} classes were not found in the hierarchy tree")
        print(f"They have been mapped to 'Other': {missing_classes}")

    return final_mapping

def generate_l1_mapping_from_dataset_and_file(dataset, hierachy_tree_path):

    with open(hierachy_tree_path, 'r', encoding='utf-8') as f:
        hierachy = json.load(f)
    
    return generate_l1_mapping(dataset.class_names, hierachy)