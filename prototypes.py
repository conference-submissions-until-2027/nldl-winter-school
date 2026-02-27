
import sys

import torch
import numpy as np
import os 

from misc.create_diagnostic_tree import TAXONOMY_TREE, SkinConditions, BaseDataframeHandler
import torch
import numpy as np
import math

# --- 1. INPUTS ---

EMBEDDING_DIM = 10 

'melanoma-nevus-meta-data.csv')

diagnosis = BaseDataframeHandler(diagnosis_path)
skin_conds = SkinConditions(training_skin_conds_path, diagnosis)
counts = skin_conds.diagnosis_counts

# Your Hierarchy (Reconstructed from your list logic)
# Note: I've mapped the flat list back to the likely structure
taxonomy_tree = TAXONOMY_TREE 


import math

# --- 1. NORMALIZATION FUNCTIONS ---

def normalize_tree_keys(node):
    """
    Recursively converts all keys in a nested dictionary to lowercase/stripped.
    Handles collisions by merging children (e.g. 'Melanoma' and 'melanoma' become one).
    """
    cleaned_node = {}
    
    for key, children in node.items():
        # Clean the key
        clean_key = str(key).lower().strip()
        
        # Recursively clean the children
        clean_children = normalize_tree_keys(children)
        
        # If key already exists (collision), merge the new children into existing ones
        if clean_key in cleaned_node:
            # We use a helper or simple update. Since clean_children is a dict, 
            # .update works, but deep merging is safer if the structure is complex.
            # For strict taxonomies, a shallow update of the children dict is usually sufficient 
            # assuming keys at the next level don't collide in a way that overwrites leaf vs branch.
            # Here we do a recursive merge to be safe:
            for child_k, child_v in clean_children.items():
                if child_k not in cleaned_node[clean_key]:
                    cleaned_node[clean_key][child_k] = child_v
                else:
                    # If this child also exists, we'd theoretically need to recurse deeper,
                    # but for this specific step, .update() is usually acceptable 
                    # unless you have the exact same sub-diagnosis defined twice.
                    cleaned_node[clean_key].update(clean_children)
        else:
            cleaned_node[clean_key] = clean_children
            
    return cleaned_node


taxonomy_tree = normalize_tree_keys(taxonomy_tree)
counts = {key.lower().strip(): val for key, val in counts.items()}
breakpoint()
# --- 2. PRUNING ALGORITHM ---

def prune_tree(node, node_name):
    """
    Recursively removes nodes that have 0 count AND no active children.
    Returns: (cleaned_node, total_branch_count)
    """
    # Get current node count (default to 0 if missing)
    my_count = counts.get(node_name, 0)
 
    if not node: # It's a leaf
        if my_count == 0:
            return None, 0 # Cut this leaf
        else:
            return {}, my_count # Keep this leaf

    # Process children
    cleaned_children = {}
    total_branch_sum = my_count
    
    for child_name, child_node in node.items():
        pruned_child, child_sum = prune_tree(child_node, child_name)
        
        # If child exists (has count) OR has active sub-children
        if child_sum > 0:
            cleaned_children[child_name] = pruned_child
            total_branch_sum += child_sum
            
    # Decision: Do we kill this parent?
    # Only kill if itself is 0 AND all children are gone.
    if total_branch_sum == 0:
        return None, 0
        
    return cleaned_children, total_branch_sum

print("Pruning Tree...")
clean_tree, total_data = prune_tree(taxonomy_tree["skin lesion"], "skin lesion")
# Wrap it back in root
final_tree = {"skin lesion": clean_tree}


# --- 3. WEIGHT CALCULATION (LOGARITHMIC) ---

def get_weight(name):
    c = counts.get(name, 0)
    # Log smoothing: log(1 + count)
    # +1 ensures even count=1 gets some weight
    # Adding a small base constant (e.g. 0.5) ensures rare classes don't vanish
    return math.log(1 + c) + 0.5 

def get_branch_weight(node):
    """Sum of log-weights for a branch (used for angle allocation)"""
    if not node: return 1.0 # Base weight for a leaf
    
    return sum(get_branch_weight(child) for child in node.values())


# --- 4. PLACEMENT (SAME LORENTZ LOGIC) ---

prototype_vectors = {}
prototype_meta = {}

def polar_to_lorentz(r, theta, dim):
    x = r * np.cos(theta); y = r * np.sin(theta)
    denom = 1 - r**2 + 1e-6
    x0 = (1 + r**2) / denom; x1 = (2 * x) / denom; x2 = (2 * y) / denom
    vec = torch.zeros(dim)
    vec[0] = x0; vec[1] = x1; vec[2] = x2
    return vec

def place_prototypes(node, name, depth, start_angle, end_angle):
    # Radius
    r_step = 0.9 / 4.0 
    r = depth * r_step
    if depth == 0: r = 0.0
    
    # Angle Center
    theta = (start_angle + end_angle) / 2
    
    # Store
    if depth == 0:
        vec = torch.zeros(EMBEDDING_DIM); vec[0] = 1.0
    else:
        vec = polar_to_lorentz(r, theta, EMBEDDING_DIM)
        
    prototype_vectors[name] = vec
    prototype_meta[name] = {'r': r, 'theta': theta, 'depth': depth, 'count': counts.get(name,0)}
    
    if not node: return

    # Calculate total weight of this specific layer's children
    # (We use get_branch_weight which sums up the tree logic)
    child_weights = {k: get_branch_weight(v) for k,v in node.items()}
    total_weight = sum(child_weights.values())
    
    current_angle = start_angle
    
    for child_name, child_node in node.items():
        w = child_weights[child_name]
        
        # Allocate wedge proportional to Log-Weight
        fraction = w / total_weight
        angle_width = (end_angle - start_angle) * fraction
        
        place_prototypes(child_node, child_name, depth + 1, current_angle, current_angle + angle_width)
        current_angle += angle_width

# Execute
place_prototypes(final_tree["skin lesion"], "skin lesion", 0, 0, 2 * np.pi)

# --- 5. RESULTS ---

print(f"{'Diagnosis':<35} | {'Count':<6} | {'Radius':<5} | {'Wedge'}")
print("-" * 70)
# Check specific interesting nodes
check_list = ['skin lesion', 'melanoma', 'dermal melanoma', 'primary dermal melanoma', 'unspecified nevus', 'nodular melanoma', 'compound nevus']

for name in check_list:
    if name in prototype_meta:
        m = prototype_meta[name]
        deg = np.degrees(m['theta'])
        print(f"{name:<35} | {m['count']:<6} | {m['r']:.2f}  | {deg:.1f}°")
    else:
        print(f"{name:<35} | PRUNED (0 count)")
