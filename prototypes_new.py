import sys
import os
import torch
import numpy as np
import math
import pickle
import json

# Add source path if needed
sys.path.append('/home/pjtka/hyperbolic/src')

from misc.create_diagnostic_tree import TAXONOMY_TREE, SkinConditions, BaseDataframeHandler

# --- 1. CONFIGURATION & PATHS ---

EMBEDDING_DIM = 10 

# Paths
image_path = '/scratch/pjtka/meladata-24112025/11222025-dump/2025-11-22_00.48.38_images'
training_assesment_path = '/scratch/pjtka/meladata-24112025/11222025-dump/training-assessments-112125.tsv'
training_skin_conds_path = '/scratch/pjtka/meladata-24112025/11222025-dump/training-skin-conditions-112125.tsv'
diagnosis_path = '/home/pjtka/hyperbolic/dermloopapi_public_dermloop_diagnosis.csv'

output_base_dir = '/scratch/pjtka/hyperbolic/meta-data'
os.makedirs(output_base_dir, exist_ok=True)
prototypes_save_path = os.path.join(output_base_dir, 'prototypes.pkl')
hierarchy_save_path = os.path.join(output_base_dir, 'hierarchy_tree.json')

# --- 2. LOAD DATA ---

print("Loading data...")
diagnosis_handler = BaseDataframeHandler(diagnosis_path)
skin_conds = SkinConditions(training_skin_conds_path, diagnosis_handler)
raw_counts = skin_conds.diagnosis_counts
raw_taxonomy_tree = TAXONOMY_TREE 

# --- 3. NORMALIZATION FUNCTIONS ---

def normalize_tree_keys(node):
    cleaned_node = {}
    for key, children in node.items():
        clean_key = str(key).lower().strip()
        clean_children = normalize_tree_keys(children)
        
        if clean_key in cleaned_node:
            for child_k, child_v in clean_children.items():
                if child_k not in cleaned_node[clean_key]:
                    cleaned_node[clean_key][child_k] = child_v
                else:
                    cleaned_node[clean_key][child_k].update(child_v)
        else:
            cleaned_node[clean_key] = clean_children
    return cleaned_node

print("Normalizing taxonomy and counts...")
taxonomy_tree = normalize_tree_keys(raw_taxonomy_tree)
counts = {key.lower().strip(): val for key, val in raw_counts.items()}

# --- 4. PRUNING ALGORITHM (FIXED) ---

def prune_tree(node, node_name):
    """
    Returns: (cleaned_node_dict, total_branch_count)
    If node should be pruned, returns (None, 0).
    """
    my_count = counts.get(node_name, 0)
    
    # Base Case: Leaf
    if not node: 
        if my_count == 0:
            return None, 0
        else:
            return {}, my_count

    # Recursive Step
    cleaned_children = {}
    total_branch_sum = my_count
    
    for child_name, child_node in node.items():
        pruned_child, child_sum = prune_tree(child_node, child_name)
        
        # ERROR FIX: Only add if not None
        if pruned_child is not None:
            cleaned_children[child_name] = pruned_child
            total_branch_sum += child_sum
            
    # Prune Dead Branches (Nodes with 0 count and 0 active children)
    if total_branch_sum == 0:
        return None, 0
        
    return cleaned_children, total_branch_sum

print("Pruning Tree...")
if "skin lesion" not in taxonomy_tree:
    root_key = list(taxonomy_tree.keys())[0]
else:
    root_key = "skin lesion"

clean_tree_inner, total_data = prune_tree(taxonomy_tree[root_key], root_key)
# Safety check if root got pruned (unlikely but possible)
if clean_tree_inner is None:
    raise ValueError("Root node was pruned! Check your counts dictionary.")
final_tree = {root_key: clean_tree_inner}


# --- 5. WEIGHT CALCULATION (FIXED LOGIC) ---

def get_branch_weight(node, node_name):
    """
    Returns the 'importance' of a branch for angular allocation.
    Uses log(1 + N) of the leaves to handle imbalance.
    """
    # If it's a leaf, weight is its own log count
    if not node:
        c = counts.get(node_name, 0)
        return math.log(1 + c)
    
    # If it's a node, weight is sum of children's weights
    # (This ensures space is reserved for the leaves below)
    return sum(get_branch_weight(child_node, child_name) for child_name, child_node in node.items())

# --- 6. PLACEMENT (LORENTZ LOGIC FIXED) ---

prototype_vectors = {}
prototype_meta = {}

def polar_to_lorentz(r_geo, theta, dim):
    """
    Converts Intrinsic Polar Coordinates (Geodesic Distance r, Angle theta)
    directly to Lorentz coordinates.
    
    Math:
    t = cosh(r)
    x = sinh(r) * cos(theta)
    y = sinh(r) * sin(theta)
    ... other dims 0 ...
    """
    # 1. Time Component
    t = math.cosh(r_geo)
    
    # 2. Spatial Magnitude
    r_spatial = math.sinh(r_geo)
    
    # 3. Direction
    x = r_spatial * math.cos(theta)
    y = r_spatial * math.sin(theta)
    
    # 4. Assemble Vector
    vec = torch.zeros(dim)
    vec[0] = t
    vec[1] = x
    vec[2] = y
    # Dimensions 3-9 remain 0.0 (planar embedding)
    
    return vec

def place_prototypes(node, name, depth, start_angle, end_angle):
    # --- SCALING FIX: Target Radius = 5.0 ---
    # Max depth is approx 4. So step = 5.0 / 4.0 = 1.25
    r_step = 1.25 
    r = depth * r_step
    
    # Angle Center
    theta = (start_angle + end_angle) / 2
    
    # Store Prototype
    if depth == 0:
        # Origin in Lorentz is [1, 0, ...]
        vec = torch.zeros(EMBEDDING_DIM); vec[0] = 1.0
        r = 0.0 # Force numeric 0
    else:
        vec = polar_to_lorentz(r, theta, EMBEDDING_DIM)
        
    prototype_vectors[name] = vec
    prototype_meta[name] = {'r': r, 'theta': theta, 'depth': depth, 'count': counts.get(name,0)}
    
    if not node: return

    # Calculate weights for allocation
    child_weights = {k: get_branch_weight(v, k) for k,v in node.items()}
    total_weight = sum(child_weights.values())
    
    current_angle = start_angle
    
    for child_name, child_node in node.items():
        w = child_weights[child_name]
        
        # Avoid division by zero
        fraction = w / total_weight if total_weight > 0 else 1.0/len(node)
        
        angle_width = (end_angle - start_angle) * fraction
        
        place_prototypes(child_node, child_name, depth + 1, current_angle, current_angle + angle_width)
        current_angle += angle_width

# Execute Placement
print(f"Generating Prototypes (Max Radius ~ 5.0)...")
place_prototypes(final_tree[root_key], root_key, 0, 0, 2 * np.pi)

# --- 7. SAVE OUTPUTS ---

print(f"Saving sorted prototypes to {prototypes_save_path}...")
# Sort to ensure tensor alignment matches alphanumeric class_to_idx
sorted_keys = sorted(prototype_vectors.keys())
sorted_dict = {k: prototype_vectors[k] for k in sorted_keys}

with open(prototypes_save_path, 'wb') as f:
    pickle.dump(sorted_dict, f)

print(f"Saving hierarchy tree to {hierarchy_save_path}...")
with open(hierarchy_save_path, 'w') as f:
    json.dump(final_tree, f, indent=4)

# --- 8. VERIFICATION ---

print("\n--- Generation Complete ---")
print(f"Total Prototypes: {len(prototype_vectors)}")
print(f"{'Diagnosis':<35} | {'Count':<6} | {'Radius':<5} | {'Wedge'}")
print("-" * 70)

check_list = ['skin lesion', 'melanoma','nevus','benign', 'malign',
 'dermal melanoma', 'primary dermal melanoma', 'unspecified nevus', 'nodular melanoma', 'compound nevus']

for name in check_list:
    name_clean = name.lower().strip()
    if name_clean in prototype_meta:
        m = prototype_meta[name_clean]
        deg = np.degrees(m['theta'])
        # Show cosh(r) (Time component) to verify Lorentz scaling
        t_val = math.cosh(m['r'])
        print(f"{name:<35} | {m['count']:<6} | {m['r']:.2f}  | {deg:.1f}° (t={t_val:.1f})")
    else:
        print(f"{name:<35} | PRUNED")