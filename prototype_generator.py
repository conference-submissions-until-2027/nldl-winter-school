import os
import torch
import numpy as np
import math
import pickle
import json
import sys
# Add source path if needed
sys.path.append('hyperbolic/src')

from misc.create_diagnostic_tree import TAXONOMY_TREE, SkinConditions, BaseDataframeHandler

class HyperbolicPrototypeGenerator:
    """
    Generates hyperbolic prototype embeddings for hierarchical classification 
    using the Lorentz model.
    """
    def __init__(self, embedding_dim=10, target_radius=5.0, angular_allocation_metric = 'log'):
        self.embedding_dim = embedding_dim
        self.target_radius = target_radius
        
        # State variables to hold generated data
        self.counts = {}
        self.prototype_vectors = {}
        self.prototype_meta = {}
        self.final_tree = {}
        self.angular_allocation_metric = angular_allocation_metric
    def _normalize_tree_keys(self, node):
        cleaned_node = {}
        for key, children in node.items():
            clean_key = str(key).lower().strip()
            clean_children = self._normalize_tree_keys(children)
            
            if clean_key in cleaned_node:
                for child_k, child_v in clean_children.items():
                    if child_k not in cleaned_node[clean_key]:
                        cleaned_node[clean_key][child_k] = child_v
                    else:
                        cleaned_node[clean_key][child_k].update(child_v)
            else:
                cleaned_node[clean_key] = clean_children
        return cleaned_node

    def _prune_tree(self, node, node_name):
        my_count = self.counts.get(node_name, 0)
        
        if not node: 
            if my_count == 0:
                return None, 0
            else:
                return {}, my_count

        cleaned_children = {}
        total_branch_sum = my_count
        
        for child_name, child_node in node.items():
            pruned_child, child_sum = self._prune_tree(child_node, child_name)
            if pruned_child is not None:
                cleaned_children[child_name] = pruned_child
                total_branch_sum += child_sum
                
        if total_branch_sum == 0:
            return None, 0
            
        return cleaned_children, total_branch_sum

    def _get_branch_weight(self, node, node_name):
        
        
        if self.angular_allocation_metric == 'log':
            if not node:
                c = self.counts.get(node_name, 0)
                return math.log(1 + c)
            return sum(self._get_branch_weight(child_node, child_name) 
                    for child_name, child_node in node.items())
        elif self.angular_allocation_metric == 'count':
            if not node:
                
                return 1.0 
            # If it's a parent, its weight is the sum of its leaves
            return sum(self._get_branch_weight(child_node, child_name) 
                    for child_name, child_node in node.items())

        elif self.angular_allocation_metric == 'equal':
            return 0    
        
        else:
            raise NotImplementedError("Branch weight not implemented for angular allocation metric", self.angular_allocation_metric)

    def _polar_to_lorentz(self, r_geo, theta):
        """
        Converts Intrinsic Polar Coordinates to Lorentz coordinates.
        t = cosh(r), x = sinh(r) * cos(theta), y = sinh(r) * sin(theta)
        """
        t = math.cosh(r_geo)
        r_spatial = math.sinh(r_geo)
        
        x = r_spatial * math.cos(theta)
        y = r_spatial * math.sin(theta)
        
        vec = torch.zeros(self.embedding_dim)
        vec[0] = t
        vec[1] = x
        vec[2] = y
        return vec

    def _get_max_depth(self, node, current_depth=0):
        if not node:
            return current_depth
        return max([self._get_max_depth(child, current_depth + 1) for child in node.values()])

    def _place_prototypes(self, node, name, depth, start_angle, end_angle, r_step):
        r = depth * r_step
        theta = (start_angle + end_angle) / 2
        
        if depth == 0:
            vec = torch.zeros(self.embedding_dim)
            vec[0] = 1.0
            r = 0.0 
        else:
            vec = self._polar_to_lorentz(r, theta)
            
        self.prototype_vectors[name] = vec
        self.prototype_meta[name] = {
            'r': r, 
            'theta': theta, 
            'depth': depth, 
            'count': self.counts.get(name, 0)
        }
        
        if not node: return

        child_weights = {k: self._get_branch_weight(v, k) for k, v in node.items()}
        total_weight = sum(child_weights.values())
        
        current_angle = start_angle
        for child_name, child_node in node.items():
            w = child_weights[child_name]
            fraction = w / total_weight if total_weight > 0 else 1.0 / len(node)
            angle_width = (end_angle - start_angle) * fraction
            
            self._place_prototypes(
                child_node, child_name, depth + 1, 
                current_angle, current_angle + angle_width, r_step
            )
            current_angle += angle_width

    def generate(self, raw_taxonomy_tree, raw_counts, root_key_override=None):
        """
        Processes the tree and counts, and calculates the hyperbolic embeddings.
        """
        self.counts = {key.lower().strip(): val for key, val in raw_counts.items()}
        taxonomy_tree = self._normalize_tree_keys(raw_taxonomy_tree)

        root_key = root_key_override
        if not root_key:
            root_key = "skin lesion" if "skin lesion" in taxonomy_tree else list(taxonomy_tree.keys())[0]

        clean_tree_inner, _ = self._prune_tree(taxonomy_tree[root_key], root_key)
        if clean_tree_inner is None:
            raise ValueError("Root node was pruned! Check your counts dictionary.")
            
        self.final_tree = {root_key: clean_tree_inner}

        # Calculate dynamic radius step based on tree depth
        max_depth = self._get_max_depth(self.final_tree[root_key])
        r_step = self.target_radius / max_depth if max_depth > 0 else 0

        self.prototype_vectors.clear()
        self.prototype_meta.clear()
        
        self._place_prototypes(self.final_tree[root_key], root_key, 0, 0, 2 * np.pi, r_step)
        
        return self.prototype_vectors, self.prototype_meta, self.final_tree

    
    def save(self, output_dir):
        """Saves the generated prototypes and hierarchy to the specified directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        sorted_keys = sorted(self.prototype_vectors.keys())
        sorted_dict = {k: self.prototype_vectors[k] for k in sorted_keys}

        prototype_save_path = os.path.join(output_dir, 'prototypes.pkl')
        hierarchy_save_path = os.path.join(output_dir, 'hierarchy_tree.json')
        with open(prototype_save_path, 'wb') as f:
            pickle.dump(sorted_dict, f)

        with open(hierarchy_save_path, 'w') as f:
            json.dump(self.final_tree, f, indent=4)
        return prototype_save_path, hierarchy_save_path

def generate_prototypes(embedding_dim=10, target_radius=5, outdir = "", angular_allocation_metric = 'log'):
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

    # 2. Initialize the generator with desired parameters
    generator = HyperbolicPrototypeGenerator(embedding_dim=embedding_dim, target_radius=target_radius, angular_allocation_metric = angular_allocation_metric)

    vectors, metadata, tree = generator.generate(
    raw_taxonomy_tree=TAXONOMY_TREE,
    raw_counts=skin_conds.diagnosis_counts
    )

    prototypes_save_path, hierarchy_save_path = generator.save(outdir)

    # 5. Verify
    print(f"Generated {len(vectors)} prototypes!")
    

    print("\n--- Generation Complete ---")
    print(f"Total Prototypes: {len(vectors)}")
    print(f"{'Diagnosis':<35} | {'Count':<6} | {'Radius':<5} | {'Wedge'}")
    print("-" * 70)

    check_list = ['skin lesion', 'melanoma','nevus','benign', 'malign',
    'dermal melanoma', 'primary dermal melanoma', 'unspecified nevus', 'nodular melanoma', 'compound nevus']

    for name in check_list:
        name_clean = name.lower().strip()
        if name_clean in metadata:
            m = metadata[name_clean]
            deg = np.degrees(m['theta'])
            # Show cosh(r) (Time component) to verify Lorentz scaling
            t_val = math.cosh(m['r'])
            print(f"{name:<35} | {m['count']:<6} | {m['r']:.2f}  | {deg:.1f}° (t={t_val:.1f})")
        else:
            print(f"{name:<35} | PRUNED")

    return prototypes_save_path, hierarchy_save_path

