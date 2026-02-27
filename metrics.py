from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import json
import sys
sys.path.append('/home/pjtka/hyperbolic/src/misc')
from create_diagnostic_tree import TAXONOMY_TREE


def generate_hierarchy_maps(taxonomy_tree):
    """
    Parses TAXONOMY_TREE to generate INDEPENDENT flat mapping dictionaries.
    
    Fixes the 'Chaining' bug: 
    - leaf_to_fam: Maps nodes to their Level 2 Family (e.g., 'Nodular Melanoma' -> 'Melanoma')
    - leaf_to_mal: Maps nodes DIRECTLY to Level 1 Status (e.g., 'Nodular Melanoma' -> 'Malign')
      This ensures that if a model predicts 'Malign' (Level 1), it is correctly 
      scored as 'Malign' in L1 metrics, even if it has no Family.
    """
    leaf_to_fam = {}
    leaf_to_mal = {}
    
    # Helper to recursively map all descendants to a specific label
    def map_all_descendants(node, label, target_map):
        for key, children in node.items():
            clean_key = str(key).lower().strip()
            target_map[clean_key] = label
            map_all_descendants(children, label, target_map)

    # Handle Root Wrapper
    root_keys = list(taxonomy_tree.keys())
    if len(root_keys) == 1 and root_keys[0].lower() == 'skin lesion':
        root_node = taxonomy_tree[root_keys[0]]
    else:
        root_node = taxonomy_tree

    # --- TRAVERSAL ---
    for malignancy_status, families in root_node.items():
        clean_mal_status = malignancy_status.strip() # e.g. 'Malign'
        
        # 1. Map the Level 1 Node (Malignancy) itself
        # This fixes the bug: 'malign' -> 'Malign'
        leaf_to_mal[clean_mal_status.lower()] = clean_mal_status
        
        # 2. Map ALL descendants (Families + Subtypes) to this Malignancy Status
        map_all_descendants(families, clean_mal_status, leaf_to_mal)

        for family_name, subtypes in families.items():
            clean_family_name = family_name.strip() # e.g. 'Melanoma'
            
            # 3. Map Level 2 Node (Family) itself
            leaf_to_fam[clean_family_name.lower()] = clean_family_name
            
            # 4. Map Subtypes to this Family
            map_all_descendants(subtypes, clean_family_name, leaf_to_fam)

    return leaf_to_fam, leaf_to_mal


class HierarchicalEvaluator:
    def __init__(self, dataset_class_names, path_to_hierachy=""):
        self.class_names = dataset_class_names
        
        if path_to_hierachy:
            with open(path_to_hierachy, 'r', encoding='utf-8') as f:
                taxonomy = json.load(f)
        else:
            # Assuming TAXONOMY_TREE is imported globally or passed in
            from create_diagnostic_tree import TAXONOMY_TREE
            taxonomy = TAXONOMY_TREE
        
        # Generate decoupled maps
        self.family_map, self.malignancy_map = generate_hierarchy_maps(taxonomy)
        

    def _map_preds(self, leaf_indices, mapping_dict):
        """Converts list of leaf indices to mapped labels."""
        mapped_labels = []
        for idx in leaf_indices:
            # Handle string vs int index input safely
            if isinstance(idx, (int, np.integer)):
                leaf_name = self.class_names[idx].lower().strip()
            else:
                leaf_name = str(idx).lower().strip()

            mapped_label = mapping_dict.get(leaf_name, 'Other')
            mapped_labels.append(mapped_label)
            
        return np.array(mapped_labels)
    
    def calculate_refusal_rate(self, preds, targets, refusal_label="skin lesion"):
        """
        Calculates the percentage of samples for each True Family that were 
        predicted as the generic 'refusal_label' (e.g., 'skin lesion').
        
        Args:
            preds: List of predicted leaf indices
            targets: List of true leaf indices
            refusal_label: The string label representing the root/unknown class.
                           Must match the formatted class names (lowercase/stripped).
        """
        # 1. Map Targets to Families (so we can group by 'Melanoma', 'Nevus', etc.)
        true_families = self._map_preds(targets, self.family_map)
        
        # 2. Map Predictions to their raw names (to check for 'skin lesion')
        # We don't map to family here because we specifically want to catch the Root node
        pred_names = np.array([self.class_names[i].lower().strip() for i in preds])
        
        # 3. Identify Refusals
        # Boolean array: True where model predicted "skin lesion"
        is_refusal = (pred_names == refusal_label.lower().strip())
        
        results = {}
        unique_families = np.unique(true_families)
        
        for fam in unique_families:
            # Filter for samples that actually belong to this family
            fam_mask = (true_families == fam)
            total_samples = np.sum(fam_mask)
            
            if total_samples == 0:
                continue
                
            # Count how many of these were predicted as refusal_label
            refusal_count = np.sum(is_refusal & fam_mask)
            
            # Calculate Percentage
            results[f'Refusal_Rate_{fam}'] = refusal_count / total_samples
            
        return results

    def calculate_metrics(self, preds, targets):
        results = {}
        
        # --- HELPER: Compute per-class metrics from CM ---
        def compute_class_metrics(true_labels, pred_labels, prefix):
            # Get unique labels sorted to ensure alignment
            unique_labels = sorted(list(set(true_labels) | set(pred_labels)))
            cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                # 1. Basic Components
                tp = cm.diagonal()
                fp = cm.sum(axis=0) - tp
                fn = cm.sum(axis=1) - tp
                tn = cm.sum() - (tp + fp + fn)
                
                # 2. Sensitivity (Recall): TP / (TP + FN) -> Of actual cases, how many found?
                sensitivity = tp / (tp + fn)
                sensitivity = np.nan_to_num(sensitivity)
                
                # 3. Specificity: TN / (TN + FP) -> Of non-cases, how many correctly rejected?
                specificity = tn / (tn + fp)
                specificity = np.nan_to_num(specificity)
                
                # 4. Precision (PPV): TP / (TP + FP) -> Of predicted cases, how many were real?
                precision = tp / (tp + fp)
                precision = np.nan_to_num(precision)
            
            # Store Results
            metrics = {}
            for i, label in enumerate(unique_labels):
                metrics[f'{prefix}_Sens_{label}'] = sensitivity[i]
                metrics[f'{prefix}_Spec_{label}'] = specificity[i]
                metrics[f'{prefix}_Prec_{label}'] = precision[i]
                
            return metrics, unique_labels, cm

        # --- LEVEL 2: DIAGNOSIS FAMILY ---
        pred_fam = self._map_preds(preds, self.family_map)
        true_fam = self._map_preds(targets, self.family_map)
        
        results['L2_Family_BalAcc'] = balanced_accuracy_score(true_fam, pred_fam)
        
        fam_metrics, _, _ = compute_class_metrics(true_fam, pred_fam, prefix='L2')
        results.update(fam_metrics)

        # --- LEVEL 1: MALIGNANCY ---
        pred_mal = self._map_preds(preds, self.malignancy_map)
        true_mal = self._map_preds(targets, self.malignancy_map)
        
        results['L1_Malignancy_BalAcc'] = balanced_accuracy_score(true_mal, pred_mal)
        
        mal_metrics, _, _ = compute_class_metrics(true_mal, pred_mal, prefix='L1')
        results.update(mal_metrics)
        
        # --- REFUSAL RATES ---
        refusal_metrics = self.calculate_refusal_rate(preds, targets, refusal_label="skin lesion")
        results.update(refusal_metrics)
        
        return results