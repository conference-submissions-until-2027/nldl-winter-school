import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import geoopt
import networkx as nx
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import json
from tqdm import tqdm 
from typing import List
from metrics import HierarchicalEvaluator
from utils import print_metrics_report
from pathlib import Path
import pickle
import csv

ISIC_IMAGE_PATH = ""
ISIC_LABELS_PATH = ""

def load_isic_data(csv_path, data_dir):
    """
    Reads the ISIC 2019 ground truth CSV and returns full image paths and mapped labels.
    
    Args:
        csv_path (str): Path to the ISIC 2019 ground truth CSV file.
        data_dir (str): Path to the directory containing the ISIC images.
        
    Returns:
        tuple: (image_paths, labels) where both are lists of strings.
    """
    
    # Our flat mapping, including 'UNK' for the 9th column
    isic_2019_flat_map = {
        "MEL":  "melanoma",
        "BCC":  "basal cell carcinoma",
        "SCC":  "squamous cell carcinoma",
        "NV":   "nevus",
        "AK":   "actinic keratosis",
        "BKL":  "benign keratosis",
        "DF":   "dermatofibroma",
        "VASC": "vascular lesion",
        "UNK":  "unknown" 
    }

    image_paths = []
    labels = []

    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            # 1. Construct the full image path
            image_id = row['image']
            full_path = os.path.join(data_dir, f"{image_id}.jpg")
            image_paths.append(full_path)
            
            # 2. Find the one-hot encoded label (where the value is 1.0)
            current_label = "unknown"
            for isic_class, taxonomy_name in isic_2019_flat_map.items():
                # Checking for both '1.0' and '1' to be safe with CSV formatting
                if row.get(isic_class) in ['1.0', '1']:
                    current_label = taxonomy_name
                    break
            
            labels.append(current_label)

    return image_paths, labels


class BaseTrainingCallback:

    def __init__(self, action = None):
        
        self.action = None
        self.base_action = lambda: None
        
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
    

class TrainingBreakpointCallback(BaseTrainingCallback):

    def __init__(self,):
        super().__init__()

        self.action = breakpoint
    
    def __call__(self, tensors: List[torch.tensor]):

        if any(torch.any(torch.isnan(tens) for tens in tensors)):
            return self.action 
        return self.base_action
    

class BreakPointOnEpochCallback(BaseTrainingCallback):

    def __init__(self, action=None, target_epoch = None):
        super().__init__(action)

        self.action = breakpoint
        self.target_epoch = target_epoch
        
    def __call__(self, state):
        if state.get('epoch') == self.target_epoch:
            return self.action
        return self.base_action
    


class CallbackState:

    def __init__(self):
        self.state_dict = {}
    
    def update(self, key, val):
        self.state_dict[key] = val 

    def append(self, key, val):
        if not isinstance(self.state_dict[key], list):
            self.state_dict[key] = [self.state_dict[key]]
        self.state_dict[key].append(val)

    def get(self, key):
        return self.state_dict.get(key, None)
        
        
class SkinCancerTrainer:
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader, 
                 hierarchy_graph=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 output_dir='./output',
                 lr=1e-4,
                 callbacks = None,
                 path_to_hierachy = "",
                 hyperbolic = True,
                 weights = None, 
                 radius_penalty = 0.05,
                 radius_clamping_epochs = -1, 
                 radius_penalty_type = 'exponential'):


        self.weights = weights
        if isinstance(weights, torch.Tensor):
            self.weights = weights.to(device)
        
        self.radius_penalty = radius_penalty
        self.radius_clamping_epochs = radius_clamping_epochs
        self.radius_penalty_type = radius_penalty_type

        self.radius_threshold = 0.1
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.hierarchy_graph = hierarchy_graph
        self.lr = lr
        self.hyperbolic = hyperbolic
        self.path_to_hierachy = path_to_hierachy
        os.makedirs(output_dir, exist_ok=True)
        if callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = callbacks
        
        self.callback_state = CallbackState()

        n_classes = train_loader.dataset.num_classes
        
        device = next(self.model.parameters()).device
        self.prototypes_euc, self.prototypes_hyp,self.prototypes = None, None, None
        # --- GEOMETRY SPECIFIC SETUP ---
        if self.hyperbolic == "mixed":
            # 1. Hyperbolic: Fixed Prototypes from clinical taxonomy
            raw_protos = train_loader.dataset.get_prototypes().to(device)
            self.prototypes_hyp = geoopt.ManifoldTensor(raw_protos, manifold=self.model.manifold)
            
            # 2. Euclidean: Learnable Prototypes (Random Init on Hypersphere)
            # Match the euc_dim defined in the model (10 in our previous discussion)
            self.prototypes_euc = nn.Parameter(torch.randn(n_classes, self.model.euc_dim).to(device))
            nn.init.normal_(self.prototypes_euc, std=0.01) # Stable start
            
            # 3. Optimizer: Split Group
            # Includes model parameters, learnable prototypes, and the temperature scale
            self.optimizer = optim.AdamW([
                {'params': self.model.parameters(), 'lr': self.lr},
                {'params': [self.prototypes_euc], 'lr': self.lr * 10}
            ], weight_decay=1e-4)
            lorentz_origin = torch.zeros((1,raw_protos.shape[-1]), device='cuda')
            lorentz_origin[:, 0] = 1.0
            
            self.prototype_max_radius = self.model.manifold.dist(self.prototypes_hyp, lorentz_origin).max()

        elif self.hyperbolic: # Pure Hyperbolic
            raw_protos = train_loader.dataset.get_prototypes().to(device)
            self.prototypes = geoopt.ManifoldTensor(raw_protos, manifold=self.model.manifold)
            self.hyp_tau = nn.Parameter(torch.tensor(2.0).to(self.device))
            self.optimizer = optim.AdamW([
            {'params': self.model.parameters(), 'lr': self.lr},
            {'params': [self.hyp_tau], 'lr': self.lr} 
            ], weight_decay=1e-4)

            lorentz_origin = torch.zeros((1,raw_protos.shape[-1]), device='cuda')
            lorentz_origin[:, 0] = 1.0
            
            self.prototype_max_radius = self.model.manifold.dist(self.prototypes, lorentz_origin).max()
            
        else: # Pure Euclidean
            self.prototypes = nn.Parameter(torch.randn(n_classes, 10).to(device))
            self.optimizer = optim.AdamW([
                {'params': self.model.parameters(), 'lr': self.lr},
                {'params': [self.prototypes], 'lr': self.lr * 10} 
            ], weight_decay=1e-4)


        # 2. Optimizer
        # RiemannianAdam is strictly better for hyperbolic parameters, 
        # though standard AdamW often works if expmap is the last layer.
        # self.optimizer = geoopt.optim.RiemannianAdam(self.model.parameters(), lr=lr)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.lr)
        # # 3. Scheduler
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        
    
    def setup_hyperbolic(self, raw_protos):

        # --- GEOOPT SETUP ---
        # 1. Prototypes
        # We load them to the device and wrap them as a ManifoldTensor
        # This ensures operations like dist() recognize them correctly
        prototypes = geoopt.ManifoldTensor(raw_protos, manifold=self.model.manifold)
        return prototypes

    def loss(self, outputs, targets):
        """
        Args:
            outputs: For mixed, a dict {"hyp": ..., "euc": ..., "euc_scale": ...}
                     Otherwise, a raw Tensor (Pure Hyp or Pure Euc).
            targets: Class labels.
        """
        if self.hyperbolic == "mixed":
            # 1. Hyperbolic Branch (Fixed Taxonomy)
            # Use the dedicated hyp prototypes and embeddings
            l_hyp = self.hyperbolic_loss(outputs["hyp"], targets, mixed_mode=True)
            
            # 2. Euclidean Branch (Learned Nuance)
            # Use learned prototypes and the model's learnable scale
            l_euc = self.euclidian_loss(
                outputs["euc"], 
                targets, 
                mixed_mode=True, 
                scale=outputs["euc_scale"]
            )
            
            # 3. Weighted Sum (Alpha balances the two geometries)
            # alpha=0.5 is a safe starting point
            alpha = getattr(self, 'loss_alpha', 0.5)
            return (alpha * l_hyp) + ((1.0 - alpha) * l_euc)
        
        elif self.hyperbolic:
            return self.hyperbolic_loss(outputs, targets)
        else:
            return self.euclidian_loss(outputs, targets)

    def radius_penalty_update(self, epoch, max_epoch):
        
        if epoch < max_epoch / 20:
            self.radius_penalty = 0.15
        else:
            self.radius_penalty = 0.05

    def radius_threshold_update(self, epoch):
        
        if epoch > self.radius_clamping_epochs:
            self.radius_threshold = None

    def radius_penalty_loss(self, embeddings):

        base_penalty = 0.05
        
        origin = torch.zeros_like(embeddings)
        origin[:, 0] = 1.0 # Lorentz origin
        radii = self.model.manifold.dist(embeddings, origin)
        
        if self.radius_penalty_type == 'linear':
            radius_penalty = torch.mean(radii) * self.radius_penalty
            return radius_penalty
        
        if self.radius_penalty_type == 'exponential':
            exp_penalty = torch.exp(radii - self.prototype_max_radius)
            radius_penalty = base_penalty * torch.mean(torch.clamp(exp_penalty, max = 1e4))
            return radius_penalty
        
        return 0
      
        
    def hyperbolic_loss(self, embeddings, targets, mixed_mode=False):
        # Pick correct prototypes
        protos = self.prototypes_hyp if mixed_mode else self.prototypes
        
        # Calculate Distances
        dists = self.model.manifold.dist(embeddings.unsqueeze(1), protos.unsqueeze(0))
        
        # Clamp to prevent gradient explosion
        dists = torch.clamp(dists, max=100.0)
        safe_tau = torch.nn.functional.softplus(self.hyp_tau) + 1e-4
        logits = -dists / safe_tau
        
        # 3. Standard Cross Entropy
        loss = nn.functional.cross_entropy(logits, targets, self.weights, ignore_index=-1)
        # 4. Radius Penalty (The "Anti-Entanglement" mechanism)
        # We calculate the distance from the origin

        loss = loss + self.radius_penalty_loss(embeddings)

        return loss

    def euclidian_loss(self, embeddings, targets, mixed_mode=False, scale=30.0):
        # Pick correct prototypes
        protos = self.prototypes_euc if mixed_mode else self.prototypes
        
        # 1. Normalize
        emb_norm = nn.functional.normalize(embeddings, p=2, dim=1)
        proto_norm = nn.functional.normalize(protos, p=2, dim=1)
        
        # 2. Cosine Similarity
        logits = torch.matmul(emb_norm, proto_norm.T)
        
        # 3. Scaling (Use learned scale if provided, else fixed 30.0)
        logits = logits * scale
        
        return nn.functional.cross_entropy(logits, targets, self.weights, ignore_index=-1)

    def distance_matrix(self, embeddings, prototypes, mode='hyp'):
        """Helper to get distances for prediction."""
        if mode == 'hyp':
            return self.model.manifold.dist(embeddings.unsqueeze(1), prototypes.unsqueeze(0))
        else:
            emb_norm = nn.functional.normalize(embeddings, p=2, dim=1)
            proto_norm = nn.functional.normalize(prototypes, p=2, dim=1)
            sims = torch.matmul(emb_norm, proto_norm.T)
            return -sims # Argmin(-sim) = Argmax(sim)
    
    def calculate_hed(self, pred_indices, true_indices):
        """Calculates Hierarchical Error Distance using NetworkX."""
        if self.hierarchy_graph is None: return 0.0
        
        idx_to_name = {v: k for k, v in self.train_loader.dataset.label_map.items()}
        total_dist = 0
        
        for p_idx, t_idx in zip(pred_indices, true_indices):
            p_name = idx_to_name[p_idx]
            t_name = idx_to_name[t_idx]
            
            try:
                # Graph must have lower-case keys matching the dataset map
                d = nx.shortest_path_length(self.hierarchy_graph, source=p_name, target=t_name)
                total_dist += d
            except nx.NetworkXNoPath:
                total_dist += 5 # Max penalty
            except KeyError:
                pass # Node not in graph (pruned?)
                
        return total_dist / len(pred_indices)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(enumerate(self.train_loader), desc = f'Training epoch {epoch}', total=len(self.train_loader))
        for batch_idx, (images, labels) in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward (Output is on manifold)
            embeddings = self.model(images)
            # Loss
            loss = self.loss(embeddings, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def get_radius(self, embeddings):
        embeddings = embeddings.to(self.device)
        origin = torch.zeros_like(embeddings)
        origin[:, 0] = 1.0 # Lorentz origin
        radii = self.model.manifold.dist(embeddings, origin)
        return radii.mean().item(), radii.max().item()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        # Initialize the hierarchical evaluator
        hier_eval = HierarchicalEvaluator(
            self.train_loader.dataset.class_names, 
            self.path_to_hierachy
        )
        
        results = {
            "hyp_preds": [], "hyp_preds_forced": [],
            "euc_preds": [], "euc_preds_forced": [],
            "targets": [],
            'embeddings': []
        }
        
        # Get the mask of indices to ignore during forced fallback
        # (Assuming your dataset provides a list of non-leaf indices)
        non_leaf_indices = getattr(self.train_loader.dataset, 'non_leaf_indices', [])
        
        use_forced = self.hyperbolic in [True, 'mixed']
        # 1. Gather Predictions
        for images, labels in tqdm(self.val_loader, desc = 'Evaluating', total = len(self.val_loader)):
            images = images.to(self.device)
            outputs = self.model(images)
            results["targets"].extend(labels.numpy())

            # Helper function to compute standard and forced predictions
            def get_preds(dists):
                standard_preds = torch.argmin(dists, dim=1).cpu().numpy()
                
                if use_forced and non_leaf_indices:
                    dists_forced = dists.clone()
                    # Set distance to non-leaf nodes to infinity so they are never chosen
                    dists_forced[:, non_leaf_indices] = float('inf')
                    forced_preds = torch.argmin(dists_forced, dim=1).cpu().numpy()
                else:
                    forced_preds = standard_preds # Fallback if no indices provided
                    
                return standard_preds, forced_preds

            if self.hyperbolic == "mixed":
                # Branch 1: Hyperbolic
                d_hyp = self.distance_matrix(outputs["hyp"], self.prototypes_hyp, mode='hyp')
                p_hyp, p_hyp_f = get_preds(d_hyp)
                results["hyp_preds"].extend(p_hyp)
                results["hyp_preds_forced"].extend(p_hyp_f)
                
                # Branch 2: Euclidean
                d_euc = self.distance_matrix(outputs["euc"], self.prototypes_euc, mode='euc')
                p_euc, p_euc_f = get_preds(d_euc)
                results["euc_preds"].extend(p_euc)
                results["euc_preds_forced"].extend(p_euc_f)

                results['embeddings'].append(outputs[:, :3].detach().cpu())
                
            elif self.hyperbolic:
                d_hyp = self.distance_matrix(outputs, self.prototypes, mode='hyp')
                p_hyp, p_hyp_f = get_preds(d_hyp)
                results["hyp_preds"].extend(p_hyp)
                results["hyp_preds_forced"].extend(p_hyp_f)
                results['embeddings'].append(outputs[:, :3].detach().cpu())
                
            else:
                d_euc = self.distance_matrix(outputs, self.prototypes, mode='euc')
                p_euc, p_euc_f = get_preds(d_euc)
                results["euc_preds"].extend(p_euc)
                results["euc_preds_forced"].extend(p_euc_f)
                results['embeddings'].append(outputs.detach().cpu())
                

        targets = results["targets"]
        
        mapped_family_names = hier_eval._map_preds(targets, hier_eval.family_map)
        embeddings = {}
        
        embeddings = {
            'embeddings': torch.cat(results['embeddings'], dim = 0),
            'targets': targets,
            'class_names': mapped_family_names.tolist(),
            'class_name_mapping': self.val_loader.dataset.class_names,
            'prototypes': self.prototypes.detach().cpu()
        }

        final_metrics = {}
    
        # --- Helper to calculate and prefix metrics ---
        def compute_and_add_metrics(preds, targets, prefix=""):
            acc = accuracy_score(targets, preds)
            hed = self.calculate_hed(preds, targets)
            hier_metrics = hier_eval.calculate_metrics(preds, targets)
            
            metrics_dict = {f"{prefix}{k}": v for k, v in hier_metrics.items()}
            metrics_dict[f"{prefix}Leaf_Acc"] = acc
            metrics_dict[f"{prefix}Leaf_HED"] = hed
            return metrics_dict

        # 2. Evaluate Hyperbolic Branch (Standard & Forced)
        if self.hyperbolic in [True, "mixed"]:
            # Standard
            final_metrics.update(compute_and_add_metrics(results["hyp_preds"], targets, prefix=""))
            # Forced
            if non_leaf_indices:
                final_metrics.update(compute_and_add_metrics(results["hyp_preds_forced"], targets, prefix="Forced_"))

            mean_radius, max_radius = self.get_radius(embeddings['embeddings'])
            final_metrics['Mean_radius'] = mean_radius
            final_metrics['Max_radius'] = max_radius

        # 3. Evaluate Euclidean Branch (Standard & Forced)
        if self.hyperbolic in [False, "mixed"]:
            euc_prefix = "Euc_" if self.hyperbolic == "mixed" else ""
            
            # Standard
            final_metrics.update(compute_and_add_metrics(results["euc_preds"], targets, prefix=euc_prefix))
            # Forced
            if non_leaf_indices:
                final_metrics.update(compute_and_add_metrics(results["euc_preds_forced"], targets, prefix=f"Forced_{euc_prefix}"))

        # 4. Trigger Callbacks
        for callback in self.callbacks:
            action = callback(self.callback_state)
            action()
            
        return final_metrics, embeddings

    def train(self, epochs):
        history = []
        best_acc = 0.0
        
        print(f"Starting Training on {self.device} for {epochs} epochs...")
        
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        for epoch in range(epochs):
            self.callback_state.update('epoch', epoch)         
            self.radius_threshold_update(epoch=epoch)   
            train_loss = self.train_epoch(epoch)
            metrics, embeddings = self.evaluate()
            self.scheduler.step()
            self.radius_penalty_update(epoch, epochs)            
            # Save State
            log_entry = {'epoch': epoch+1, 'loss': train_loss, **metrics}
            history.append(log_entry)
            
            print_metrics_report(metrics)
            
            # Checkpoint
            if metrics['L2_Family_BalAcc'] > best_acc:
                best_acc = metrics['L2_Family_BalAcc']
                self.save_model(path = os.path.join(self.output_dir, 'best_model.pth'))

            # Save Metrics CSV
            pd.DataFrame(history).to_csv(os.path.join(self.output_dir, "training_log.csv"), index=False)
            self.save_embeddings(embeddings, epoch)
            print("Training Complete.")
    

    def evaluate_and_save_to(self, outdir):
        final_metrics, embeddings = self.evaluate()
        log_entry = {'epoch': 1, 'loss': -1, **final_metrics}
        pd.DataFrame([log_entry]).to_csv(os.path.join(outdir, "evaluation.csv"), index=False)
        
        with open(os.path.join(outdir, 'evaluation_embeddings.pkl'), 'wb') as handle:
            pickle.dump(embeddings, handle, protocol = pickle.HIGHEST_PROTOCOL)

        print_metrics_report(final_metrics)

    
    def save_embeddings(self, embeddings, epoch):
        
        embeddings_save_dir = os.path.join(self.output_dir, 'embeddings')
        os.makedirs(embeddings_save_dir, exist_ok=True)

        with open(os.path.join(embeddings_save_dir, f'embeddings_{epoch+1}.pkl'), 'wb') as handle:
            pickle.dump(embeddings, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    def save_model(self, path):
        
        save_dict = {
            'model_state': self.model.state_dict(),
            'raw_protos': None,
            'euclidean_protos': None,
            'hyp_tau': None,
            'hyperbolic': self.hyperbolic
        }
        if self.hyperbolic == True:
            save_dict['raw_protos'] = self.train_loader.dataset.get_prototypes()
            save_dict['hyp_tau'] = self.hyp_tau
        elif self.hyperbolic == 'mixed':
            save_dict['raw_protos'] = self.train_loader.dataset.get_prototypes()
            save_dict['euclidean_protos'] = self.prototypes_euc
        else:
            save_dict['euclidean_protos'] = self.prototypes

        torch.save(save_dict, path)
        print("Saved checkpoint to ", path)
        
    def load_weights(self, path):
        print(f"Loading weights from {path}")

        if os.path.isdir(path):
            potential_paths = Path(path).rglob("*.pt*")
            if len(potential_paths) > 1 or len(potential_paths) == 0:
                raise ValueError(f"Providing directory to load weights only possible if one and only one set of weights exists in its subdirectories. " \
                f"Current number of weights {len(potential_paths)}")
            
            path = potential_paths[0]
        
        
        full_state = torch.load(path, map_location=self.device)
        
        old_format = 'hyperbolic' not in full_state.keys()
        if old_format:
            self.model.load_state_dict(full_state)
            print("Using old loading format, prototypes will be taken from dataset instead")
        else:
            self.model.load_state_dict(full_state['model_state'])
            
            if self.hyperbolic == True:
                self.hyp_tau = full_state['hyp_tau']
                self.prototypes = geoopt.ManifoldTensor(full_state['raw_protos'], manifold=self.model.manifold)
            elif self.hyperbolic == 'mixed':
                self.prototypes_euc = full_state['euclidean_protos']
                self.prototypes_hyp = geoopt.ManifoldTensor(full_state['raw_protos'], manifold=self.model.manifold)
            
            else:
                self.prototypes = full_state['euclidean_protos']
            
        print("Succesfully loaded model from checkpoint", path)
    

    def evaluate_isic(self, outdir):

        image_paths, labels = load_isic_data(ISIC_LABELS_PATH, ISIC_IMAGE_PATH)
        self.val_loader.dataset.from_image_paths(image_paths, labels)
        print("Evaluating ", len(image_paths), 'Images from isic dataset')
        os.makedirs(outdir, exist_ok=True)
        self.evaluate_and_save_to(outdir)









