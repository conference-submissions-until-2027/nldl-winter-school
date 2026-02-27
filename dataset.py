import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold


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


def get_transforms(split='train', img_size=224):
    if split == 'train':
        return A.Compose([
            # 1. RESIZE / CROP (Must be first)
            # RandomResizedCrop is better than Resize because it adds scale invariance.
            # We use a high scale range (0.7 to 1.0) to avoid cropping out the lesion.
            A.RandomResizedCrop(size = (img_size, img_size), scale=(0.7, 1.0), ratio=(0.8, 1.2), p=1.0),
            
            # 2. GEOMETRIC (Rotations are safe after square crop)
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            # 3. MORPHOLOGICAL (Elastic/Grid)
            # Now applied to the 224x224 image (Fast)
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.0, p=1),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            ], p=0.3),
            
            # 4. COLOR CONSTANCY
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
            
            # 5. REGULARIZATION
            A.CoarseDropout(max_holes=8, max_height=img_size//8, max_width=img_size//8, p=0.3),
            
            # 6. NORMALIZATION (Must be last before ToTensor)
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        # Validation/Test: Deterministic Resize
        return A.Compose([
            # Resize small edge to img_size, maintaining aspect ratio, then crop center
            A.SmallestMaxSize(max_size=img_size),
            A.CenterCrop(height=img_size, width=img_size),
            
            # Alternatively, if you accept squeezing:
            # A.Resize(height=img_size, width=img_size),
            
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# --- 2. DATASET CLASS ---
class SkinLesionDataset(Dataset):
    def __init__(self, csv_path, prototypes_pkl_path, split='train', val_ratio=0.2, seed=42, img_size=224, taxonomy_path = ""):
        """
        Args:
            csv_path: Path to metadata CSV.
            prototypes_pkl_path: Path to the generated prototypes pickle.
            split: 'train' or 'val'.
            val_ratio: Fraction of data to use for validation (default 0.2).
            seed: Random seed for deterministic splitting.
            taxonomy_path: path to the hierachial_tree.json containing the hierachy structure of the data. 
        """
        self.split = split
        self.img_size = img_size
        
        self._diagnostic_mapping = None
        self._build_hierarchy_paths(taxonomy_path)
        # 1. Load Prototypes First (Defines the Class Map)
        if not os.path.exists(prototypes_pkl_path):
            raise FileNotFoundError(f"Prototypes file not found: {prototypes_pkl_path}")
            
        with open(prototypes_pkl_path, 'rb') as f:
            raw_prototypes = pickle.load(f)

        # Sort keys to ensure deterministic index mapping
        self.class_names = sorted(list(raw_prototypes.keys()))
        self.label_map = {name: i for i, name in enumerate(self.class_names)}
        
        # Stack into tensor (N_classes, Dim)
        self.prototype_bank = torch.stack([raw_prototypes[name] for name in self.class_names])
        self.non_leaf_indices = [self.label_map[class_name] for class_name in ['skin lesion', 'benign', 'malign']]

        # 2. Load and Split Data
        df = read_table(csv_path)
        # Create a clean target column for stratification
        # (Must match the normalization logic used for prototypes)
        df['stratify_label'] = df['diagnosis_names'].astype(str).str.lower().str.strip()
        
        # Identify Singletons (Classes with only 1 sample)
        counts = df['stratify_label'].value_counts()
        singletons = counts[counts < 2].index.tolist()
        
        # Separate Data
        df_singletons = df[df['stratify_label'].isin(singletons)].copy()
        df_main = df[~df['stratify_label'].isin(singletons)].copy()
        
        # Perform Stratified Split on the Main Data
        skf = StratifiedKFold(n_splits=int(1/val_ratio), shuffle=True, random_state=seed)
        
        # We only need the indices of the first fold
        train_idx, val_idx = next(skf.split(df_main, df_main['stratify_label']))
        
        # Assign based on split argument
        if split == 'train':
            # Train = Main Train Fold + All Singletons (Forced inclusion)
            self.df = pd.concat([df_main.iloc[train_idx], df_singletons])
        elif split == 'val':
            # Val = Main Val Fold only
            self.df = df_main.iloc[val_idx]
        else:
            raise ValueError(f"Unknown split '{split}'")
            
        # 3. Setup Transform
        self.transform = get_transforms(split, img_size)
        
        print(f"[{split.upper()}] Dataset Loaded: {len(self.df)} images.")
        if split == 'train':
            print(f"   - Included {len(df_singletons)} singleton samples to prevent class loss.")

        self.image_paths = self.df['image_path'].tolist()
        self.diagnoses = self.df['diagnosis_names'].astype(str).str.lower().str.strip().tolist()

    def from_image_paths(self, image_paths, diagnoses = None):

        self.image_paths = image_paths
        if diagnoses is None:
            self.diagnoses = ['skin lesion'] * len(image_paths)
        else:
            self.diagnoses = diagnoses
        
    def _build_hierarchy_paths(self, taxonomy_path):
        """
        Parses the JSON taxonomy tree and precomputes the L1, L2, and L3 
        ancestors for every single diagnosis in the dataset.
        """
        import json
        self.hierarchy_paths = {}
        
        if not taxonomy_path or not os.path.exists(taxonomy_path):
            print("Warning: No valid taxonomy_path provided. Hierarchy paths will be empty.")
            return

        with open(taxonomy_path, 'r') as f:
            taxonomy = json.load(f)
            
        # Handle the common 'skin lesion' root wrapper if it exists
        root_keys = list(taxonomy.keys())
        if len(root_keys) == 1 and root_keys[0].lower().strip() == 'skin lesion':
            tree = taxonomy[root_keys[0]]
        else:
            tree = taxonomy
            
        def traverse(node_dict, current_path):
            """Recursive helper to walk the tree and build the path dictionaries."""
            for node_name, children in node_dict.items():
                # Normalize the string to guarantee perfect matching
                clean_name = str(node_name).lower().strip()
                
                # Create the new path branch including this node
                new_path = current_path + [clean_name]
                
                # Safely assign L1, L2, and L3 based on the current depth of the branch
                path_dict = {
                    'L1': new_path[0] if len(new_path) > 0 else None,
                    'L2': new_path[1] if len(new_path) > 1 else None,
                    'L3': new_path[2] if len(new_path) > 2 else None,
                }
                
                # Map this specific node to its full ancestral path
                self.hierarchy_paths[clean_name] = path_dict
                
                # If this node has children, keep digging deeper
                if isinstance(children, dict) and len(children) > 0:
                    traverse(children, new_path)

        # Start the traversal from the top-level (L1) nodes
        traverse(tree, current_path=[])
        
        print(f"Mapped hierarchy paths for {len(self.hierarchy_paths)} unique diagnostic nodes.")

    def set_diagnostic_mapping(self, new_mapping):
        
        if not isinstance(new_mapping, dict):
            raise ValueError("input argument new mapping must be a dictionary")
        # Check if mapping is defined for all
        assert all(elem in new_mapping for elem in self.label_map), 'param new_mapping must map all class names'
        self._diagnostic_mapping = new_mapping
        
        new_index_mapping = {name: i for i, name in enumerate(sorted(set(new_mapping.values())))} # The new label indices e.g. malign/benign
        new_label_map = {}
        for name in self.label_map:
            new_label_map[name] = new_index_mapping[new_mapping[name]]
        self.label_map = new_label_map
        
        
    def _map_diagnosis(self, diagnosis):
        return self._diagnostic_mapping(diagnosis)

    def get_prototypes(self):
        return self.prototype_bank

    def __len__(self):
        return len(self.df)
    
    @property
    def num_classes(self,):
        return len(set(self.label_map.values()))

    def __getitem__(self, idx):
        # row = self.df.iloc[idx]
        img_path = self.image_paths[idx]
        
        # 1. Load Image (Convert to Numpy for Albumentations)
        try:
            image_pil = Image.open(img_path).convert('RGB')
            # buffer_size = int(self.img_size * 2)
            # image_pil = image_pil.resize((buffer_size, buffer_size), resample=Image.BILINEAR)
            image_np = np.array(image_pil)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return black image on corruption to prevent crash
            image_np = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # 2. Apply Strong Augmentations
        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        # 3. Get Label
        # Normalize to match prototype keys
        diagnosis = self.diagnoses[idx]
        # diagnosis = str(row['diagnosis_names']).lower().strip()
        label_idx = self.label_map.get(diagnosis, -1)
        
        if label_idx == -1:
            # Fallback (Should happen rarely if prototypes generated from same taxonomy)
            label_idx = 0 
            breakpoint()
            
        return image_tensor, label_idx