import torch
import torch.nn as nn
import timm
import geoopt

class HybridSkinCancerModel(nn.Module):
    def __init__(self, hyp_dim=3, euc_dim=10, backbone_name='convnext_tiny', pretrained=True):
        """
        Args:
            hyp_dim (int): Lorentz ambient dimensions (1 time + (hyp_dim-1) spatial).
            euc_dim (int): Dimensions for the Euclidean embedding.
        """
        super().__init__()
        
        # 1. Geometry Definitions
        self.manifold = geoopt.Lorentz(k=1.0)
        self.hyp_spatial_dim = hyp_dim - 1
        self.euc_dim = euc_dim
        
        # 2. Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        input_dim = self.backbone.num_features
        
        # 3. Hyperbolic Head (Maps to tangent space at origin)
        self.hyp_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.hyp_spatial_dim) 
        )
        
        # 4. Euclidean Head (Maps to hypersphere/standard R^n)
        self.euc_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.euc_dim)
        )

        # 5. Learnable Temperature for Euclidean Cosine Similarity
        self.euc_logit_scale = nn.Parameter(torch.ones([]) * 30.0)

        self._init_weights()

    def _init_weights(self):
        # Xavier for intermediate, "Kick" for final hyp layer
        for m in [self.hyp_head, self.euc_head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # Increase variance of hyp final layer to push away from origin
        nn.init.normal_(self.hyp_head[-1].weight, mean=0.0, std=0.05)

    def forward(self, x):
        features = self.backbone(x)
        
        # --- Hyperbolic Branch ---
        hyp_spatial = self.hyp_head(features)
        zeros = torch.zeros(hyp_spatial.size(0), 1, device=x.device)
        tangent_vec = torch.cat([zeros, hyp_spatial], dim=1)
        
        # Numerical Stability Rail (15.0 to avoid float32 overflow)
        norms = torch.norm(tangent_vec, p=2, dim=-1, keepdim=True)
        safe_scale = torch.clamp(15.0 / (norms + 1e-6), max=1.0)
        z_hyp = self.manifold.expmap0(tangent_vec * safe_scale)
        
        # --- Euclidean Branch ---
        z_euc = self.euc_head(features)
        # Normalize to unit hypersphere for Cosine similarity fairness
        z_euc = torch.nn.functional.normalize(z_euc, p=2, dim=1)
        
        return {
            "hyp": z_hyp,        # For Lorentz distance to fixed prototypes
            "euc": z_euc,        # For dot-product with learned prototypes
            "euc_scale": self.euc_logit_scale
        }