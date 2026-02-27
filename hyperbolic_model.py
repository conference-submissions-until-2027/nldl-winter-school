import torch
import torch.nn as nn
import timm
import geoopt

class HyperbolicConvNeXt(nn.Module):
    def __init__(self, embedding_dim=10, backbone_name='convnext_tiny', pretrained=True):
        """
        Args:
            embedding_dim (int): The dimension of the Hyperboloid in Ambient Space.
                                 e.g., if embedding_dim=10, points are in R^10.
        """
        super().__init__()
        
        # 1. Define the Manifold
        # k=1.0 implies curvature = -1.0 in geoopt convention
        self.manifold = geoopt.Lorentz(k=1.0)
        
        # 2. Euclidean Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        input_dim = self.backbone.num_features
        
        # 3. Projection Head
        # We predict the spatial components (indices 1 to d)
        # The time component (index 0) at the origin's tangent space is always 0.
        self.spatial_dim = embedding_dim - 1
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.spatial_dim) 
        )

        self._init_weights()


    def _init_weights(self,):
        
        # 1. Standard init for the intermediate layers
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 2. "Kick" the final layer
        # We want the output to be non-zero to push z away from the origin.
        # We increase the variance of the final layer slightly.
        last_layer = self.head[-1]
        nn.init.normal_(last_layer.weight, mean=0.0, std=0.05)
        # OPTIONAL: Initialize bias to push towards a specific sector 
        # (not strictly necessary if std is high enough)


    def forward(self, x):
        # 1. Extract Euclidean Features
        features = self.backbone(x)
        
        # 2. Predict Spatial Components
        spatial_components = self.head(features)

        # 3. Construct Tangent Vector at Origin (Time=0)
        zeros = torch.zeros(spatial_components.size(0), 1, device=x.device)
        tangent_vector = torch.cat([zeros, spatial_components], dim=1)
        
        # --- STABILITY FIX START ---
        # Calculate the norm of the tangent vector
        # (Since dim 0 is 0, this is just the norm of spatial_components)
        norms = torch.norm(tangent_vector, p=2, dim=-1, keepdim=True)
        
        # Define a safe radius. 
        # r=15 implies scaling of e^15 (~3 million), which is huge but safe.
        # r=50 puts you near float32 limit. 
        MAX_RADIUS = 15.0 
        
        # Clip the tangent vector if it exceeds MAX_RADIUS
        # We use a clamp-based scaling factor to avoid boolean indexing issues
        scale = torch.clamp(MAX_RADIUS / (norms + 1e-6), max=1.0)
        tangent_vector = tangent_vector * scale
        # --- STABILITY FIX END ---

        # 4. Map to Hyperbolic Space
        z = self.manifold.expmap0(tangent_vector)
        
        return z