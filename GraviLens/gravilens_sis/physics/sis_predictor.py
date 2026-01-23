class HybridSISPredictor(nn.Module):
    """
    RECOMMENDED: Hybrid model with physics-motivated constraints
    
    Predicts:
    1. Global Einstein radius θ_E (pure SIS component)
    2. Perturbation map Δκ(x,y) (departures from SIS)
    
    Total: κ(θ) = (θ_E/2θ) + Δκ(x,y)
    
    This bridges pure physics and flexible ML.
    """
    def __init__(self, embed_dim, image_size=64, 
                 theta_E_min=0.7, theta_E_max=1.4, device=None):
        super().__init__()
        self.image_size = image_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.theta_E_min = theta_E_min
        self.theta_E_max = theta_E_max
        
        # Branch 1: Predict global Einstein radius (pure SIS)
        self.sis_branch = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1),
            nn.Sigmoid()  # → [0, 1]
        )
        
        # Branch 2: Predict perturbation map (departures from SIS)
        self.perturbation_branch = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()  # → [-1, 1]
        )
        
        # Learnable perturbation strength
        self.perturbation_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        """
        Args:
            x: (B, N+1, embed_dim) transformer output with CLS token
        
        Returns:
            theta_E: (B,) Einstein radius [pure SIS parameter]
            kappa_map: (B, 1, H, W) full convergence map [SIS + perturbations]
        """
        batch_size = x.size(0)
        
        # Extract CLS token (global information)
        cls_token = x[:, 0, :]  # (B, embed_dim)
        
        # Predict Einstein radius (global SIS strength)
        theta_E_normalized = self.sis_branch(cls_token)  # (B, 1)
        theta_E = self.theta_E_min + (self.theta_E_max - self.theta_E_min) * theta_E_normalized
        theta_E = theta_E.squeeze(-1)  # (B,)
        
        # Predict spatial perturbations (substructure, ellipticity, etc.)
        patch_tokens = x[:, 1:, :]  # (B, N, embed_dim)
        perturbations = self.perturbation_branch(patch_tokens)  # (B, N, 1)
        
        # Reshape to spatial map
        H = W = int(torch.sqrt(torch.tensor(patch_tokens.size(1))))
        perturbations = perturbations.view(batch_size, H, W, 1)
        
        # Upsample if needed
        if H != self.image_size:
            perturbations = F.interpolate(
                perturbations.permute(0, 3, 1, 2),
                size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=True
            ).permute(0, 2, 3, 1)
        
        # Compute theoretical SIS convergence at each position
        coords = torch.linspace(-3.2, 3.2, self.image_size, device=self.device)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')
        r = torch.sqrt(grid_x**2 + grid_y**2 + 1e-6)  # radial distance
        
        # SIS formula: κ(r) = θ_E / (2r)
        kappa_sis = theta_E.view(-1, 1, 1, 1) / (2 * r.unsqueeze(0).unsqueeze(0))
        
        # Add learned perturbations (scaled)
        delta_kappa = perturbations[..., 0:1].permute(0, 3, 1, 2) * torch.sigmoid(self.perturbation_scale)
        
        # Final convergence map: SIS + perturbations
        kappa_map = kappa_sis + delta_kappa * 0.1 * (self.theta_E_max - self.theta_E_min)
        
        # Physical constraints
        kappa_map = torch.clamp(kappa_map, 0.8, 1.2)  # Keep physically reasonable
        
        return theta_E, kappa_map