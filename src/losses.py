import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class PhysicsInformedWorldModelLoss(nn.Module):
    """
    Composite Physics-Informed Loss Function with Multi-Step Horizon & Work-Energy Constraints.
    
    Components:
      1. Reconstruction Loss: Observation autoencoder fidelity ||x - D(E(x))||^2
      2. Multi-Step Latent Dynamics Loss: Autoregressively unrolled consistency in [s, v] canonical phase space
      3. Future Observation Loss: Decoding accuracy for rolled-out future steps
      4. Work-Energy Consistency Loss: Work done by predicted acceleration matches kinetic energy change:
         Delta(0.5 * ||v||^2) = (a_net . v_avg) * dt
    """
    def __init__(
        self,
        w_recon: float = 1.0,
        w_latent: float = 1.0,
        w_future: float = 1.0,
        w_energy: float = 0.1,
        gamma_decay: float = 0.95
    ):
        super().__init__()
        self.w_recon = w_recon
        self.w_latent = w_latent
        self.w_future = w_future
        self.w_energy = w_energy
        self.gamma_decay = gamma_decay
        self.mse = nn.MSELoss()

    def forward(
        self,
        model: nn.Module,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        dt: float = 0.05
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            model: PhysicsInformedWorldModel instance
            obs_seq: Ground-truth observations (batch_size, horizon + 1, obs_dim)
            action_seq: Executed actions (batch_size, horizon, action_dim)
            dt: Time step duration in seconds
            
        Returns:
            total_loss: Scalar PyTorch loss tensor
            metrics: Dictionary of individual loss components
        """
        batch_size, horizon, _ = action_seq.shape
        latent_dim = model.latent_dim
        
        # 1. Encode all true observations into ground-truth latent targets
        flat_obs = obs_seq.view(-1, obs_seq.shape[-1])
        flat_z_true = model.encode(flat_obs)
        z_seq_true = flat_z_true.view(batch_size, horizon + 1, -1)
        
        # 2. Initial state autoencoding loss at t = 0
        x_0_true = obs_seq[:, 0, :]
        z_0_true = z_seq_true[:, 0, :]
        x_0_hat = model.decode(z_0_true)
        loss_recon = self.mse(x_0_hat, x_0_true)
        
        # 3. Roll out in latent imagination from z_0_true using action sequence
        z_imagined_seq, accel_imagined_seq = model.dream(z_0_true, action_seq, dt=dt)
        
        # 4. Multi-Step Latent Dynamics Loss with exponential horizon weighting
        weights = torch.tensor(
            [self.gamma_decay ** t for t in range(horizon)],
            device=obs_seq.device,
            dtype=torch.float32
        ).view(1, horizon, 1)
        
        latent_diff_sq = (z_imagined_seq[:, 1:, :] - z_seq_true[:, 1:, :]) ** 2
        loss_latent = torch.mean(weights * latent_diff_sq)
        
        # 5. Future Observation Decoding Loss
        flat_z_imagined_future = z_imagined_seq[:, 1:, :].reshape(-1, z_imagined_seq.shape[-1])
        flat_x_future_hat = model.decode(flat_z_imagined_future)
        flat_x_future_true = obs_seq[:, 1:, :].reshape(-1, obs_seq.shape[-1])
        loss_future = self.mse(flat_x_future_hat, flat_x_future_true)
        
        # 6. Physical Work-Energy Balance Loss in Latent Space
        # Kinetic energy proxy in canonical velocity space: T = 0.5 * ||v||^2
        v_curr = z_imagined_seq[:, :-1, latent_dim:]
        v_next = z_imagined_seq[:, 1:, latent_dim:]
        
        delta_kinetic_energy = 0.5 * (torch.sum(v_next ** 2, dim=-1) - torch.sum(v_curr ** 2, dim=-1))
        
        # Work done: W = a_net . v_avg * dt
        v_avg = 0.5 * (v_curr + v_next)
        work_done = torch.sum(accel_imagined_seq * v_avg, dim=-1) * dt
        loss_energy = self.mse(delta_kinetic_energy, work_done)
        
        # Total Weighted Multi-Objective Loss
        total_loss = (
            self.w_recon * loss_recon +
            self.w_latent * loss_latent +
            self.w_future * loss_future +
            self.w_energy * loss_energy
        )
        
        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_recon": float(loss_recon.item()),
            "loss_latent": float(loss_latent.item()),
            "loss_future": float(loss_future.item()),
            "loss_energy": float(loss_energy.item())
        }
        
        return total_loss, metrics
