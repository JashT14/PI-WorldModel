import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict

class StateEncoder(nn.Module):
    """
    Encodes physical observations into decoupled canonical coordinates:
    z = [s, v], where s in R^d is generalized position and v in R^d is generalized velocity.
    """
    def __init__(self, obs_dim: int = 3, latent_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim * 2) # Outputs [s, v]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class SymplecticKinematicsTransitionNet(nn.Module):
    """
    Physics-Informed Latent Transition Layer.
    
    Predicts net generalized acceleration:
        a_net = f_theta(s, v, action)
    
    Enforces exact symplectic Newtonian kinematics:
        v_(t+1) = v_t + a_net * dt
        s_(t+1) = s_t + v_t * dt + 0.5 * a_net * (dt^2)
        
    Mathematical guarantee:
        d(s)/dt = v is analytically preserved by construction.
    """
    def __init__(
        self,
        latent_dim: int = 2,
        action_dim: int = 1,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Acceleration predictor: inputs are [s, v, action], output is a_net in R^latent_dim
        in_dim = (latent_dim * 2) + action_dim
        self.accel_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        dt: float = 0.05
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_t: Latent state tensor of shape (batch, 2 * latent_dim) = [s_t, v_t]
            action: Action tensor of shape (batch, action_dim)
            dt: Time step duration in seconds
            
        Returns:
            z_next: Next latent state [s_(t+1), v_(t+1)]
            accel: Predicted net acceleration a_net
        """
        s_t = z_t[:, :self.latent_dim]
        v_t = z_t[:, self.latent_dim:]
        
        # Predict physical acceleration from state + action
        inputs = torch.cat([z_t, action], dim=-1)
        accel = self.accel_net(inputs)
        
        # Exact Symplectic Kinematic Integration
        v_next = v_t + accel * dt
        s_next = s_t + (v_t * dt) + (0.5 * accel * (dt ** 2))
        
        z_next = torch.cat([s_next, v_next], dim=-1)
        return z_next, accel


class StateDecoder(nn.Module):
    """
    Decodes latent canonical states [s, v] back to original observation space x_hat.
    """
    def __init__(self, latent_dim: int = 2, obs_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, obs_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PhysicsInformedWorldModel(nn.Module):
    """
    Complete Action-Conditioned Physics-Informed Latent World Model (PI-WorldModel).
    """
    def __init__(
        self,
        obs_dim: int = 3,
        action_dim: int = 1,
        latent_dim: int = 2,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        self.encoder = StateEncoder(obs_dim, latent_dim, hidden_dim)
        self.transition = SymplecticKinematicsTransitionNet(latent_dim, action_dim, hidden_dim)
        self.decoder = StateDecoder(latent_dim, obs_dim, hidden_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def step_latent(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        dt: float = 0.05
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transition(z_t, action, dt=dt)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def get_latent_kinetic_energy(self, z: torch.Tensor) -> torch.Tensor:
        """T_latent = 0.5 * ||v||^2"""
        v = z[:, self.latent_dim:]
        return 0.5 * torch.sum(v ** 2, dim=-1)

    def dream(
        self,
        z_0: torch.Tensor,
        action_seq: torch.Tensor,
        dt: float = 0.05
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressively rolls out imaginary trajectories in latent space.
        
        Args:
            z_0: Initial latent state (batch_size, 2 * latent_dim)
            action_seq: Action sequence tensor (batch_size, horizon, action_dim)
            dt: Simulation time step
            
        Returns:
            z_trajectory: Imagined latent states (batch_size, horizon + 1, 2 * latent_dim)
            accel_trajectory: Imagined accelerations (batch_size, horizon, latent_dim)
        """
        batch_size, horizon, _ = action_seq.shape
        z_curr = z_0
        
        z_list = [z_curr]
        accel_list = []
        
        for t in range(horizon):
            a_t = action_seq[:, t, :]
            z_next, accel = self.transition(z_curr, a_t, dt=dt)
            z_list.append(z_next)
            accel_list.append(accel)
            z_curr = z_next
            
        z_trajectory = torch.stack(z_list, dim=1)
        accel_trajectory = torch.stack(accel_list, dim=1)
        return z_trajectory, accel_trajectory


class UnconstrainedWorldModel(nn.Module):
    """
    Standard Unconstrained Baseline World Model (Black-Box MLP Transition).
    Used as an ablation baseline to demonstrate physical drift and momentum hallucination.
    """
    def __init__(
        self,
        obs_dim: int = 3,
        action_dim: int = 1,
        latent_dim: int = 4,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, obs_dim)
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def dream(
        self,
        z_0: torch.Tensor,
        action_seq: torch.Tensor,
        dt: float = 0.05
    ) -> Tuple[torch.Tensor, None]:
        batch_size, horizon, _ = action_seq.shape
        z_curr = z_0
        z_list = [z_curr]
        
        for t in range(horizon):
            a_t = action_seq[:, t, :]
            inputs = torch.cat([z_curr, a_t], dim=-1)
            # Unconstrained delta addition
            delta_z = self.transition(inputs)
            z_next = z_curr + delta_z * dt
            z_list.append(z_next)
            z_curr = z_next
            
        z_trajectory = torch.stack(z_list, dim=1)
        return z_trajectory, None
