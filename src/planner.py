import torch
import numpy as np
import time
from typing import Tuple, Dict, Any, Optional, List
from .models import PhysicsInformedWorldModel

class LatentMPCPlanner:
    """
    Model Predictive Control (MPC) inside the Latent World Model.
    
    Supported Optimization Algorithms:
      1. 'cem': Cross-Entropy Method with elite selection and temporal smoothing
      2. 'mppi': Model-Predictive Path Integral control with softmax temperature weighting
      3. 'random_shooting': Uniform Monte-Carlo trajectory sampling
    """
    def __init__(
        self,
        world_model: PhysicsInformedWorldModel,
        horizon: int = 15,
        num_candidates: int = 256,
        max_torque: float = 2.5,
        dt: float = 0.05,
        method: str = "cem",
        cem_iterations: int = 3,
        cem_elite_fraction: float = 0.1,
        mppi_temperature: float = 0.5,
        temporal_smoothing: float = 0.2,
        device: str = "cpu"
    ):
        self.world_model = world_model
        self.world_model.eval()
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.max_torque = max_torque
        self.dt = dt
        self.method = method.lower()
        self.cem_iterations = cem_iterations
        self.elite_count = max(4, int(num_candidates * cem_elite_fraction))
        self.mppi_temperature = mppi_temperature
        self.temporal_smoothing = temporal_smoothing
        self.device = torch.device(device)

    def _evaluate_candidate_trajectories(
        self,
        z_0: torch.Tensor,
        action_candidates: torch.Tensor,
        target_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rolls out batch of action candidates and calculates cumulative cost.
        
        Args:
            z_0: Replicated initial latent state (num_candidates, 2 * latent_dim)
            action_candidates: (num_candidates, horizon, action_dim)
            target_state: Optional target observation tensor [cos, sin, vel]
            
        Returns:
            total_costs: Tensor of shape (num_candidates,)
            obs_traj: Imagined observation trajectories (num_candidates, horizon, obs_dim)
        """
        with torch.no_grad():
            z_traj, _ = self.world_model.dream(z_0, action_candidates, dt=self.dt)
            # Decode imaginary observations: (num_candidates, horizon, obs_dim)
            flat_z = z_traj[:, 1:, :].reshape(-1, z_traj.shape[-1])
            flat_obs = self.world_model.decode(flat_z)
            obs_traj = flat_obs.view(self.num_candidates, self.horizon, -1)
            
            cos_th = obs_traj[:, :, 0]
            sin_th = obs_traj[:, :, 1]
            th_dot = obs_traj[:, :, 2]
            
            # Goal is upright: cos_th = 1, sin_th = 0, th_dot = 0
            # Angle deviation cost: (1 - cos_th) + sin_th^2
            angle_cost = (1.0 - cos_th) + 0.5 * (sin_th ** 2)
            velocity_cost = 0.1 * (th_dot ** 2)
            control_cost = 0.005 * (action_candidates.squeeze(-1) ** 2)
            
            # Terminal step priority bonus
            step_costs = angle_cost + velocity_cost + control_cost
            step_costs[:, -1] *= 2.0
            
            total_costs = torch.sum(step_costs, dim=1)
            
        return total_costs, obs_traj

    def plan_action(
        self,
        current_obs: np.ndarray,
        return_imagined_trajectory: bool = False
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Plans the optimal immediate action given current physical observation.
        
        Args:
            current_obs: np.ndarray [cos(th), sin(th), th_dot]
            return_imagined_trajectory: If true, includes best imagined rollout in info
            
        Returns:
            best_action: Float scalar torque to apply to physical actuator
            info: Diagnostics containing planning time, expected cost, etc.
        """
        start_time = time.perf_counter()
        
        obs_tensor = torch.tensor(current_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            z_single = self.world_model.encode(obs_tensor)
            z_0_batch = z_single.repeat(self.num_candidates, 1)
        
        best_action_seq = None
        best_cost = float("inf")
        best_obs_traj = None
        
        if self.method == "random_shooting":
            actions = torch.empty(
                (self.num_candidates, self.horizon, 1),
                device=self.device
            ).uniform_(-self.max_torque, self.max_torque)
            
            costs, obs_traj = self._evaluate_candidate_trajectories(z_0_batch, actions)
            best_idx = torch.argmin(costs).item()
            best_action_seq = actions[best_idx]
            best_cost = costs[best_idx].item()
            best_obs_traj = obs_traj[best_idx]
            
        elif self.method == "mppi":
            # Model-Predictive Path Integral Control
            mean = torch.zeros((self.horizon, 1), device=self.device)
            std = torch.ones((self.horizon, 1), device=self.device) * (self.max_torque / 2.0)
            
            noise = torch.randn((self.num_candidates, self.horizon, 1), device=self.device)
            actions = torch.clamp(
                mean.unsqueeze(0) + noise * std.unsqueeze(0),
                -self.max_torque,
                self.max_torque
            )
            
            costs, obs_traj = self._evaluate_candidate_trajectories(z_0_batch, actions)
            min_cost = torch.min(costs)
            
            # Softmax weights: w_i = exp(-1/temperature * (cost_i - min_cost))
            weights = torch.exp(- (costs - min_cost) / self.mppi_temperature)
            weights = weights / (torch.sum(weights) + 1e-8) # (num_candidates,)
            
            # Weighted average action sequence
            weighted_actions = torch.sum(actions * weights.view(-1, 1, 1), dim=0) # (horizon, 1)
            best_action_seq = weighted_actions
            best_cost = min_cost.item()
            best_idx = torch.argmin(costs).item()
            best_obs_traj = obs_traj[best_idx]
            
        else: # Default: Cross-Entropy Method (CEM)
            mean = torch.zeros((self.horizon, 1), device=self.device)
            std = torch.ones((self.horizon, 1), device=self.device) * (self.max_torque / 2.0)
            
            for iteration in range(self.cem_iterations):
                noise = torch.randn((self.num_candidates, self.horizon, 1), device=self.device)
                actions = torch.clamp(
                    mean.unsqueeze(0) + noise * std.unsqueeze(0),
                    -self.max_torque,
                    self.max_torque
                )
                
                # Apply optional temporal smoothing filter
                if self.temporal_smoothing > 0.0:
                    for t in range(1, self.horizon):
                        actions[:, t, :] = (
                            (1.0 - self.temporal_smoothing) * actions[:, t, :] +
                            self.temporal_smoothing * actions[:, t - 1, :]
                        )
                
                costs, obs_traj = self._evaluate_candidate_trajectories(z_0_batch, actions)
                
                sorted_indices = torch.argsort(costs)
                elites = actions[sorted_indices[:self.elite_count]]
                
                current_best_idx = sorted_indices[0].item()
                if costs[current_best_idx].item() < best_cost:
                    best_cost = costs[current_best_idx].item()
                    best_action_seq = actions[current_best_idx]
                    best_obs_traj = obs_traj[current_best_idx]
                
                # Update Gaussian parameters from elite actions
                mean = torch.mean(elites, dim=0)
                std = torch.std(elites, dim=0) + 1e-4
                
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        immediate_action = float(best_action_seq[0, 0].item())
        
        info = {
            "elapsed_ms": elapsed_ms,
            "predicted_cost": best_cost,
            "horizon": self.horizon,
            "method": self.method,
            "candidates_evaluated": self.num_candidates * (self.cem_iterations if self.method == "cem" else 1)
        }
        
        if return_imagined_trajectory and best_obs_traj is not None:
            info["imagined_obs_trajectory"] = best_obs_traj.cpu().numpy()
            
        return immediate_action, info
