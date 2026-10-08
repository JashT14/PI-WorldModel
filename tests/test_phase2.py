import os
import unittest
import torch
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import (
    StateEncoder,
    SymplecticKinematicsTransitionNet,
    StateDecoder,
    PhysicsInformedWorldModel,
    UnconstrainedWorldModel
)

class TestPhase2PhysicsInformedArchitecture(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        self.latent_dim = 2 # [s in R^2, v in R^2] -> canonical phase space R^4
        self.obs_dim = 3    # [cos(th), sin(th), th_dot]
        self.action_dim = 1 # [torque]
        self.dt = 0.05

    def test_latent_kinematic_invariance(self):
        """
        Mathematical Invariance Test:
        Verifies that the transition layer strictly executes symplectic Newtonian integration:
            v_(t+1) = v_t + a_net * dt
            s_(t+1) = s_t + v_t * dt + 0.5 * a_net * dt^2
        """
        transition_net = SymplecticKinematicsTransitionNet(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=32
        )
        
        batch_size = 16
        # Generate arbitrary latent states [s, v]
        z_t = torch.randn(batch_size, 2 * self.latent_dim)
        actions = torch.randn(batch_size, self.action_dim)
        
        s_t = z_t[:, :self.latent_dim]
        v_t = z_t[:, self.latent_dim:]
        
        # Forward pass
        z_next, a_net = transition_net(z_t, actions, dt=self.dt)
        s_next = z_next[:, :self.latent_dim]
        v_next = z_next[:, self.latent_dim:]
        
        # Analytical expected values
        expected_v_next = v_t + a_net * self.dt
        expected_s_next = s_t + v_t * self.dt + 0.5 * a_net * (self.dt ** 2)
        
        # Max numerical differences
        v_diff = torch.max(torch.abs(v_next - expected_v_next)).item()
        s_diff = torch.max(torch.abs(s_next - expected_s_next)).item()
        
        print(f"\n[Test Phase 2] Latent Kinematics Invariance Max Error: v_diff={v_diff:.8e}, s_diff={s_diff:.8e}")
        self.assertLess(v_diff, 1e-6, "Velocity kinematic violation detected")
        self.assertLess(s_diff, 1e-6, "Position kinematic violation detected")

    def test_gradient_flow_and_backprop(self):
        """
        Verifies that gradients flow cleanly through multi-step unrolling
        without gradient vanishing or explosion.
        """
        model = PhysicsInformedWorldModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            hidden_dim=32
        )
        
        batch_size = 8
        horizon = 10
        
        obs_0 = torch.randn(batch_size, self.obs_dim, requires_grad=True)
        action_seq = torch.randn(batch_size, horizon, self.action_dim)
        
        z_0 = model.encode(obs_0)
        z_traj, a_traj = model.dream(z_0, action_seq, dt=self.dt)
        
        # Dummy target loss at horizon step
        target_z = torch.zeros_like(z_traj[:, -1, :])
        loss = torch.mean((z_traj[:, -1, :] - target_z) ** 2)
        loss.backward()
        
        # Check encoder and transition gradients
        encoder_grad_norm = torch.norm(model.encoder.net[0].weight.grad).item()
        transition_grad_norm = torch.norm(model.transition.accel_net[0].weight.grad).item()
        
        print(f"[Test Phase 2] Multi-Step Backprop Gradients: Encoder={encoder_grad_norm:.4f}, Transition={transition_grad_norm:.4f}")
        self.assertGreater(encoder_grad_norm, 1e-5)
        self.assertGreater(transition_grad_norm, 1e-5)

    def test_unconstrained_baseline_compatibility(self):
        """Tests that UnconstrainedWorldModel exposes compatible API for ablation benchmarking"""
        base_model = UnconstrainedWorldModel(obs_dim=3, action_dim=1, latent_dim=4, hidden_dim=32)
        obs_0 = torch.randn(4, 3)
        act_seq = torch.randn(4, 5, 1)
        z_0 = base_model.encode(obs_0)
        z_traj, _ = base_model.dream(z_0, act_seq, dt=0.05)
        self.assertEqual(z_traj.shape, (4, 6, 4))

if __name__ == "__main__":
    unittest.main()
