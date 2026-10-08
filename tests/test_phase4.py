import os
import unittest
import torch
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PendulumPhysicsEnv
from src.models import PhysicsInformedWorldModel
from src.planner import LatentMPCPlanner

class TestPhase4LatentMPCPlanner(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.model = PhysicsInformedWorldModel(obs_dim=3, action_dim=1, latent_dim=2, hidden_dim=32)
        self.env = PendulumPhysicsEnv(max_torque=2.5, dt=0.05)

    def test_mpc_algorithms_execution(self):
        """Tests that CEM, MPPI, and Random Shooting planners execute cleanly and return valid torques"""
        obs = self.env.reset(initial_state=[0.5, 0.0])
        
        for method in ["cem", "mppi", "random_shooting"]:
            planner = LatentMPCPlanner(
                world_model=self.model,
                horizon=10,
                num_candidates=64,
                max_torque=self.env.max_torque,
                method=method,
                device="cpu"
            )
            action, info = planner.plan_action(obs, return_imagined_trajectory=True)
            
            print(f"[Test Phase 4] Planner '{method:<15}': Action={action:+.3f} N*m, Latency={info['elapsed_ms']:.2f} ms")
            self.assertTrue(isinstance(action, float))
            self.assertTrue(-self.env.max_torque <= action <= self.env.max_torque)
            self.assertIn("imagined_obs_trajectory", info)
            self.assertEqual(info["imagined_obs_trajectory"].shape, (10, 3))
            self.assertLess(info["elapsed_ms"], 100.0, f"Planner '{method}' took too long on CPU")

    def test_closed_loop_step_execution(self):
        """Tests a 5-step closed-loop environment interaction with Latent MPC"""
        planner = LatentMPCPlanner(
            world_model=self.model,
            horizon=8,
            num_candidates=32,
            max_torque=self.env.max_torque,
            method="cem",
            cem_iterations=2
        )
        
        obs = self.env.reset(initial_state=[0.3, 0.0])
        for step in range(5):
            action, info = planner.plan_action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            self.assertEqual(len(next_obs), 3)
            obs = next_obs

if __name__ == "__main__":
    unittest.main()
