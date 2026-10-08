import os
import unittest
import numpy as np
import torch
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PendulumPhysicsEnv, CartPolePhysicsEnv
from src.dataset import TrajectoryCollector, TrajectoryDataset, create_dataloader

class TestPhase1EnvironmentAndData(unittest.TestCase):
    
    def test_zero_damping_energy_conservation(self):
        """
        Physical Invariance Test:
        In conservative motion (zero torque, zero friction), total mechanical energy
        E = T + V must remain strictly constant over hundreds of steps.
        """
        env = PendulumPhysicsEnv(m=1.0, l=1.0, g=9.81, b=0.0, max_torque=0.0, dt=0.01, integrator="rk4")
        # Release from 45 degrees with zero initial velocity
        env.reset(initial_state=[np.pi / 4.0, 0.0])
        
        initial_energy = env.get_total_energy()
        energy_history = []
        
        for _ in range(500):
            _, _, _, info = env.step(0.0)
            energy_history.append(info["total_energy"])
            
        max_energy_deviation = np.max(np.abs(np.array(energy_history) - initial_energy))
        relative_error = max_energy_deviation / initial_energy
        
        print(f"\n[Test Phase 1] Max Relative Energy Drift (RK4, 500 steps, b=0): {relative_error * 100:.4f}%")
        self.assertLess(relative_error, 0.001, "RK4 energy drift exceeded 0.1% in conservative test")

    def test_work_energy_balance_with_actuator(self):
        """
        First Law of Thermodynamics:
        delta(E) = W_actuator - W_damping
        """
        env = PendulumPhysicsEnv(m=1.0, l=1.0, g=9.81, b=0.2, max_torque=2.0, dt=0.02, integrator="rk4")
        env.reset(initial_state=[np.pi * 0.5, 0.0])
        
        E_0 = env.get_total_energy()
        cumulative_work = 0.0
        cumulative_damping = 0.0
        
        for t in range(100):
            u = float(np.sin(t * 0.1) * 1.5)
            _, _, _, info = env.step(u)
            cumulative_work += info["step_work"]
            cumulative_damping += info["step_damping"]
            
        E_final = env.get_total_energy()
        delta_E = E_final - E_0
        net_energy_balance = delta_E - (cumulative_work - cumulative_damping)
        
        print(f"[Test Phase 1] Work-Energy Residual: {abs(net_energy_balance):.5f} Joules")
        self.assertLess(abs(net_energy_balance), 0.05)

    def test_cartpole_env(self):
        """Tests CartPole continuous step dynamics"""
        env = CartPolePhysicsEnv(dt=0.02)
        obs = env.reset(initial_state=[0.0, 0.0, 0.1, 0.0])
        self.assertEqual(len(obs), 5) # [x, x_dot, cos(th), sin(th), th_dot]
        next_obs, reward, done, _ = env.step(2.0)
        self.assertEqual(len(next_obs), 5)

    def test_trajectory_collector_modes(self):
        """Tests that all exploration policy modes generate valid bounded trajectories"""
        env = PendulumPhysicsEnv(dt=0.05)
        collector = TrajectoryCollector(env)
        
        for mode in ["chirp", "multisine", "bang_bang", "impulse", "random"]:
            trajs = collector.collect_trajectories(num_episodes=2, episode_length=50, policy_modes=[mode])
            self.assertEqual(len(trajs), 2)
            self.assertEqual(trajs[0]["observations"].shape, (51, 3))
            self.assertEqual(trajs[0]["actions"].shape, (50, 1))

    def test_dataset_serialization_and_dataloader(self):
        """Tests saving, loading, and batching sliding windows"""
        env = PendulumPhysicsEnv(dt=0.05)
        collector = TrajectoryCollector(env)
        trajectories = collector.collect_trajectories(num_episodes=4, episode_length=60)
        
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_file:
            temp_path = tmp_file.name
            
        try:
            TrajectoryCollector.save_trajectories(trajectories, temp_path)
            loaded_trajectories = TrajectoryCollector.load_trajectories(temp_path)
            self.assertEqual(len(loaded_trajectories), 4)
            np.testing.assert_allclose(trajectories[0]["observations"], loaded_trajectories[0]["observations"])
            
            loader = create_dataloader(loaded_trajectories, horizon=5, batch_size=16, shuffle=True)
            for obs_batch, act_batch in loader:
                self.assertEqual(obs_batch.shape, (16, 6, 3)) # (B, horizon + 1, obs_dim)
                self.assertEqual(act_batch.shape, (16, 5, 1)) # (B, horizon, act_dim)
                break
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    unittest.main()
