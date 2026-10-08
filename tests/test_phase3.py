import os
import unittest
import torch
import numpy as np
import tempfile
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import PhysicsInformedWorldModel
from src.losses import PhysicsInformedWorldModelLoss
from src.trainer import WorldModelTrainer
from src.dataset import TrajectoryCollector, create_dataloader

class TestPhase3MultiStepTraining(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.model = PhysicsInformedWorldModel(obs_dim=3, action_dim=1, latent_dim=2, hidden_dim=32)
        self.loss_fn = PhysicsInformedWorldModelLoss(w_recon=1.0, w_latent=1.0, w_future=1.0, w_energy=0.1)

    def test_loss_backward_and_stability(self):
        """Tests that composite loss produces finite gradients across all heads"""
        batch_size = 8
        horizon = 6
        obs_seq = torch.randn(batch_size, horizon + 1, 3)
        act_seq = torch.randn(batch_size, horizon, 1)
        
        loss, metrics = self.loss_fn(self.model, obs_seq, act_seq, dt=0.05)
        
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["loss_recon"], 0.0)
        self.assertGreater(metrics["loss_latent"], 0.0)
        
        loss.backward()
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
            self.assertTrue(torch.all(torch.isfinite(param.grad)), f"Non-finite gradient in {name}")

    def test_trainer_fit_and_loss_decrease(self):
        """Tests that a short training run decreases loss on synthetic data"""
        collector = TrajectoryCollector()
        trajs = collector.collect_trajectories(num_episodes=4, episode_length=50)
        train_loader = create_dataloader(trajs, horizon=4, batch_size=16)
        
        trainer = WorldModelTrainer(self.model, loss_fn=self.loss_fn, lr=5e-3)
        history = trainer.fit(train_loader, epochs=5, dt=0.05, verbose=False)
        
        initial_loss = history["train_loss"][0]
        final_loss = history["train_loss"][-1]
        print(f"\n[Test Phase 3] 5-Epoch Loss Convergence: {initial_loss:.4f} -> {final_loss:.4f}")
        self.assertLess(final_loss, initial_loss, "Training loss failed to decrease")

    def test_checkpoint_save_and_load(self):
        """Tests checkpoint saving, loading, and parameter fidelity"""
        trainer = WorldModelTrainer(self.model)
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            trainer.save_checkpoint(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create a new fresh model and load checkpoint
            new_model = PhysicsInformedWorldModel(obs_dim=3, action_dim=1, latent_dim=2, hidden_dim=32)
            new_trainer = WorldModelTrainer(new_model)
            new_trainer.load_checkpoint(tmp_path)
            
            # Verify weight equality
            for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
                np.testing.assert_allclose(p1.detach().numpy(), p2.detach().numpy())
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()
