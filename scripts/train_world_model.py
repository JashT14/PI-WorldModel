import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PendulumPhysicsEnv
from src.models import PhysicsInformedWorldModel
from src.losses import PhysicsInformedWorldModelLoss
from src.dataset import TrajectoryCollector, create_dataloader
from src.trainer import WorldModelTrainer

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Train Physics-Informed Latent World Model")
    parser.add_argument("--data", type=str, default="data/exploration_trajectories.npz", help="Path to trajectory dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--horizon", type=int, default=6, help="Unrolled multi-step horizon")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="models/trained_world_model.pt", help="Checkpoint save path")
    args = parser.parse_args()

    print("=" * 75)
    print("   PHASE 3: MULTI-STEP SELF-SUPERVISED PHYSICS PRETRAINING")
    print("=" * 75)
    
    torch.set_num_threads(4)
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.data))
    
    # Check if dataset exists, else generate automatically
    if os.path.exists(data_path):
        print(f"Loading existing exploration trajectories from: {data_path}")
        trajectories = TrajectoryCollector.load_trajectories(data_path)
    else:
        print(f"Dataset not found at {data_path}. Collecting fresh exploration data...")
        env = PendulumPhysicsEnv(m=1.0, l=1.0, g=9.81, b=0.1, max_torque=2.5, dt=0.05)
        collector = TrajectoryCollector(env)
        trajectories = collector.collect_trajectories(num_episodes=50, episode_length=150)
        TrajectoryCollector.save_trajectories(trajectories, data_path)
        print(f"Saved {len(trajectories)} exploration episodes to: {data_path}")
        
    num_train = int(len(trajectories) * 0.8)
    train_trajs = trajectories[:num_train]
    val_trajs = trajectories[num_train:]
    
    train_loader = create_dataloader(train_trajs, horizon=args.horizon, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_trajs, horizon=args.horizon, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train/Val Split: {len(train_trajs)} / {len(val_trajs)} episodes ({len(train_loader)} batches/epoch).")

    # Instantiate model & multi-objective physics loss
    model = PhysicsInformedWorldModel(
        obs_dim=3,
        action_dim=1,
        latent_dim=2,
        hidden_dim=64
    )
    
    loss_fn = PhysicsInformedWorldModelLoss(
        w_recon=1.0,
        w_latent=1.0,
        w_future=1.0,
        w_energy=0.1,
        gamma_decay=0.95
    )
    
    trainer = WorldModelTrainer(model, loss_fn=loss_fn, lr=args.lr, device="cpu")
    save_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.save_path))
    
    print(f"\nStarting {args.epochs} Training Epochs on CPU (Horizon K={args.horizon})...\n")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        dt=0.05,
        save_best_path=save_full_path,
        verbose=True
    )
    
    print(f"\n  [SUCCESS] Best model checkpoint saved to: {save_full_path}")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.6f} | Final Val Loss: {history['val_loss'][-1]:.6f}")

if __name__ == "__main__":
    main()
