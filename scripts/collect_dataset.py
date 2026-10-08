import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PendulumPhysicsEnv
from src.dataset import TrajectoryCollector

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Generate & Save Physics Exploration Trajectories")
    parser.add_argument("--episodes", type=int, default=50, help="Number of exploration episodes")
    parser.add_argument("--length", type=int, default=200, help="Steps per episode")
    parser.add_argument("--output", type=str, default="data/exploration_trajectories.npz", help="Output filepath")
    args = parser.parse_args()

    print("=" * 70)
    print("   PHASE 1: SYNTHETIC PHYSICAL DATASET GENERATION")
    print("=" * 70)
    
    env = PendulumPhysicsEnv(m=1.0, l=1.0, g=9.81, b=0.1, max_torque=2.5, dt=0.05)
    collector = TrajectoryCollector(env)
    
    print(f"Collecting {args.episodes} episodes x {args.length} steps across mixed policies...")
    trajectories = collector.collect_trajectories(
        num_episodes=args.episodes,
        episode_length=args.length
    )
    
    total_steps = sum(len(t["actions"]) for t in trajectories)
    all_energies = np.concatenate([t["energies"] for t in trajectories])
    all_torques = np.concatenate([t["actions"] for t in trajectories])
    
    print(f"\nDataset Statistics:")
    print(f"  - Total Episodes: {len(trajectories)}")
    print(f"  - Total Transitions: {total_steps:,}")
    print(f"  - Energy Range (Joules): [{all_energies.min():.2f}, {all_energies.max():.2f}] (Mean: {all_energies.mean():.2f})")
    print(f"  - Torque Range (N*m): [{all_torques.min():.2f}, {all_torques.max():.2f}]")
    
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.output))
    TrajectoryCollector.save_trajectories(trajectories, out_path)
    file_size_kb = os.path.getsize(out_path) / 1024.0
    print(f"\n  [SUCCESS] Dataset successfully saved to: {out_path} ({file_size_kb:.1f} KB)")

if __name__ == "__main__":
    main()
