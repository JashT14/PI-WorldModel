import os
import sys
import argparse
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PendulumPhysicsEnv
from src.models import PhysicsInformedWorldModel
from src.trainer import WorldModelTrainer
from src.planner import LatentMPCPlanner

def ascii_rod_visualizer(theta_rad: float) -> str:
    """Returns a simple ASCII indicator of pendulum orientation"""
    # 0 = Up, pi/2 = Right, pi = Down, -pi/2 = Left
    deg = np.degrees(theta_rad) % 360
    if 337.5 <= deg or deg < 22.5:
        return "[ |  UP  ]"
    elif 22.5 <= deg < 67.5:
        return "[ /  UPR ]"
    elif 67.5 <= deg < 112.5:
        return "[ -- RGT ]"
    elif 112.5 <= deg < 157.5:
        return "[ \\  DNR ]"
    elif 157.5 <= deg < 202.5:
        return "[ |  DWN ]"
    elif 202.5 <= deg < 247.5:
        return "[ /  DNL ]"
    elif 247.5 <= deg < 292.5:
        return "[ -- LFT ]"
    else:
        return "[ \\  UPL ]"

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Closed-Loop Latent Model Predictive Control")
    parser.add_argument("--model_path", type=str, default="models/trained_world_model.pt", help="Path to model weights")
    parser.add_argument("--steps", type=int, default=30, help="Number of closed-loop control steps")
    parser.add_argument("--init_angle", type=float, default=0.75, help="Initial angle in radians (0=upright)")
    parser.add_argument("--method", type=str, default="cem", choices=["cem", "mppi", "random_shooting"])
    parser.add_argument("--horizon", type=int, default=12, help="Planning horizon")
    parser.add_argument("--candidates", type=int, default=128, help="Number of candidate imagined rollouts")
    args = parser.parse_args()

    print("=" * 85)
    print(f"   PHASE 4: CLOSED-LOOP LATENT MPC (Method: {args.method.upper()})")
    print("=" * 85)
    
    torch.set_num_threads(4)
    model = PhysicsInformedWorldModel(obs_dim=3, action_dim=1, latent_dim=2, hidden_dim=64)
    model_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.model_path))
    
    if os.path.exists(model_full_path):
        trainer = WorldModelTrainer(model)
        trainer.load_checkpoint(model_full_path)
        print(f"Loaded trained world model from: {model_full_path}")
    else:
        print(f"Model checkpoint not found at {model_full_path}. Please run 'python scripts/train_world_model.py' first.")
        return

    env = PendulumPhysicsEnv(m=1.0, l=1.0, g=9.81, b=0.1, max_torque=2.5, dt=0.05)
    obs = env.reset(initial_state=[args.init_angle, 0.0])
    
    planner = LatentMPCPlanner(
        world_model=model,
        horizon=args.horizon,
        num_candidates=args.candidates,
        max_torque=env.max_torque,
        dt=env.dt,
        method=args.method,
        device="cpu"
    )
    
    print(f"Starting Closed-Loop Simulation from Initial Angle: {np.degrees(args.init_angle):.1f} deg\n")
    print(f"{'Step':<6} | {'Angle (deg)':<12} | {'Vel (rad/s)':<12} | {'Torque (N*m)':<14} | {'Latency':<10} | {'Visual'}")
    print("-" * 75)
    
    angle_history = []
    latency_history = []
    total_reward = 0.0
    
    for step in range(1, args.steps + 1):
        action, info = planner.plan_action(obs)
        latency_history.append(info["elapsed_ms"])
        
        theta_rad = float(env.state[0])
        theta_deg = np.degrees(theta_rad)
        theta_vel = float(env.state[1])
        angle_history.append(theta_deg)
        
        vis = ascii_rod_visualizer(theta_rad)
        print(
            f"{step:<6} | "
            f"{theta_deg:<12.2f} | "
            f"{theta_vel:<12.2f} | "
            f"{action:<14.3f} | "
            f"{info['elapsed_ms']:<7.2f} ms | "
            f"{vis}"
        )
        
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        
    rms_angle_error = np.sqrt(np.mean(np.array(angle_history) ** 2))
    avg_latency = np.mean(latency_history)
    
    print("-" * 75)
    print(f"\nClosed-Loop Performance Summary:")
    print(f"  - Total Closed-Loop Reward:    {total_reward:.2f}")
    print(f"  - RMS Angle Error:             {rms_angle_error:.2f} degrees")
    print(f"  - Average Planning Latency:    {avg_latency:.2f} ms per step on CPU")
    print(f"  - Real-Time Execution Headroom: {(1.0 / (env.dt) - (1000.0 / avg_latency)):.1f} Hz margin")
    print(f"\n  [SUCCESS] Closed-Loop MPC Simulation Completed!")

if __name__ == "__main__":
    main()
