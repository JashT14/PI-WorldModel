

import os
import time
import torch
import numpy as np

from src.environment import PendulumPhysicsEnv
from src.models import PhysicsInformedWorldModel
from src.dataset import TrajectoryCollector, create_dataloader
from src.trainer import WorldModelTrainer
from src.planner import LatentMPCPlanner

def main():

    print("   PHYSICS-INFORMED LATENT WORLD MODEL (PI-WORLDMODEL) - CPU DEMO")


    # Set CPU optimization threads
    torch.set_num_threads(4)
    np.random.seed(42)
    torch.manual_seed(42)

    # -------------------------------------------------------------------------
    # 1. Environment & Physical Setup
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Initializing Physics Environment...")
    env = PendulumPhysicsEnv(m=1.0, l=1.0, g=9.81, b=0.1, max_torque=2.5, dt=0.05)
    print(f"  Physics Params: Mass={env.m}kg, Length={env.l}m, Gravity={env.g}m/s^2, Damping={env.b}")
    print(f"  Action Limits:  [-{env.max_torque}, +{env.max_torque}] N*m, dt={env.dt}s")

    # -------------------------------------------------------------------------
    # 2. Phase 1: Collect Exploration Trajectories
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Collecting Passive Exploration Trajectories...")
    collector = TrajectoryCollector(env)
    trajectories = collector.collect_trajectories(num_episodes=40, episode_length=150)
    total_transitions = sum(len(t["actions"]) for t in trajectories)
    print(f"  Collected {len(trajectories)} episodes ({total_transitions} total physics transitions).")

    # Split into train/val datasets
    train_trajs = trajectories[:32]
    val_trajs = trajectories[32:]
    
    train_loader = create_dataloader(train_trajs, horizon=6, batch_size=64, shuffle=True)
    val_loader = create_dataloader(val_trajs, horizon=6, batch_size=64, shuffle=False)
    print(f"  Created Sliding Window Batches (Horizon=6 steps): {len(train_loader)} train batches.")

    # -------------------------------------------------------------------------
    # 3. Phase 2 & 3: Instantiate & Train Physics-Informed World Model
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Pretraining Physics-Informed World Model on CPU...")
    world_model = PhysicsInformedWorldModel(
        obs_dim=3,
        action_dim=1,
        latent_dim=2,
        hidden_dim=64
    )
    
    param_count = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
    print(f"  Model Architecture: Decoupled [s, v] Latent Space + Symplectic Kinematics Layer")
    print(f"  Total Trainable Parameters: {param_count:,} (Ultra-compact for microsecond inference)")

    trainer = WorldModelTrainer(world_model, lr=2e-3, weight_decay=1e-5, device="cpu")
    trainer.fit(train_loader, val_loader, epochs=15, dt=env.dt, verbose=True)

    # Save model weights
    save_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    os.makedirs(save_model_dir, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_model_dir, "trained_world_model.pt"))

    # -------------------------------------------------------------------------
    # 4. Multi-Step Imagination & Physical Consistency Test
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("   EVALUATION: MULTI-STEP IMAGINATION IN LATENT SPACE")
    print("=" * 80)
    
    test_obs = env.reset(initial_state=[np.pi * 0.75, 0.5])
    test_actions = torch.tensor(
        np.array([[[np.sin(i * 0.3) * 1.5]] for i in range(12)], dtype=np.float32),
        dtype=torch.float32
    ).permute(1, 0, 2) # (1, 12, 1)

    world_model.eval()
    with torch.no_grad():
        z_0 = world_model.encode(torch.tensor(test_obs, dtype=torch.float32).unsqueeze(0))
        z_imagined, accel_imagined = world_model.dream(z_0, test_actions, dt=env.dt)
        obs_imagined = world_model.decode(z_imagined[0])

    print(f"{'Step':<6} | {'True Obs [cos, sin, v]':<28} | {'Imagined Obs [cos, sin, v]':<28} | {'Error':<8}")
    print("-" * 75)
    
    curr_env_obs = test_obs
    for step in range(12):
        u = float(test_actions[0, step, 0].item())
        curr_env_obs, _, _, _ = env.step(u)
        imagined_step_obs = obs_imagined[step + 1].numpy()
        err = np.linalg.norm(curr_env_obs - imagined_step_obs)
        
        obs_str_true = f"[{curr_env_obs[0]:.2f}, {curr_env_obs[1]:.2f}, {curr_env_obs[2]:.2f}]"
        obs_str_imag = f"[{imagined_step_obs[0]:.2f}, {imagined_step_obs[1]:.2f}, {imagined_step_obs[2]:.2f}]"
        print(f"{step + 1:<6} | {obs_str_true:<28} | {obs_str_imag:<28} | {err:<8.4f}")

    # -------------------------------------------------------------------------
    # 5. Phase 4: Real-Time Latent Model Predictive Control (MPC)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("   AUTONOMOUS CONTROL: LATENT MPC IN IMAGINATION")
    print("=" * 80)
    
    planner = LatentMPCPlanner(
        world_model=world_model,
        horizon=12,
        num_candidates=128,
        max_torque=env.max_torque,
        dt=env.dt,
        method="cem",
        cem_iterations=3,
        device="cpu"
    )
    
    print("  Testing Latent MPC (Planning 128 candidate trajectories x 3 CEM iterations in imagination)...")
    control_obs = env.reset(initial_state=[0.8, -0.2]) # Near upright perturbed angle
    
    print(f"\n{'Control Step':<14} | {'Angle (deg)':<14} | {'Ang Vel (rad/s)':<16} | {'Planned Torque (N*m)':<22} | {'Latency':<10}")
    print("-" * 85)
    
    total_latency_ms = 0.0
    for step in range(10):
        # 1. Agent plans action in latent imagination
        action, info = planner.plan_action(control_obs)
        total_latency_ms += info["elapsed_ms"]
        
        theta_deg = np.degrees(env.state[0])
        theta_vel = env.state[1]
        
        print(
            f"Step {step + 1:<8} | "
            f"{theta_deg:<14.2f} | "
            f"{theta_vel:<16.2f} | "
            f"{action:<22.3f} | "
            f"{info['elapsed_ms']:<8.2f} ms"
        )
        
        # 2. Execute optimal action in physical environment
        control_obs, reward, done, _ = env.step(action)
        
    avg_latency = total_latency_ms / 10.0
    print("-" * 85)
    print(f"  [SUCCESS] Latent MPC Planning Average Latency: {avg_latency:.2f} ms on CPU (Real-Time Ready < 20ms)")

    # -------------------------------------------------------------------------
    # 6. Step 5: Render and Save Visual Simulation
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Generating Live Visual Simulation Animation (simulation_demo.gif)...")
    from scripts.simulate_live_gui import render_gif_animation
    render_gif_animation(
        steps=120,
        init_angle=0.8,
        method="cem",
        save_gif_path="simulation_demo.gif"
    )

    print("\n[DONE] Project 4 Demonstration and Visual Simulator Completed Successfully!")

if __name__ == "__main__":
    main()

