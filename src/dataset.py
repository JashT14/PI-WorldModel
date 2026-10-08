import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union
from .environment import PendulumPhysicsEnv

class TrajectoryCollector:
    """
    Collects structured multi-policy physical exploration trajectories.
    """
    def __init__(self, env: Optional[PendulumPhysicsEnv] = None):
        self.env = env if env is not None else PendulumPhysicsEnv()

    def generate_action(
        self,
        t_step: int,
        total_steps: int,
        mode: str = "mixed"
    ) -> float:
        max_u = self.env.max_torque
        dt = self.env.dt
        t_sec = t_step * dt
        
        if mode == "random":
            return float(np.random.uniform(-max_u, max_u))
            
        elif mode == "chirp":
            # Continuous frequency sweep from 0.2 Hz to 2.5 Hz
            freq = 0.2 + 2.3 * (t_step / total_steps)
            return float(max_u * np.sin(2.0 * np.pi * freq * t_sec))
            
        elif mode == "multisine":
            # Multi-frequency excitation for rich dynamical parameter identification
            sig = (
                0.5 * np.sin(2.0 * np.pi * 0.5 * t_sec) +
                0.3 * np.sin(2.0 * np.pi * 1.2 * t_sec) +
                0.2 * np.cos(2.0 * np.pi * 2.0 * t_sec)
            )
            return float(np.clip(max_u * sig, -max_u, max_u))
            
        elif mode == "bang_bang":
            period = 25 # Switch direction every 25 steps
            return float(max_u if (t_step // period) % 2 == 0 else -max_u)
            
        elif mode == "impulse":
            # Periodic impulse spikes
            if t_step % 30 == 0:
                return float(max_u * np.random.choice([-1.0, 1.0]))
            return 0.0
            
        else: # "mixed" policy
            r = np.random.rand()
            if r < 0.35:
                return self.generate_action(t_step, total_steps, "chirp")
            elif r < 0.60:
                return self.generate_action(t_step, total_steps, "multisine")
            elif r < 0.85:
                return self.generate_action(t_step, total_steps, "random")
            else:
                return self.generate_action(t_step, total_steps, "bang_bang")

    def collect_trajectories(
        self,
        num_episodes: int = 50,
        episode_length: int = 200,
        policy_modes: Optional[List[str]] = None
    ) -> List[Dict[str, np.ndarray]]:
        if policy_modes is None:
            policy_modes = ["chirp", "multisine", "random", "bang_bang", "impulse", "mixed"]
            
        trajectories = []
        
        for ep in range(num_episodes):
            obs = self.env.reset()
            obs_list = [obs]
            action_list = []
            reward_list = []
            energy_list = [self.env.get_total_energy()]
            
            chosen_mode = np.random.choice(policy_modes)
            
            for t in range(episode_length):
                action = self.generate_action(t, episode_length, mode=chosen_mode)
                next_obs, reward, done, info = self.env.step(action)
                
                obs_list.append(next_obs)
                action_list.append([action])
                reward_list.append(reward)
                energy_list.append(info["total_energy"])
                obs = next_obs
                
            trajectories.append({
                "observations": np.array(obs_list, dtype=np.float32), # (T + 1, obs_dim)
                "actions": np.array(action_list, dtype=np.float32),       # (T, action_dim)
                "rewards": np.array(reward_list, dtype=np.float32),       # (T,)
                "energies": np.array(energy_list, dtype=np.float32),      # (T + 1,)
                "mode": chosen_mode
            })
            
        return trajectories

    @staticmethod
    def save_trajectories(trajectories: List[Dict[str, np.ndarray]], filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # Pack array of dictionaries into npz
        np.savez_compressed(
            filepath,
            num_episodes=len(trajectories),
            **{f"obs_{i}": t["observations"] for i, t in enumerate(trajectories)},
            **{f"act_{i}": t["actions"] for i, t in enumerate(trajectories)},
            **{f"rew_{i}": t["rewards"] for i, t in enumerate(trajectories)},
            **{f"eng_{i}": t["energies"] for i, t in enumerate(trajectories)},
            **{f"mode_{i}": np.array(t["mode"]) for i, t in enumerate(trajectories)}
        )

    @staticmethod
    def load_trajectories(filepath: str) -> List[Dict[str, np.ndarray]]:
        data = np.load(filepath, allow_pickle=True)
        num_episodes = int(data["num_episodes"])
        trajectories = []
        for i in range(num_episodes):
            trajectories.append({
                "observations": data[f"obs_{i}"],
                "actions": data[f"act_{i}"],
                "rewards": data[f"rew_{i}"],
                "energies": data[f"eng_{i}"],
                "mode": str(data[f"mode_{i}"])
            })
        return trajectories


class TrajectoryDataset(Dataset):
    """
    Sliding window dataset for training multi-step autoregressive world models.
    """
    def __init__(
        self,
        trajectories: List[Dict[str, np.ndarray]],
        horizon: int = 6
    ):
        self.horizon = horizon
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        
        for traj in trajectories:
            obs = traj["observations"]  # (T + 1, obs_dim)
            actions = traj["actions"]   # (T, action_dim)
            total_steps = len(actions)
            
            # Extract sliding windows of length horizon + 1 for obs, and horizon for actions
            for t in range(total_steps - horizon + 1):
                obs_slice = obs[t : t + horizon + 1]
                act_slice = actions[t : t + horizon]
                self.samples.append((obs_slice, act_slice))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_slice, act_slice = self.samples[idx]
        return torch.tensor(obs_slice, dtype=torch.float32), torch.tensor(act_slice, dtype=torch.float32)


def create_dataloader(
    trajectories: List[Dict[str, np.ndarray]],
    horizon: int = 6,
    batch_size: int = 64,
    shuffle: bool = True
) -> DataLoader:
    dataset = TrajectoryDataset(trajectories, horizon=horizon)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
