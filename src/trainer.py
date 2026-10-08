import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
from .models import PhysicsInformedWorldModel
from .losses import PhysicsInformedWorldModelLoss
from .dataset import TrajectoryDataset

class WorldModelTrainer:
    """
    CPU-optimized multi-step trainer for Physics-Informed Latent World Models
    with curriculum horizon scheduling and checkpoint persistence.
    """
    def __init__(
        self,
        model: PhysicsInformedWorldModel,
        loss_fn: Optional[PhysicsInformedWorldModelLoss] = None,
        lr: float = 2e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn if loss_fn is not None else PhysicsInformedWorldModelLoss()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-5
        )
        
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "loss_recon": [],
            "loss_latent": [],
            "loss_future": [],
            "loss_energy": []
        }

    def train_epoch(self, dataloader: DataLoader, dt: float = 0.05) -> Dict[str, float]:
        self.model.train()
        total_epoch_loss = 0.0
        metric_sums = {"loss_recon": 0.0, "loss_latent": 0.0, "loss_future": 0.0, "loss_energy": 0.0}
        num_batches = len(dataloader)

        for obs_batch, act_batch in dataloader:
            obs_batch = obs_batch.to(self.device)
            act_batch = act_batch.to(self.device)
            
            self.optimizer.zero_grad()
            loss, metrics = self.loss_fn(self.model, obs_batch, act_batch, dt=dt)
            loss.backward()
            
            # Gradient clipping to prevent instability in multi-step unrolling
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            
            total_epoch_loss += loss.item()
            for k in metric_sums:
                metric_sums[k] += metrics[k]

        avg_loss = total_epoch_loss / max(1, num_batches)
        avg_metrics = {k: v / max(1, num_batches) for k, v in metric_sums.items()}
        avg_metrics["train_loss"] = avg_loss
        return avg_metrics

    def evaluate(self, dataloader: DataLoader, dt: float = 0.05) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_val_loss = 0.0
        metric_sums = {"loss_recon": 0.0, "loss_latent": 0.0, "loss_future": 0.0, "loss_energy": 0.0}
        num_batches = len(dataloader)

        with torch.no_grad():
            for obs_batch, act_batch in dataloader:
                obs_batch = obs_batch.to(self.device)
                act_batch = act_batch.to(self.device)
                loss, metrics = self.loss_fn(self.model, obs_batch, act_batch, dt=dt)
                total_val_loss += loss.item()
                for k in metric_sums:
                    metric_sums[k] += metrics[k]

        avg_loss = total_val_loss / max(1, num_batches)
        avg_metrics = {k: v / max(1, num_batches) for k, v in metric_sums.items()}
        avg_metrics["val_loss"] = avg_loss
        return avg_loss, avg_metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 25,
        dt: float = 0.05,
        save_best_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        if verbose:
            print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Recon':<10} | {'Latent':<10} | {'Energy':<10}")
            print("-" * 75)

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, dt=dt)
            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(val_loader, dt=dt)
            else:
                val_loss = train_metrics["train_loss"]
                val_metrics = train_metrics
                
            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["val_loss"].append(val_loss)
            self.history["loss_recon"].append(train_metrics["loss_recon"])
            self.history["loss_latent"].append(train_metrics["loss_latent"])
            self.history["loss_future"].append(train_metrics["loss_future"])
            self.history["loss_energy"].append(train_metrics["loss_energy"])

            if val_loss < best_val_loss and save_best_path is not None:
                best_val_loss = val_loss
                self.save_checkpoint(save_best_path)

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                print(
                    f"{epoch:<8} | "
                    f"{train_metrics['train_loss']:<12.6f} | "
                    f"{val_loss:<12.6f} | "
                    f"{train_metrics['loss_recon']:<10.5f} | "
                    f"{train_metrics['loss_latent']:<10.5f} | "
                    f"{train_metrics['loss_energy']:<10.5f}"
                )

        return self.history

    def save_checkpoint(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "latent_dim": self.model.latent_dim,
            "obs_dim": self.model.obs_dim,
            "action_dim": self.model.action_dim
        }, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
