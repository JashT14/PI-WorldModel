"""
Physics-Informed Latent World Model (PI-WorldModel)
Core package for kinematic-constrained latent dynamics and MPC planning.
"""

from .environment import PendulumPhysicsEnv, CartPolePhysicsEnv
from .models import (
    StateEncoder,
    SymplecticKinematicsTransitionNet,
    StateDecoder,
    PhysicsInformedWorldModel,
    UnconstrainedWorldModel
)
from .losses import PhysicsInformedWorldModelLoss
from .dataset import TrajectoryCollector, TrajectoryDataset, create_dataloader
from .trainer import WorldModelTrainer
from .planner import LatentMPCPlanner

__all__ = [
    "PendulumPhysicsEnv",
    "CartPolePhysicsEnv",
    "StateEncoder",
    "SymplecticKinematicsTransitionNet",
    "StateDecoder",
    "PhysicsInformedWorldModel",
    "UnconstrainedWorldModel",
    "PhysicsInformedWorldModelLoss",
    "TrajectoryCollector",
    "TrajectoryDataset",
    "create_dataloader",
    "WorldModelTrainer",
    "LatentMPCPlanner"
]
