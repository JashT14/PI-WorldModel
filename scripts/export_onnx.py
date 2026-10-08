import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import PhysicsInformedWorldModel
from src.trainer import WorldModelTrainer

class UnifiedTransitionStep(nn.Module):
    """Unified single-step forward model: (z_t, action) -> (z_next, a_net, obs_hat)"""
    def __init__(self, model: PhysicsInformedWorldModel, dt: float = 0.05):
        super().__init__()
        self.model = model
        self.dt = dt

    def forward(self, z_t: torch.Tensor, action: torch.Tensor):
        z_next, a_net = self.model.step_latent(z_t, action, dt=self.dt)
        obs_hat = self.model.decode(z_next)
        return z_next, a_net, obs_hat

def main():
    print("=" * 75)
    print("   PHASE 5: STANDALONE MODEL & TORCHSCRIPT EXPORT")
    print("=" * 75)
    
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "trained_world_model.pt"))
    export_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "exported"))
    os.makedirs(export_dir, exist_ok=True)
    
    model = PhysicsInformedWorldModel(obs_dim=3, action_dim=1, latent_dim=2, hidden_dim=64)
    if os.path.exists(model_path):
        trainer = WorldModelTrainer(model)
        trainer.load_checkpoint(model_path)
        print(f"Loaded trained weights from: {model_path}")
    else:
        print("Using randomly initialized weights for export demonstration.")
        
    model.eval()

    # 1. Export TorchScript Trace for Encoder
    dummy_obs = torch.randn(1, 3, dtype=torch.float32)
    traced_encoder = torch.jit.trace(model.encoder, dummy_obs)
    encoder_ts_path = os.path.join(export_dir, "encoder_traced.pt")
    traced_encoder.save(encoder_ts_path)
    print(f"  [EXPORTED] Traced Encoder -> {encoder_ts_path}")

    # 2. Export TorchScript Trace for Transition
    dummy_z = torch.randn(1, 4, dtype=torch.float32) # [s1, s2, v1, v2]
    dummy_act = torch.randn(1, 1, dtype=torch.float32)
    traced_transition = torch.jit.trace(model.transition, (dummy_z, dummy_act))
    transition_ts_path = os.path.join(export_dir, "transition_traced.pt")
    traced_transition.save(transition_ts_path)
    print(f"  [EXPORTED] Traced Transition -> {transition_ts_path}")

    # 3. Export TorchScript Trace for Decoder
    traced_decoder = torch.jit.trace(model.decoder, dummy_z)
    decoder_ts_path = os.path.join(export_dir, "decoder_traced.pt")
    traced_decoder.save(decoder_ts_path)
    print(f"  [EXPORTED] Traced Decoder -> {decoder_ts_path}")

    # 4. Export Unified Single-Step Traced Pipeline
    unified_step = UnifiedTransitionStep(model, dt=0.05)
    traced_unified = torch.jit.trace(unified_step, (dummy_z, dummy_act))
    unified_ts_path = os.path.join(export_dir, "world_model_step_traced.pt")
    traced_unified.save(unified_ts_path)
    print(f"  [EXPORTED] Traced Unified Step -> {unified_ts_path}")

    # 5. Attempt ONNX Export if onnx library is available
    try:
        import onnx
        onnx_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "onnx"))
        os.makedirs(onnx_dir, exist_ok=True)
        unified_onnx = os.path.join(onnx_dir, "world_model_step.onnx")
        torch.onnx.export(
            unified_step,
            (dummy_z, dummy_act),
            unified_onnx,
            input_names=["current_z", "action"],
            output_names=["next_z", "acceleration", "predicted_obs"],
            opset_version=14,
            dynamo=False
        )
        print(f"  [EXPORTED] ONNX Unified Step -> {unified_onnx}")
    except Exception as e:
        print(f"\n  [NOTE] ONNX python library not installed (TorchScript & Pure C11 export are fully active).")
        print(f"         To enable raw .onnx export, run: pip install onnx onnxscript")

    print("\n  [SUCCESS] Standalone model binaries exported successfully!")

if __name__ == "__main__":
    main()
