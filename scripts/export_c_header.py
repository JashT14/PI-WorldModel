import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import PhysicsInformedWorldModel
from src.trainer import WorldModelTrainer

def format_c_array(arr: np.ndarray, name: str, indent: int = 4) -> str:
    """Formats a flat or 2D numpy array as a C const float array."""
    lines = []
    flat = arr.flatten()
    lines.append(f"static const float {name}[{len(flat)}] = {{")
    
    # 8 numbers per line
    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        nums_str = ", ".join(f"{x:+.8e}f" for x in chunk)
        comma = "," if i + 8 < len(flat) else ""
        lines.append(" " * indent + nums_str + comma)
        
    lines.append("};\n")
    return "\n".join(lines)

def main():
    print("=" * 75)
    print("   PHASE 5: EXPORT WEIGHTS TO PURE C11 HEADER")
    print("=" * 75)
    
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "trained_world_model.pt"))
    header_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "c_runtime", "weights.h"))
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    
    model = PhysicsInformedWorldModel(obs_dim=3, action_dim=1, latent_dim=2, hidden_dim=64)
    if os.path.exists(model_path):
        trainer = WorldModelTrainer(model)
        trainer.load_checkpoint(model_path)
        print(f"Loaded trained weights from: {model_path}")
    else:
        print("Using randomly initialized weights for header export.")

    model.eval()
    sd = model.state_dict()
    
    header_content = [
        "/* ========================================================================= */",
        "/* Auto-Generated Physics-Informed World Model Static Weights for C11 Runtime */",
        "/* Zero Dynamic Allocation (malloc) - Embedded / Microcontroller Ready      */",
        "/* ========================================================================= */",
        "",
        "#ifndef WM_WEIGHTS_H",
        "#define WM_WEIGHTS_H",
        "",
        "#define WM_OBS_DIM 3",
        "#define WM_ACTION_DIM 1",
        "#define WM_LATENT_POS_DIM 2",
        "#define WM_LATENT_VEL_DIM 2",
        "#define WM_LATENT_DIM 4",
        "#define WM_HIDDEN_DIM 64",
        ""
    ]

    # Map PyTorch layers to C weights
    for k, v in sd.items():
        arr = v.cpu().numpy()
        c_name = "wm_" + k.replace(".", "_")
        header_content.append(f"/* Layer: {k}, Shape: {arr.shape} */")
        header_content.append(format_c_array(arr, c_name))

    header_content.append("#endif /* WM_WEIGHTS_H */\n")
    
    full_text = "\n".join(header_content)
    with open(header_path, "w", encoding="utf-8") as f:
        f.write(full_text)
        
    size_kb = len(full_text.encode("utf-8")) / 1024.0
    print(f"  [SUCCESS] Exported C static weights to: {header_path} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    main()
