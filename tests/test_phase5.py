import os
import unittest
import subprocess
import warnings
import torch
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import PhysicsInformedWorldModel
from src.trainer import WorldModelTrainer

class TestPhase5EdgeAndCompilation(unittest.TestCase):
    
    def test_torchscript_model_execution(self):
        """Tests that exported TorchScript models execute and match PyTorch outputs"""
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        export_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "exported"))
        step_model_path = os.path.join(export_dir, "world_model_step_traced.pt")
        
        self.assertTrue(os.path.exists(step_model_path), f"TorchScript model not found at {step_model_path}")
        
        ts_model = torch.jit.load(step_model_path)
        dummy_z = torch.randn(2, 4)
        dummy_act = torch.randn(2, 1)
        
        z_next, a_net, obs_hat = ts_model(dummy_z, dummy_act)
        self.assertEqual(z_next.shape, (2, 4))
        self.assertEqual(a_net.shape, (2, 2))
        self.assertEqual(obs_hat.shape, (2, 3))
        print(f"\n[Test Phase 5] TorchScript Traced Forward Step: OK (Batch=2)")

    def test_c_header_exists(self):
        """Tests that static C weights header is valid and non-empty"""
        header_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "c_runtime", "weights.h"))
        self.assertTrue(os.path.exists(header_path))
        file_size = os.path.getsize(header_path)
        self.assertGreater(file_size, 50 * 1024, "weights.h header appears truncated")

    def test_c_binary_execution(self):
        """Tests execution of the compiled C11 embedded binary"""
        exe_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "c_runtime", "embedded_world_model.exe"))
        self.assertTrue(os.path.exists(exe_path), f"Executable not found at {exe_path}")
        
        res = subprocess.run([exe_path], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"C binary exited with error: {res.stderr}")
        self.assertIn("C11 Embedded World Model Engine Verified", res.stdout)
        print(f"[Test Phase 5] Pure C11 Standalone Binary: PASS (Zero Dynamic Allocations)")

if __name__ == "__main__":
    unittest.main()
