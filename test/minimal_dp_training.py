#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the full STPN model with differential privacy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# Import your STPN model
try:
    from model import STPN
    print("✓ Successfully imported STPN")
except ImportError as e:
    print(f"✗ Cannot import STPN: {e}")
    exit(1)

def create_dummy_data():
    """Create dummy data that matches STPN input requirements"""
    batch_size = 4
    in_channels = 2
    num_nodes = 10
    in_len = 12
    out_len = 12
    num_weather = 8
    
    # Input data: (batch, in_channels, nodes, time)
    x = torch.randn(batch_size, in_channels, num_nodes, in_len)
    
    # Target data: (batch, out_channels, nodes, time)
    y = torch.randn(batch_size, 2, num_nodes, out_len)  # Assuming out_channels=2
    
    # Time inputs
    t_in = torch.randn(batch_size, in_len)
    t_out = torch.randn(batch_size, out_len)
    
    # Weather data: (batch, nodes, time)
    w = torch.randint(0, num_weather, (batch_size, num_nodes, in_len))
    
    # Adjacency matrices (supports)
    supports = [torch.randn(num_nodes, num_nodes) for _ in range(3)]
    
    return x, y, t_in, t_out, w, supports

def test_stpn_forward_pass():
    """Test STPN forward pass without DP"""
    print("=== Testing STPN Forward Pass ===")
    
    # Create model with minimal complexity
    model = STPN(
        h_layers=1,
        in_channels=2,
        hidden_channels=[16, 8],
        out_channels=2,
        emb_size=8,
        dropout=0.0,
        wemb_size=4,
        time_d=4,
        heads=2,
        support_len=3,
        order=2,
        num_weather=8,
        use_se=False,  # Disable SE for now
        use_cov=True
    )
    
    # Create dummy data
    x, y, t_in, t_out, w, supports = create_dummy_data()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    try:
        model.eval()
        with torch.no_grad():
            output = model(x, t_in, supports, t_out, w)
            print(f"✓ Forward pass successful. Output shape: {output.shape}")
            print(f"Expected output shape: {y.shape}")
            return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stpn_dp_compatibility():
    """Test STPN with differential privacy"""
    print("\n=== Testing STPN with Differential Privacy ===")
    
    # Create model
    model = STPN(
        h_layers=1,
        in_channels=2,
        hidden_channels=[16, 8],
        out_channels=2,
        emb_size=8,
        dropout=0.0,
        wemb_size=4,
        time_d=4,
        heads=2,
        support_len=3,
        order=2,
        num_weather=8,
        use_se=False,  # Start without SE
        use_cov=True
    )
    
    # Create dummy data
    x, y, t_in, t_out, w, supports = create_dummy_data()
    
    # Validate model
    print("Validating model for DP...")
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"Validation errors found: {len(errors)}")
        for i, error in enumerate(errors[:5]):  # Show first 5
            print(f"  {i+1}. {error}")
    else:
        print("✓ Model passes DP validation")
    
    # Fix model
    print("Fixing model...")
    try:
        model = ModuleValidator.fix(model)
        print("✓ Model fixed successfully")
    except Exception as e:
        print(f"✗ Model fix failed: {e}")
        return False
    
    # Create simple dataset
    class SimpleSTPN_Dataset(torch.utils.data.Dataset):
        def __init__(self, x, y, t_in, t_out, w, n_samples=20):
            self.x = x[:n_samples] if len(x) >= n_samples else x
            self.y = y[:n_samples] if len(y) >= n_samples else y
            self.t_in = t_in[:n_samples] if len(t_in) >= n_samples else t_in
            self.t_out = t_out[:n_samples] if len(t_out) >= n_samples else t_out
            self.w = w[:n_samples] if len(w) >= n_samples else w
            
        def __len__(self):
            return len(self.x)
        
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx], self.t_in[idx], self.t_out[idx], self.w[idx]
    
    # Expand data for dataset
    x_expanded = x.repeat(10, 1, 1, 1)  # Repeat to get more samples
    y_expanded = y.repeat(10, 1, 1, 1)
    t_in_expanded = t_in.repeat(10, 1)
    t_out_expanded = t_out.repeat(10, 1)
    w_expanded = w.repeat(10, 1, 1)
    
    dataset = SimpleSTPN_Dataset(x_expanded, y_expanded, t_in_expanded, t_out_expanded, w_expanded)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # CRITICAL: Set to training mode and verify
    model.train()
    print(f"Model training mode: {model.training}")
    
    # Check all submodules
    non_training_modules = []
    for name, module in model.named_modules():
        if not module.training:
            non_training_modules.append(name)
    
    if non_training_modules:
        print(f"⚠ Non-training modules found: {len(non_training_modules)}")
        print("Forcing all modules to training mode...")
        for module in model.modules():
            module.train()
        print("✓ All modules set to training mode")
    else:
        print("✓ All modules already in training mode")
    
    # Setup PrivacyEngine
    print("Setting up PrivacyEngine...")
    try:
        privacy_engine = PrivacyEngine()
        
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=0.5,
            max_grad_norm=1.0,
        )
        print("✓ PrivacyEngine setup successful!")
        
    except Exception as e:
        print(f"✗ PrivacyEngine setup failed: {e}")
        
        # Debug which modules are problematic
        print("\nDebugging model state...")
        print(f"Top-level model training mode: {model.training}")
        
        non_training = []
        for name, module in model.named_modules():
            if not module.training:
                non_training.append(name)
        
        if non_training:
            print(f"Non-training modules: {non_training[:10]}")
        
        return False
    
    # Test training loop
    print("Testing training loop...")
    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 2:  # Only test 2 batches
                break
                
            x_batch, y_batch, t_in_batch, t_out_batch, w_batch = batch
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(x_batch, t_in_batch, supports, t_out_batch, w_batch)
            
            # Simple MSE loss
            loss = nn.MSELoss()(output, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        print("✓ STPN DP training successful!")
        return True
        
    except Exception as e:
        print(f"✗ STPN DP training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_different_configurations():
    """Test STPN with different configurations to find working setup"""
    print("\n=== Testing Different STPN Configurations ===")
    
    configurations = [
        {"use_se": False, "use_cov": False, "heads": 1, "h_layers": 0, "description": "Minimal"},
        {"use_se": False, "use_cov": True, "heads": 1, "h_layers": 0, "description": "Minimal + Covariate"},
        {"use_se": False, "use_cov": True, "heads": 2, "h_layers": 1, "description": "Standard"},
        {"use_se": True, "use_cov": True, "heads": 2, "h_layers": 1, "description": "With SE"},
    ]
    
    for config in configurations:
        print(f"\nTesting {config['description']} configuration...")
        
        try:
            model = STPN(
                h_layers=config["h_layers"],
                in_channels=2,
                hidden_channels=[16, 8],
                out_channels=2,
                emb_size=8,
                dropout=0.0,
                wemb_size=4,
                time_d=4,
                heads=config["heads"],
                support_len=3,
                order=2,
                num_weather=8,
                use_se=config["use_se"],
                use_cov=config["use_cov"]
            )
            
            # Quick validation test
            errors = ModuleValidator.validate(model, strict=False)
            if len(errors) == 0:
                print(f"✓ {config['description']}: Passes validation")
            else:
                print(f"✗ {config['description']}: {len(errors)} validation errors")
                
        except Exception as e:
            print(f"✗ {config['description']}: Creation failed - {e}")

if __name__ == "__main__":
    # Step 1: Test basic forward pass
    if not test_stpn_forward_pass():
        print("Basic forward pass failed. Check your model implementation.")
        exit(1)
    
    # Step 2: Test different configurations
    test_with_different_configurations()
    
    # Step 3: Test full DP integration
    if test_stpn_dp_compatibility():
        print("\n" + "="*50)
        print("🎉 STPN works with Differential Privacy!")
        print("You can now use this configuration in your main training script.")
    else:
        print("\n" + "="*50)
        print("❌ STPN DP integration failed.")
        print("Try using the minimal configuration that worked in the tests above.")