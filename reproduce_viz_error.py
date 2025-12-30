
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from visualize_training_classification import visualize_classification_results
except ImportError:
    print("Could not import visualize_classification_results")
    sys.exit(1)

def test_visualization():
    print("Testing visualization with mixed types...")
    
    # Simulate the scenario: y_true is float (soft labels), y_pred is int (binary)
    y_true = np.array([0.0, 1.0, 0.0, 1.0, 0.5, 0.3])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    
    # Dummy regression values
    y_reg_true = y_true * 10
    y_reg_pred = y_pred * 10
    
    try:
        visualize_classification_results(
            y_true,
            y_pred,
            y_reg_true,
            y_reg_pred,
            threshold=0.5,
            save_path="test_viz_fail.png"
        )
        print("Visualization SUCCESS (Unexpected)")
    except Exception as e:
        print(f"Visualization FAILED as expected: {e}")

    print("\nTesting visualization with fix (binarized targets)...")
    
    # Apply the fix: binarize y_true
    y_true_bin = (y_true >= 0.5).astype(int)
    
    try:
        visualize_classification_results(
            y_true_bin,
            y_pred,
            y_reg_true,
            y_reg_pred,
            threshold=0.5,
            save_path="test_viz_success.png"
        )
        print("Visualization SUCCESS with fix")
    except Exception as e:
        print(f"Visualization FAILED with fix: {e}")

if __name__ == "__main__":
    test_visualization()
