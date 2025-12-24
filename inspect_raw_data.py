import numpy as np
import os

def analyze_delay_data():
    data_dir = 'udata'
    delay_file = 'udelay.npy'
    path = os.path.join(data_dir, delay_file)
    
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    data = np.load(path)
    print(f"Shape: {data.shape}")
    
    threshold = 5.0
    
    # Check features separately
    for i in range(data.shape[2]):
        feat_data = data[:, :, i]
        total = feat_data.size
        delayed = np.sum(feat_data >= threshold)
        print(f"\nFeature {i}:")
        print(f"  Min: {np.nanmin(feat_data)}")
        print(f"  Max: {np.nanmax(feat_data)}")
        print(f"  Mean: {np.nanmean(feat_data)}")
        print(f"  Delayed elements: {delayed}")
        print(f"  Percentage delayed: {delayed / total * 100:.2f}%")

    # Check max over features
    max_feat_data = np.max(data, axis=2)
    total = max_feat_data.size
    delayed = np.sum(max_feat_data >= threshold)
    print(f"\nMax over features (per node, per step):")
    print(f"  Delayed elements: {delayed}")
    print(f"  Percentage delayed: {delayed / total * 100:.2f}%")

if __name__ == "__main__":
    analyze_delay_data()
