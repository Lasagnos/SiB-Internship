import numpy as np

dataset_path = 'dataset_lithium.npz'  # Path to the original dataset file
print(f"Loading dataset from '{dataset_path}'...")
dataset = np.load(dataset_path)

X_raw = dataset['X']          # Shape: (N, 4)
Y_raw = dataset['Y']          # Shape: (N, 64, 100)
T_raw = dataset['T']          # Shape: (N,)

num_samples = X_raw.shape[0]

# print("="*60)
# print("   RAW DATASET STATISTICS (BEFORE NORMALIZATION)")
# print("="*60)
# param_names = [
#     "Current [A]", 
#     "Anode Diffusivity [m2/s]", 
#     "Cathode Diffusivity [m2/s]", 
#     "Electrolyte Diffusivity [m2/s]"
# ]
# for i in range(4):
#     data = X_raw[:, i]
#     print(f"--- {param_names[i]} ---")
#     print(f"  Min:      {np.min(data):.4e}")
#     print(f"  Max:      {np.max(data):.4e}")
#     print(f"  Mean:     {np.mean(data):.4e}")
#     print(f"  Median:   {np.median(data):.4e}")
#     print(f"  Variance: {np.var(data):.4e}\n")
# print(f"--- Output Y (Concentration [mol/m3]) ---")
# print(f"  Global Min: {np.min(Y_raw):.4e}")
# print(f"  Global Max: {np.max(Y_raw):.4e}\n")


print("Normalizing features to [-1, 1] range (Zero-Centered Deep Learning Standard)...")
X_norm = np.zeros_like(X_raw)

# Feature 0: Current (Linear scale)
I_min, I_max = X_raw[:, 0].min(), X_raw[:, 0].max()
X_norm[:, 0] = 2.0 * ((X_raw[:, 0] - I_min) / (I_max - I_min)) - 1.0

# Feature 1: Anode Diffusivity (Logarithmic scale)
Da_log = np.log10(X_raw[:, 1])
Da_min, Da_max = Da_log.min(), Da_log.max()
X_norm[:, 1] = 2.0 * ((Da_log - Da_min) / (Da_max - Da_min)) - 1.0

# Feature 2: Cathode Diffusivity (Logarithmic scale)
Dc_log = np.log10(X_raw[:, 2])
Dc_min, Dc_max = Dc_log.min(), Dc_log.max()
X_norm[:, 2] = 2.0 * ((Dc_log - Dc_min) / (Dc_max - Dc_min)) - 1.0

# Feature 3: Electrolyte Diffusivity (Logarithmic scale)
De_log = np.log10(X_raw[:, 3])
De_min, De_max = De_log.min(), De_log.max()
X_norm[:, 3] = 2.0 * ((De_log - De_min) / (De_max - De_min)) - 1.0

# Output Y: Electrolyte Concentration (Linear scale)
Y_min, Y_max = Y_raw.min(), Y_raw.max()
# Scaling Y from [0, 1] to [-1, 1] is critical for FNO activation functions
Y_norm = 2.0 * ((Y_raw - Y_min) / (Y_max - Y_min)) - 1.0
#print("NORMALIZATION COMPLETE")


print("="*60)
print("   NORMALIZATION CHECK")
print("="*60)
print(f"X Tensor -> Min: {X_norm.min():.2f}, Max: {X_norm.max():.2f}, Mean: {X_norm.mean():.2f}")
print(f"Y Tensor -> Min: {Y_norm.min():.2f}, Max: {Y_norm.max():.2f}, Mean: {Y_norm.mean():.2f}")
if np.isnan(X_norm).any() or np.isnan(Y_norm).any():
    print("WARNING: NaNs detected in normalized tensors!")
else:
    print("Check Passed: Tensors are strictly bound between -1 and 1 and zero-centered.")

# Save the normalized dataset using .npz to keep it unified
normalized_filename = 'dataset_lithium_normalized.npz'
np.savez_compressed(
    normalized_filename, 
    X_norm=X_norm, 
    Y_norm=Y_norm, 
    T_raw=T_raw, # We keep durations raw for analysis (they are not FNO inputs)
    
    # We save the scaling factors to easily invert the [-1, 1] scaling later
    scaling_factors=np.array([I_min, I_max, Da_min, Da_max, Dc_min, Dc_max, De_min, De_max, Y_min, Y_max])
)

print(f"\nNormalized dataset and scaling factors saved to '{normalized_filename}'.")