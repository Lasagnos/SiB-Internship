import numpy as np

dataset_path = 'dataset_lithium.npz'
print(f"Loading dataset from '{dataset_path}'...")
dataset = np.load(dataset_path)
X_raw, Y_raw, T_raw = dataset['X'], dataset['Y'], dataset['T']
num_samples = X_raw.shape[0]

# 1. SPLIT before normalization: train/val/test
TRAIN_SPLIT, VAL_SPLIT, SEED = 0.8, 0.1, 42
rng = np.random.default_rng(SEED)
perm = rng.permutation(num_samples)

train_size = int(TRAIN_SPLIT * num_samples)
val_size = int(VAL_SPLIT * num_samples)
train_idx = perm[:train_size]
val_idx = perm[train_size:train_size + val_size]
test_idx = perm[train_size + val_size:]

# 2. FIT only on train set
X_train, Y_train = X_raw[train_idx], Y_raw[train_idx]
I_min, I_max = X_train[:, 0].min(), X_train[:, 0].max() # Feature 0: Current (Linear scale)
Da_min, Da_max = np.log10(X_train[:, 1]).min(), np.log10(X_train[:, 1]).max()   # Feature 1: Anode Diffusivity (Logarithmic scale)
Dc_min, Dc_max = np.log10(X_train[:, 2]).min(), np.log10(X_train[:, 2]).max()   # Feature 2: Cathode Diffusivity (Logarithmic scale)
De_min, De_max = np.log10(X_train[:, 3]).min(), np.log10(X_train[:, 3]).max()   # Feature 3: Electrolyte Diffusivity (Logarithmic scale)
Y_min, Y_max = Y_train.min(), Y_train.max() # Output Y: Electrolyte Concentration (Linear scale)

# 3. APPLY to all sets (train, val, test)
X_norm = np.zeros_like(X_raw)
X_norm[:, 0] = 2.0 * ((X_raw[:, 0] - I_min) / (I_max - I_min)) - 1.0
X_norm[:, 1] = 2.0 * ((np.log10(X_raw[:, 1]) - Da_min) / (Da_max - Da_min)) - 1.0
X_norm[:, 2] = 2.0 * ((np.log10(X_raw[:, 2]) - Dc_min) / (Dc_max - Dc_min)) - 1.0
X_norm[:, 3] = 2.0 * ((np.log10(X_raw[:, 3]) - De_min) / (De_max - De_min)) - 1.0
Y_norm = 2.0 * ((Y_raw - Y_min) / (Y_max - Y_min)) - 1.0

# CAREFUL, val/test ranges could break the [-1, 1] assumption slightly
# But too much means the training set does not cover the full range of the input space, which is a problem for generalization.
print(f"X val range:  [{X_norm[val_idx].min():.2f}, {X_norm[val_idx].max():.2f}]")
print(f"X test range: [{X_norm[test_idx].min():.2f}, {X_norm[test_idx].max():.2f}]")

print("NORMALIZATION CHECK")
print(f"X Tensor -> Min: {X_norm.min():.2f}, Max: {X_norm.max():.2f}, Mean: {X_norm.mean():.2f}")
print(f"Y Tensor -> Min: {Y_norm.min():.2f}, Max: {Y_norm.max():.2f}, Mean: {Y_norm.mean():.2f}")
if np.isnan(X_norm).any() or np.isnan(Y_norm).any():
    print("WARNING: NaNs detected in normalized tensors!")

# 4. SAVE (also save the indices and scaling factors for later use in the inference)
np.savez_compressed(
    'dataset_lithium_normalized.npz',
    X_norm=X_norm, Y_norm=Y_norm, T_raw=T_raw,
    train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    scaling_factors=np.array([I_min, I_max, Da_min, Da_max, Dc_min, Dc_max, De_min, De_max, Y_min, Y_max]),
)


                                    
# Loading dataset from 'dataset_lithium.npz'...
# X val range:  [-0.98, 0.99]
# X test range: [-1.00, 1.01]
# NORMALIZATION CHECK
# X Tensor -> Min: -1.00, Max: 1.01, Mean: -0.01
# Y Tensor -> Min: -1.00, Max: 1.00, Mean: -0.47