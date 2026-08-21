import pybamm
import numpy as np
from scipy.stats import qmc
from collections import Counter
# import warnings

print("="*60)
print("   INITIALIZING VIRTUAL CELL DESIGN & LHS SAMPLING")
print("="*60)

param_li = pybamm.ParameterValues("Chen2020")       # "Target geometry" (Commercial 21700 Cell, 5Ah)
#param_na = pybamm.ParameterValues("Chayambuka2022") # Sodium chemistry (Hard Carbon / NVPF)

# Explicitly set the voltage cut-off for the Lithium cell to 2.5 Volts
param_li["Lower voltage cut-off [V]"] = 2.5   # Default, but we set it explicitly for clarity

# --- 16/8/2026 - Capture the electrolyte diffusivity curve before overriding anything ---
# Chen2020 uses a concentration-dependent function (Nyman2008) for electrolyte diffusivity, same situation as Chayambuka2022's three functions for sodium.
# Anode/cathode diffusivity for lithium ARE plain scalar constants (3.3e-14, 4e-15), only the electrolyte needs a scale-factor.
base_D_electrolyte = param_li["Electrolyte diffusivity [m2.s-1]"]  # electrolyte_diffusivity_Nyman2008(c_e, T)
def scaled_D_electrolyte(c_e, T):
    return pybamm.InputParameter("Electrolyte diffusivity scale factor") * base_D_electrolyte(c_e, T)

# Tell PyBaMM which parameters will be input (Current and Diffusivity)
param_li["Current function [A]"] = "[input]"
param_li["Negative electrode diffusivity [m2.s-1]"] = "[input]"     # Anode (Graphite)
param_li["Positive electrode diffusivity [m2.s-1]"] = "[input]"     # Cathode (NMC)
param_li["Electrolyte diffusivity [m2.s-1]"] = scaled_D_electrolyte # Liquid Electrolyte (Nyman2008)

# We want a total of 64 spatial nodes across the x-axis (21 + 20 + 21 = 62, plus 2 for the boundaries = 64)
# This is done because powers of 2 (like 64) are highly optimized for Fast Fourier Transforms in PyTorch
var_pts = {
    "x_n": 21,  # negative electrode (anode)
    "x_s": 20,  # separator
    "x_p": 21,  # positive electrode (cathode)
    "r_n": 10,  # particle radius (internal, not outputted to FNO)
    "r_p": 10   
}

# We treat PyBaMM numerical warnings as actual errors to discard unstable simulations
# warnings.filterwarnings("error", category=pybamm.SolverWarning) #Disabled, too severe currently

# Load the DFN/P2D model for Lithium (lithium_ion.DFN instead of sodium_ion.BasicDFN)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, parameter_values=param_li, var_pts=var_pts, solver=pybamm.CasadiSolver(mode="safe"))
t_eval_max = [0, 40000] # Time limit
pybamm.set_logging_level("ERROR")   # ignore warnings

# =================================================================

# Sanity check: Run some simulations with the default parameters to ensure everything is working before running the full LHS loop
print("\nRunning sanity-check solves (k=1.0 on electrolyte factor)...")
for test_current in [2.0, 12.0]:    # Check both low and high currents to ensure the model is stable across the range
    try:
        sim.solve(
            t_eval=t_eval_max,
            inputs={
                "Current function [A]": test_current,
                "Negative electrode diffusivity [m2.s-1]": 3.3e-14,   # Chen2020 default
                "Positive electrode diffusivity [m2.s-1]": 4e-15,     # Chen2020 default
                "Electrolyte diffusivity scale factor": 1.0,
            }
        )
        print(f"  I={test_current} A: PASSED. Discharge duration: {sim.solution['Time [s]'].entries[-1]:.1f} s")
    except Exception as e:
        print(f"  I={test_current} A: FAILED ({type(e).__name__}): {e}")
        print("  Stop here and check before running the full LHS loop.")
        raise

# =================================================================

# HYPERPARAMETERS for the LHS sampling of the input space
TARGET_SAMPLES = 500
POOL_SIZE = 3000    # Number of random samples to generate each time
NUM_PARAMETERS = 4  # Current, Anode/Cathode/Electrolyte Diffusivity
RANDOM_SEED = 42

sampler = qmc.LatinHypercube(d=NUM_PARAMETERS, seed=RANDOM_SEED)
raw_samples = sampler.random(n=POOL_SIZE)

# Parameter 1: Current [A] (0.5 to 15.0 - Lithium handles high C-rates better) -- unchanged
I_min, I_max = 0.5, 15.0
I_lhs = I_min + raw_samples[:, 0] * (I_max - I_min)

# Parameter 2: Anode Diffusivity [m2/s] (Graphite: 1e-14 to 1e-12) -- unchanged, default 3.3e-14 already
Ds_n_lhs = 10 ** (-14.0 + raw_samples[:, 1] * 2.0)

# Parameter 3: Cathode Diffusivity [m2/s] (NMC: 1e-15 to 1e-13) -- unchanged, default 4e-15 already
Ds_p_lhs = 10 ** (-15.0 + raw_samples[:, 2] * 2.0)

# Parameter 4: Electrolyte diffusivity SCALE FACTOR, log-uniform 0.3x to 3x, applied to the Nyman2008 concentration-dependent curve.
# Same range as the sodium factor, maybe it's best to change? DEBUG HERE
K_MIN, K_MAX = 0.3, 3.0
log_k_min, log_k_max = np.log10(K_MIN), np.log10(K_MAX)
k_electrolyte_lhs = 10 ** (log_k_min + raw_samples[:, 3] * (log_k_max - log_k_min))

X_pool = np.column_stack((I_lhs, Ds_n_lhs, Ds_p_lhs, k_electrolyte_lhs))

print(f"\nSampling ranges:")
print(f"  Current:                    {I_min}-{I_max} A")
print(f"  Anode diffusivity:          1e-14 to 1e-12 m2/s (absolute, unchanged)")
print(f"  Cathode diffusivity:        1e-15 to 1e-13 m2/s (absolute, unchanged)")
print(f"  Electrolyte diff. factor:   {K_MIN}x-{K_MAX}x  (applied to electrolyte_diffusivity_Nyman2008)")

# =================================================================

dataset_y = []       
valid_inputs = []
durations = [] # For analyzing it later
failure_counts = Counter()

print(f"\nBeginning simulations... (The process will stop once {TARGET_SAMPLES} valid samples are extracted)")

for i in range(POOL_SIZE):
    # Stop if we have reached the target number of valid samples
    if len(valid_inputs) == TARGET_SAMPLES:
        print(f"\n TARGET REACHED: {TARGET_SAMPLES} valid simulations extracted!")
        break
        
    try:
        sim.solve(
            t_eval=t_eval_max, 
            inputs={
                "Current function [A]": X_pool[i, 0],
                "Negative electrode diffusivity [m2.s-1]": X_pool[i, 1],
                "Positive electrode diffusivity [m2.s-1]": X_pool[i, 2],
                "Electrolyte diffusivity scale factor": X_pool[i, 3]
            }
        )
        
        t_max = sim.solution["Time [s]"].entries[-1]    # Extract the actual final time before the voltage cut-off
        if t_max < 1.0:
            continue    # We discard simulations that failed instantly (i.e. lasted less than 1 second)

        t_eval_uniform = np.linspace(0, t_max, 100)     # Temporal Normalization (100 uniform snapshots from start to finish)  
        
        # Spatial map extraction for Electrolyte Concentration
        c_e_obj = sim.solution["Electrolyte concentration [mol.m-3]"]
        c_e_uniform = c_e_obj(t=t_eval_uniform) # Shape will now be (64, 100)
        
        # Store successful data
        dataset_y.append(c_e_uniform)
        valid_inputs.append(X_pool[i])
        durations.append(t_max)
        
        if (len(valid_inputs)) % 25 == 0: 
            print(f"Progress: {len(valid_inputs)} valid simulations extracted...")
            
    # except Exception as e:
    #     pass    # Ignore simulations that cause errors
    except pybamm.SolverError:
        # Expected failures due to numerical instability in some simulations, we just count them
        failure_counts["SolverError"] += 1
        continue
    except Exception as e:
        exc_name = type(e).__name__
        failure_counts[exc_name] += 1
        if failure_counts[exc_name] <= 3:
            print(f"  [!] {exc_name} unexpected at sample {i}: {e}")
        continue

print("\nFailures recap:")
for exc_name, count in failure_counts.most_common():
    print(f"  {exc_name}: {count}")

# =================================================================

X_final = np.array(valid_inputs)
Y_final = np.array(dataset_y)
T_final = np.array(durations)

print("\n" + "="*50)
print("   DATASET SUCCESSFULLY GENERATED")
print("="*50)
print(f"Target valid simulations requested: {TARGET_SAMPLES}")
print(f"Total simulations attempted by LHS: {i + 1}")
print(f"Valid stable simulations stored: {X_final.shape[0]}")
print(f"Discarded due to numerical instability: {(i + 1) - X_final.shape[0]}")
print(f"X Matrix Shape [N, Parameters]: {X_final.shape}")
print(f"Y Tensor Shape [N, Space, Time]: {Y_final.shape}")
print("="*50)

# X columns: [Current (A), Anode diffusivity (m2/s), Cathode diffusivity (m2/s), Electrolyte_D_scale_factor]
dataset_filename = 'dataset_lithium.npz'
np.savez_compressed(    # Dataset structure
    dataset_filename, 
    X=X_final, # Input parameters (Current, Anode/Cathode/Electrolyte Diffusivity)
    Y=Y_final, # Objective variable (Electrolyte Concentration maps across space and time)
    T=T_final  # Extra file for stats (discharge durations before cut-off)
)
print(f"\nUnified Dataset successfully saved as '{dataset_filename}'.")