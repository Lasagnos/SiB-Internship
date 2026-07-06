import pybamm
import numpy as np
from scipy.stats import qmc
# import warnings

print("="*60)
print("   INITIALIZING VIRTUAL CELL DESIGN & LHS SAMPLING")
print("="*60)

param_li = pybamm.ParameterValues("Chen2020")       # "Target geometry" (Commercial 21700 Cell, 5Ah)
#param_na = pybamm.ParameterValues("Chayambuka2022") # Sodium chemistry (Hard Carbon / NVPF)

# Explicitly set the voltage cut-off for the Lithium cell to 2.5 Volts
param_li["Lower voltage cut-off [V]"] = 2.5   # Default, but we set it explicitly for clarity

# # List of macro-geometric parameters to copy from Lithium to Sodium
# # to ensure a rigorous iso-geometric comparison (Sulzer et al., 2021)
# macro_parameters = [
#     "Nominal cell capacity [A.h]",
#     "Electrode height [m]",
#     "Electrode width [m]",
#     "Negative electrode thickness [m]",
#     "Separator thickness [m]",
#     "Positive electrode thickness [m]",
#     "Number of electrodes connected in parallel to make a cell",
#     "Cell volume [m3]",
#     "Cell cooling surface area [m2]"
# ]

# # Overwrite the coin cell geometry with the commercial cell geometry
# for p in macro_parameters:
#     param_na[p] = param_li[p]

# Tell PyBaMM which parameters will be input (Current and Diffusivity)
param_li["Current function [A]"] = "[input]"
param_li["Negative electrode diffusivity [m2.s-1]"] = "[input]" # Anode (Graphite)
param_li["Positive electrode diffusivity [m2.s-1]"] = "[input]" # Cathode (NMC)
param_li["Electrolyte diffusivity [m2.s-1]"] = "[input]"        # Liquid Electrolyte

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
sim.solver.set_logging_level("ERROR")   # ignore warnings

# =================================================================

# HYPERPARAMETERS for the LHS sampling of the input space
TARGET_SAMPLES = 500
POOL_SIZE = 3000    # Number of random samples to generate each time
num_parameters = 4  # Current, Anode/Cathode/Electrolyte Diffusivity
RANDOM_SEED = 42

sampler = qmc.LatinHypercube(d=num_parameters, seed=RANDOM_SEED)
raw_samples = sampler.random(n=POOL_SIZE)

# Parameter 1: Current [A] (0.5 to 15.0 - Lithium handles high C-rates better)
I_min, I_max = 0.5, 15.0 
I_lhs = I_min + raw_samples[:, 0] * (I_max - I_min)

# Parameter 2: Anode Diffusivity [m2/s] (Graphite is faster: 1e-14 to 1e-12)
Ds_n_lhs = 10 ** (-14.0 + raw_samples[:, 1] * 2.0)

# Parameter 3: Cathode Diffusivity [m2/s] (NMC is faster: 1e-15 to 1e-13)
Ds_p_lhs = 10 ** (-15.0 + raw_samples[:, 2] * 2.0)

# Parameter 4: Electrolyte Diffusivity [m2/s] (1e-10 to 1e-9)
De_lhs = 10 ** (-10.0 + raw_samples[:, 3] * 1.0)

X_pool = np.column_stack((I_lhs, Ds_n_lhs, Ds_p_lhs, De_lhs))

# =================================================================

dataset_y = []       
valid_inputs = []
durations = [] # For analyzing it later

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
                "Electrolyte diffusivity [m2.s-1]": X_pool[i, 3]
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
            
    except Exception as e:
        pass    # Ignore simulations that cause errors

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

dataset_filename = 'dataset_lithium.npz'
np.savez_compressed(    # Dataset structure
    dataset_filename, 
    X=X_final, # Input parameters (Current, Anode/Cathode/Electrolyte Diffusivity)
    Y=Y_final, # Objective variable (Electrolyte Concentration maps across space and time)
    T=T_final  # Extra file for stats (discharge durations before cut-off)
)
print(f"\nUnified Dataset successfully saved as '{dataset_filename}'.")


#  TARGET REACHED: 500 valid simulations extracted!

# ==================================================
#    DATASET SUCCESSFULLY GENERATED
# ==================================================
# Target valid simulations requested: 500
# Total simulations attempted by LHS: 521
# Valid stable simulations stored: 500
# Discarded due to numerical instability: 21
# X Matrix Shape [N, Parameters]: (500, 4)
# Y Tensor Shape [N, Space, Time]: (500, 64, 100)
# ==================================================