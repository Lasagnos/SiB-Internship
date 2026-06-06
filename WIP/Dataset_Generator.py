import pybamm
import numpy as np
from scipy.stats import qmc

# ====================================================================
# 1. SETUP DEL CAMPIONAMENTO LHS (LATIN HYPERCUBE SAMPLING)
# ====================================================================
N_campioni_richiesti = 500  # Quante simulazioni vogliamo generare
N_parametri = 2             # Parametri di input: [Corrente, Diffusività Solida]

sampler = qmc.LatinHypercube(d=N_parametri)
campioni_raw = sampler.random(n=N_campioni_richiesti)

# Parametro 1: Corrente (scala lineare da 0.05 Amperes a 0.6 Amperes)
I_min, I_max = 0.05, 0.6
correnti_lhs = I_min + campioni_raw[:, 0] * (I_max - I_min)

# Parametro 2: Diffusività Anodo (scala logaritmica da 1e-16 a 1e-14)
Ds_exp_min, Ds_exp_max = -16.0, -14.0   # Campioniamo gli esponenti tra -16 e -14
Ds_exp_lhs = Ds_exp_min + campioni_raw[:, 1] * (Ds_exp_max - Ds_exp_min)
diffusivita_lhs = 10 ** Ds_exp_lhs

# Creiamo la Matrice X (Input per la Rete Neurale). Sarà una matrice di dimensioni (N_campioni_richiesti, N_parametri)
X_inputs = np.column_stack((correnti_lhs, diffusivita_lhs))


# ====================================================================
# 2. SETUP DEL MODELLO FISICO (PyBaMM - SIBs)
# ====================================================================
model = pybamm.sodium_ion.BasicDFN()
param = pybamm.ParameterValues("Chayambuka2022")

# Diciamo a PyBaMM che sia la Corrente che la Diffusività saranno dati in input "al volo"
param["Current function [A]"] = "[input]"
param["Negative electrode diffusivity [m2.s-1]"] = "[input]"

sim = pybamm.Simulation(model, parameter_values=param)
t_eval_max = [0, 7200]  # 2 ore limite massimo (ma PyBaMM si ferma da solo quando la batteria si scarica)


# ====================================================================
# 3. GENERAZIONE DEL DATASET
# ====================================================================
dataset_y = []       # Qui salveremo le heatmap (Tensore Output)
input_validi = []    # Qui salveremo gli X_input delle sole simulazioni andate a buon fine

print(f"Inizio generazione di {N_campioni_richiesti} campioni LHS...")

for i in range(N_campioni_richiesti):
    corrente_corrente = X_inputs[i, 0]  # :)
    diff_corrente = X_inputs[i, 1]
    
    # Eseguiamo in un blocco try-except per ignorare combinazioni fisicamente impossibili che potrebbero far fallire PyBaMM a tempo 0
    try:
        sim.solve(
            t_eval=t_eval_max, 
            inputs={
                "Current function [A]": corrente_corrente,
                "Negative electrode diffusivity [m2.s-1]": diff_corrente
            }
        )
        
        # Normalizzazione temporale (100 step uniformi da 0 a t_max)
        t_max = sim.solution["Time [s]"].entries[-1]
        t_eval_uniform = np.linspace(0, t_max, 100)
        
        c_e_obj = sim.solution["Electrolyte concentration [mol.m-3]"]
        c_e_uniform = c_e_obj(t=t_eval_uniform)
        
        # Salviamo i risultati
        dataset_y.append(c_e_uniform)
        input_validi.append(X_inputs[i])
        
        if (i+1) % 50 == 0:
            print(f"Completati {i+1}/{N_campioni_richiesti} campioni...")
            
    except Exception as e:
        pass    # Se la combinazione è troppo estrema e PyBaMM fallisce a tempo 0, la ignoriamo


# ====================================================================
# 4. SALVATAGGIO DEI TENSORI PER IL MODELLO ML
# ====================================================================
X_final = np.array(input_validi)
Y_final = np.array(dataset_y)

print("\n--- GENERAZIONE COMPLETATA ---")
print(f"Simulazioni riuscite: {X_final.shape[0]} su {N_campioni_richiesti}")
print(f"Forma Tensore X (Input): {X_final.shape}")
print(f"Forma Tensore Y (Output): {Y_final.shape}")

np.save('X_inputs_sodio.npy', X_final)
np.save('Y_outputs_sodio.npy', Y_final)
print("Dataset salvati: 'X_inputs_sodio.npy' e 'Y_outputs_sodio.npy'.")

# Warning ricevuto:
#  C:\python312\Lib\site-packages\pybamm\solvers\base_solver.py:1040: SolverWarning: 
#  While solving Doyle-Fuller-Newman model extrapolation occurred for ["Interpolant 'sigma_e' lower bound"]
#    self.check_extrapolation(solution, model.events)
# Spiegazione:
#  Durante la simulazione, PyBaMM ha dovuto stimare un valore di "sigma_e" (conduttività dell'elettrolita) al di fuori dell'intervallo per cui è definito il modello.
#  Usando il Sodio, in effetti, il catodo si "asciuga" e la concentrazione scende a zero (Min = -0.0),
#  quindi è un warning normale