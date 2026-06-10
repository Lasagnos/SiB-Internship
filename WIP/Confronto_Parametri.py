import pybamm
import pandas as pd

# Caricamento dei due database
print("Caricamento database PyBaMM in corso...")
param_li = pybamm.ParameterValues("Chen2020")       # Litio
param_na = pybamm.ParameterValues("Chayambuka2022") # Sodio

keys_to_compare = [
    "Nominal cell capacity [A.h]",
    "Negative electrode thickness [m]",
    "Separator thickness [m]",
    "Positive electrode thickness [m]",
    "Negative electrode porosity",
    "Positive electrode porosity",
    "Negative particle radius [m]",
    "Positive particle radius [m]"
]

# Estrazione i valori
dati = []
for key in keys_to_compare:
    try:
        # Molti parametri sono salvati come scalari o array, cerchiamo di estrarre il numero
        val_li = param_li[key]
        val_na = param_na[key]
        
        # Formattiamo i numeri in notazione scientifica se sono molto piccoli
        if isinstance(val_li, (int, float)):
            val_li_str = f"{val_li:.4e}" if val_li < 0.01 else f"{val_li:.4f}"
            val_na_str = f"{val_na:.4e}" if val_na < 0.01 else f"{val_na:.4f}"
        else:
            val_li_str = str(val_li)
            val_na_str = str(val_na)
            
        dati.append([key, val_li_str, val_na_str])
    except KeyError:
        dati.append([key, "Non Trovato", "Non Trovato"])


df = pd.DataFrame(dati, columns=["Parametro Fisico", "Litio (Chen2020)", "Sodio (Chayambuka2022)"])

print("\n" + "="*80)
print("   CONFRONTO GEOMETRICO E CAPACITÀ: LITIO vs SODIO")
print("="*80)
print(df.to_string(index=False, justify='left'))
print("="*80 + "\n")



# ================================================================================
#    CONFRONTO GEOMETRICO E CAPACITÀ: LITIO vs SODIO
# ================================================================================
# Parametro Fisico                 Litio (Chen2020) Sodio (Chayambuka2022)
#      Nominal cell capacity [A.h]     5.0000       3.0000e-03            
# Negative electrode thickness [m] 8.5200e-05       6.4000e-05            
#          Separator thickness [m] 1.2000e-05       2.5000e-05            
# Positive electrode thickness [m] 7.5600e-05       6.8000e-05            
#      Negative electrode porosity     0.2500           0.5100            
#      Positive electrode porosity     0.3350           0.2300            
#     Negative particle radius [m] 5.8600e-06       3.4800e-06            
#     Positive particle radius [m] 5.2200e-06       5.9000e-07            
# ================================================================================


# Todo - keep the code above but add the visualization from the notebook regarding the new geometry of the sodium cell's visualization