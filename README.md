Files regarding the internship of Angelo Greco in CNR's Team DAIMON, y.2026, University of Bologna.

Currently, the main code can be seen in 'Notebook.ipynb'.

Initial versions of the dataset are stored in 'dataset_sodium.npz' (raw dataset) and 'dataset_sodium_normalized.npz' (normalized dataset). The raw fields are:
* X: input parameters (Current and Diffusivity of Anode/Cathode/Electrolyte)
* Y: target variables (temporal maps of electrolyte concentration)
* T: additional file for statistics (discharge duration before cut-off)

The normalized fields are: X_norm, Y_norm and T_raw, aswell as maintaining a scaling_factors field, which are: [I_min, I_max, Da_min, Da_max, Dc_min, Dc_max, De_min, De_max, Y_min, Y_max].

