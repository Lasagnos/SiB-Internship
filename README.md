Files regarding the internship of Angelo Greco in CNR's Team DAIMON, y.2026, University of Bologna.

Currently, the main code can be seen in 'Notebook.ipynb'.

Initial versions of the dataset are stored in 'dataset_sodium.npz' (raw dataset) and 'dataset_sodium_normalized.npz' (normalized dataset). Their fields are:
* X: input parameters (Current and Diffusivity of Anode/Cathode/Electrolyte)
* Y: target variables (temporal maps of electrolyte concentration)
* T: additional file for statistics (discharge duration before cut-off)