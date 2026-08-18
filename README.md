# rdDFI
Directional mechanical susceptibility and mutation sensitivity in transmembrane proteins.
# GPCR Prediction – Processing and Plotting Scripts

This directory (`src/perturbation/`) contains two Python scripts for analyzing and visualizing anisotropic DFI‑based metrics for GPCR mutation site prediction.

## Files

- `rdDFI.py`  
  Computes precision, recall, and other statistics for each protein and metric, saving the results as a CSV file. **Does not generate any figures.**

- `plot.py`  
  Generates two publication‑ready figures:
  - **Figure 4** – ratio‑sweep line plots with magnified insets (based on `heat.py`).
  - **Figure 5** – combined Venn diagram, overview bar chart, and per‑protein mutation detail (based on `first.py`).  
    The Venn diagram numbers have **no white background**.

## Requirements

- Python 3.8+
- Packages: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`

Install with:
```bash
pip install numpy pandas scipy matplotlib seaborn

-input files:
The input files have been put in the same direction with rdDFI.py and plot.py under the path:src/pertubation/results
