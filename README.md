# Predicting sulfamethoxazole adsorption onto biochar via machine learning: A data-driven approach for performance evaluation and mechanism interpretation

Antibiotic contamination in water poses growing environmental risks. Biochar adsorption is an effective approach for removing sulfamethoxazole (SMX) from water, yet its performance is governed by coupled multi-factor interactions that are difficult to systematically evaluate through traditional experiments alone. To address this gap, this study developed four machine learning models, namely random forest, support vector regression, extra trees regressor, and multilayer perceptron (MLP), using a compiled literature dataset to predict SMX adsorption onto biochar. The MLP model achieved test R2 > 0.99, outperforming the other models across all feature combinations. SHapley Additive exPlanations (SHAP) analysis identified initial concentration, modification method, specific surface area, pore volume, contact time, pH, adsorbent dosage, and pyrolysis temperature as the key determinants of adsorption capacity. Further dependence analysis indicated the correlation between these key features and SMX adsorption mechanisms onto biochar. Finally, a simplified MLP model retaining only six easily accessible features still achieved a test R2 of approximately 0.99, offering a practical tool for rapid engineering assessment. This study provides a methodological reference for machine learning-driven adsorption research. The findings can help improve the understanding and prediction of SMX removal efficiency, facilitating biochar screening and process optimisation in water treatment.

## Installation

```bash
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for the local CUDA version when GPU
acceleration is required.
