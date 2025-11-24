# **Gal-halo_Pipeline**

Pipeline and accompanying data for reproducing the galaxy–halo relation analysis in **Liang, Jiang et al. (2025)**.



This repository provides a complete workflow for identifying empirical relations between dark matter halo properties and galaxy properties. The pipeline consists of three main steps:

## **1. Train and tune a Random Forest model**

Prepare the input feature matrix **X** (halo properties) and target vector **y** (galaxy properties).  
Train a Random Forest model and tune its hyperparameters.  
These steps are implemented in **`RF_SHAP.py`**.

## **2. Compute feature importance using SHAP**

After selecting the best Random Forest model, apply SHAP to compute feature importance.  
The SHAP analysis is also handled within **`RF_SHAP.py`**.

## **3. Perform symbolic regression with PySR**

Select the top features (e.g., top 10 or any number you choose) based on the SHAP importances.  
Use these features to perform symbolic regression with PySR.  
You can use the parameter settings and loss function from our paper or customize them in **`PySR.py`**.  
This step produces interpretable empirical formulas that relate halo properties to galaxy properties.

## **Included Materials**

- All parameter settings used in **Liang, Jiang et al. (2025)** are provided directly in the scripts.
- Precomputed results—including Random Forest models, SHAP outputs, and PySR symbolic regression results—are available in their respective folders.
