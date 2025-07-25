# Heat Transfer Correlation for Sweeping Jet Nozzles

This repository contains the Matlab script developed for my Bachelor's thesis in Aerospace Engineering, focused on the development of a semi-empirical correlation for heat transfer in impinging sweeping jets.

## 🎯 Objective
The goal of the project is to correlate the average Nusselt number on the impact surface of a sweeping jet nozzle with key geometric and flow parameters using regression techniques.

## 📊 Methodology
- Input data from experimental tests was organized in matrix form.
- Outlier detection was performed using the IQR method and manual refinement.
- The model is based on a log-log multivariate linear regression of the form:
log(Num) = Am * log(St) + Bm * log(theta) + Cm * log(Re) + Dm * log(H/D)
- The resulting correlation was evaluated through:
- Coefficient extraction (Am, Bm, Cm, Dm)
- Goodness-of-fit (R² and RMSE)
- Residual analysis
- VIF (Variance Inflation Factor) to assess multicollinearity
- Graphical comparisons between predicted and actual values (±15% bounds)

## 📈 Outputs
- Coefficients of the fitted model
- Residual histograms and scatter plots
- Actual vs. predicted Nusselt numbers with error bounds
- VIF values for model diagnostics

## 🛠 Tools Used
- Matlab (R2023 or later)
- Statistics and Plotting toolkits

## 📄 Thesis Details
**Title:** Impinging Sweeping Jet Heat Transfer Correlation  
**Author:** Elpidio Martucci  
**Degree:** BSc in Aerospace Engineering  
**University:** University of Naples “Federico II”  
**Date:** November 2024  

## ⚠️ Disclaimer
This is an academic project. Results are not intended for industrial use without further validation.
