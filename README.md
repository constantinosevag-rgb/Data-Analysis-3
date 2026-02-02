# Data-Analysis-3

# Airbnb Pricing Prediction (Paris & Bordeaux)

This repository contains the code for **Assignment 2** of Data Analysis 3. The goal was to build a pricing model for a hypothetical Airbnb management company operating in Paris, and then test its validity on a new city (Bordeaux).

## Project Overview
We analyzed Airbnb listings data to predict price per night. The workflow includes:
* **Data Wrangling:** Cleaning raw data from Inside Airbnb (Paris Q1 2024).
* **Modeling:** Compared 5 algorithms: OLS, LASSO, Random Forest, Gradient Boosting, and XGBoost.
* **Validity Checks:** Tested model performance on a later date (June 2024) and an external city (Bordeaux).

## Key Findings
* **XGBoost** was the most robust model for expansion, achieving the lowest error (RMSE ~149) on the external city (Bordeaux).
* **Random Forest** performed best within Paris but overfitted to the specific geography, failing to generalize well to the new city.

## Files
* `house_prices_paris.ipynb`: The main analysis notebook containing all code, discussion, and visualizations.
* `requirements.txt`: List of Python libraries required to run the project.

## How to Run
The data is loaded directly from the repository URLs, so no manual download is needed.

1. Clone the repository:
   ```bash
   git clone https://github.com/constantinosevag-rgb/Data-Analysis-3

   pip install -r requirements.txt
