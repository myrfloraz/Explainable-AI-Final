# Explainable AI for Financial Time Series Forecasting

This project demonstrates the use of Integrated Gradients (IG) and Layer-wise Relevance Propagation (LRP) to explain predictions made by a Transformer-based time series model trained on financial data (AAPL and TSLA).

The goal is to compare two gradient-based attribution methods in terms of how they explain feature importance and time step relevance in stock price forecasting.

---

## Installation

To run the notebook in a clean Python environment or in Google Colab, install the following dependencies:

```bash
pip install yfinance torch torchvision torchaudio captum scikit-learn matplotlib seaborn
```

If using Google Colab, you can install everything in one cell:

```python
!pip install yfinance torch torchvision torchaudio captum scikit-learn matplotlib seaborn
```

---

## Usage Guide

### Step 1: Open the notebook

Open `Final_Project_Explainable_AI.ipynb` in Jupyter Notebook or Google Colab.

### Step 2: Run cells in order

The notebook is organized into the following sections:

1. **Data Download & Preprocessing**  
   Downloads historical stock data for AAPL and TSLA using `yfinance`, normalizes it, and prepares sequences for forecasting.

2. **Model Training**  
   Trains two separate Transformer models to forecast the next-day Open price based on the past 30 days of data.

3. **Explainability (IG & LRP)**  
   Applies Integrated Gradients and LRP (via DeepLIFT) to explain a sample prediction for each ticker.

4. **Visualization & Analysis**  
   Generates attribution heatmaps, feature/time step importance plots, and compares IG vs. LRP both quantitatively and visually.

---

## Outputs

The notebook produces:
- Attribution heatmaps for feature × time step relevance
- Bar plots comparing IG vs LRP feature importance
- Line plots comparing time step relevance
- Printouts of top features and time steps

---

## Notes

- LRP is approximated using **DeepLIFT** via the Captum library.
- Data is normalized using **MinMaxScaler**, and models are trained per ticker (AAPL and TSLA).
- All explanations are based on **one representative prediction** for each model.

---

## File Structure

```
Final_Project_Explainable_AI.ipynb     # Main notebook
README.md                              # This readme
```

---

## Author & Context

This notebook was developed as part of a project on **Explainable AI** in the context of financial time series.  
It demonstrates how modern attribution techniques can provide interpretable insights into deep learning models trained on stock data.
