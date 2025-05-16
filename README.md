# 📈 Trading Algorithm - Crypto Price Movement Predictor

This repository contains a machine learning-based trading algorithm that predicts cryptocurrency (Bitcoin) price movements using historical hourly data.

## 🔍 Overview

This project aims to:
- Analyze historical Bitcoin data.
- Build a predictive model for price movement.
- Execute trades based on the model's output.

It includes a Jupyter notebook for analysis, a trained model, and a script for running predictions.

## 🚀 Features

- 🔮 Predictive model using ML
- 🕒 Uses hourly BTC data
- 💹 Simple trade simulation logic
- 📊 Jupyter notebook for EDA and training
- 🧠 Saved model for reuse

## 🗂️ Project Structure

```
├── data/              # CSV files with input and prediction data
├── model/             # Saved ML model
├── notebooks/         # Jupyter exploration notebook
├── src/               # Main logic for running the algorithm
```

## 📦 Setup

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Script

```bash
python src/algo.py
```

### Explore the Notebook

Open the notebook with:

```bash
jupyter notebook notebooks/algo.ipynb
```

## 📁 Data Sources

- `btc_1h.csv.csv`: Historical Bitcoin data (hourly)
- `prediction.csv`: Output predictions from the model

## 🤖 Model

The model (`trading_algo.pkl`) is trained to classify future price direction based on recent data trends.

## 📌 Notes

- This is a basic prototype and **not** intended for real financial trading.
- Make sure to backtest thoroughly before considering real deployment.

## 📜 License

[MIT License](LICENSE)

---

## 🤝 Contributing

Feel free to fork, improve, and suggest changes. PRs are welcome!
