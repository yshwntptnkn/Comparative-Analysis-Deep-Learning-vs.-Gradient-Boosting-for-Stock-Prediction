# Comparative Analysis: Deep Learning vs. Gradient Boosting for Stock Prediction

### 📉 Predicting ITC.NS Stock Price (2021-2026)

This project performs a rigorous engineering comparison between two dominant architectures for time-series forecasting: **Attention-Based Long Short-Term Memory (LSTM)** networks and **XGBoost**.

Unlike standard "stock prediction" tutorials, this project investigates the impact of **Stationarity** on model performance. It demonstrates why simpler Tree-based models often outperform Deep Learning on daily tabular financial data.

---

## 🧰 Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Core** | ![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white) |
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-Architecture-D00000?style=flat&logo=keras&logoColor=white) |
| **Machine Learning** | ![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-11557C?style=flat) ![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Metrics-F7931E?style=flat&logo=scikit-learn&logoColor=white) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-DataFrames-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-Arrays-013243?style=flat&logo=numpy&logoColor=white) |
| **Data Source** | ![yfinance](https://img.shields.io/badge/yfinance-API-00C805?style=flat) |

---

## 📊 Benchmark Results (The "Face-Off")

Models were evaluated on a 1-year unseen test set (Jan 2025 - Jan 2026).

| Model | Root Mean Square Error (₹) | Mean Absolute Error (₹) | Mean Absolute Percentage Error (%) | R-Squared |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **4.57** | **2.86** | **0.71%** | **0.9389** |
| Attention LSTM | 24.22 | 18.27 | 4.64% | -0.8773 |

![Model Comparison Chart](results/benchmark_chart.png)

### 🧠 Technical Conclusion: Why did Deep Learning Fail?
Despite the architectural complexity of the Attention-Based LSTM, the XGBoost model outperformed it significantly. This highlights a critical lesson in Financial ML:

1.  **The Stationarity Problem:** The LSTM was trained on raw prices (Non-Stationary), causing it to fail when the stock hit new all-time highs (Distribution Shift). XGBoost was trained on **Returns** (Stationary), allowing it to generalize perfectly to new price ranges.
2.  **Tabular Dominance:** For daily time-series data with limited features, Gradient Boosting (XGBoost) creates decision boundaries that are more robust to noise than the continuous functions approximated by Neural Networks.

---

## 🧠 Model Architectures

### 1. Attention-Based LSTM (Deep Learning)
**What it is:**
A **Long Short-Term Memory (LSTM)** network is a specialized Recurrent Neural Network (RNN) designed to process sequential data. While standard LSTMs can learn long-term dependencies, they often struggle with "memory bottlenecks" when compressing long histories into a single vector.

**The "Attention" Mechanism:**
This project implements a custom **Self-Attention** layer. This allows the model to dynamically "focus" on specific past time steps when making a prediction. Instead of treating the entire 60-day window equally, the model assigns higher weights to critical events (e.g., a volatility spike 10 days ago) while ignoring noise.

```
[ Input Sequence: (60, 4) ]  <-- Price, RSI, EMA, Volume
       │
       ▼
[   LSTM Layer (64 units)   ]  <-- Extracts temporal features
       │           │
       │           ▼
       │    [ Self-Attention ] <-- "Which days matter most?"
       │           │
       ▼           ▼
[  Concatenation Layer   ]  <-- Combines Memory + Focus
       │
       ▼
[   Dense Layer (64)     ]  <-- Non-linear processing
       │
       ▼
[    Output Neuron       ]  <-- Predicted Price (Scalar)
 ```

* **Analogy:** It’s like reading a sentence. To predict the last word, you don't just look at the previous word; you "attend" to the subject at the start of the sentence to understand the context.

### 2. XGBoost (Gradient Boosting)
**What it is:**
**XGBoost (Extreme Gradient Boosting)** is an optimized implementation of the Gradient Boosting Decision Tree (GBDT) algorithm. Unlike a Neural Network, which acts as a single complex function, XGBoost functions like a "committee of experts."

**How it works:**
The model builds hundreds of simple decision trees sequentially:
1.  **Tree 1:** Makes a rough prediction.
2.  **Tree 2:** Trains specifically to correct the errors made by Tree 1.
3.  **Tree 3:** Trains to correct the errors from Tree 2.
4.  **Final Output:** The weighted sum of all trees.

```
[ Input Features ] (RSI, Lags, Volatility)
       │
       ▼
[ Tree 1 ] --> Prediction: 400.0  |  Error: +10.0
       │
       ▼
[ Tree 2 ] --> Predicts Error: +8.0
       │
       ▼
[ Tree 3 ] --> Predicts Error: +1.5
       │
       ▼
[ Final Sum ] = 400.0 + 8.0 + 1.5 = 409.5
```

**Why it outperformed Deep Learning:**
XGBoost is mathematically optimized for **tabular data** (structured rows and columns). It excels at finding discrete, non-linear split points (e.g., *"If RSI > 70 AND Volume > 1M, then Price drops"*) without requiring the massive datasets or continuous function approximations that Deep Learning relies on.

---

## 📂 Project Structure
```
Comparative-Analysis-Deep-Learning-vs.-Gradient-Boosting-for-Stock-Prediction/
├── src/                    
│   ├── data_loader.py      
│   ├── preprocessing.py    
│   ├── model_builder.py    
│   └── evaluation.py       
│
├── main.py                 # Runner for LSTM training
├── train_xgboost.py        # Runner for XGBoost training
├── benchmark.py            # Runner for Model Comparison
│
├── results/                # Generated Plots & CSVs
└── requirements.txt
```

---

## 🚀 Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Comparative-Analysis-Deep-Learning-vs.-Gradient-Boosting-for-Stock-Prediction.git
   cd Comparative-Analysis-Deep-Learning-vs.-Gradient-Boosting-for-Stock-Prediction
   ```
2. Install dependancies
   ```bash
   pip install -r requirements.txt
   ```

This project uses a modular MLOps structure. You can reproduce the results by running the following commands:

1. **Train the Deep Learning Model (LSTM)**
  Trains an Attention-Based LSTM on 60-day sequences of Price, RSI, EMA, and Volume.
  ```bash
  python main.py --ticker ITC.NS --epochs 50
  ```
2. **Train the Machine Learning Model (XGBoost)**
   Trains an XGBRegressor on technical indicators and lagged returns.
   ```bash
   python train_xgboost.py --ticker ITC.NS
   ```

3. **Run the Benchmark**
  Generates the comparison chart and scorecard found in the ```results/``` folder.
  ```bash
  python benchmark.py
  ```

---

## 🔮 Future Work
This study established a baseline comparison between Deep Learning and Gradient Boosting for daily stock data. Future iterations of this research could explore:

* **Transformer Architectures:** Replacing the LSTM with a **Temporal Fusion Transformer (TFT)** to better handle long-term dependencies and interpretability.
* **Sentiment Analysis:** Integrating NLP features from financial news (e.g., using FinBERT) to capture market sentiment, which neither model currently accounts for.
* **High-Frequency Data:** Testing if LSTM performance improves with intraday (1-minute) data, where patterns may be more sequential and less noisy than daily closes.

---

## 📜 License
This project is open-source and available under the MIT License.

---

## ⚠️ Disclaimer
This project is for **educational and research purposes only**. The models developed here are engineering prototypes and should not be used for real-world financial trading or investment decisions.

* Stock market data is highly volatile and unpredictable.
* The "93% R-Squared" metric achieved by XGBoost is based on historical backtesting and does not guarantee future performance.
* I am not a financial advisor, and this repository does not constitute financial advice.

---

## 🙏 Acknowledgements
* **Data Source:** Historical stock data fetched using [yfinance](https://pypi.org/project/yfinance/), courtesy of Yahoo Finance.
* **Libraries:** Built with [TensorFlow](https://www.tensorflow.org/), [XGBoost](https://xgboost.readthedocs.io/), and [Scikit-Learn](https://scikit-learn.org/).
* **Inspiration:** This project was inspired by the ongoing debate between Deep Learning and Classical ML in the quantitative finance community.

---

## 👤 Authors

Yashwant Patnaikuni

📧 yashwantpatnaikuni@gmail.com <br>
ℹ️ www.linkedin.com/in/yashwant-patnaikuni

Nirup Koyilada

📧nirupkoyilada@gmail.com <br>
ℹ️www.linkedin.com/in/nirup-koyilada
---
