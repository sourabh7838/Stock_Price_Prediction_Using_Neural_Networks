# Stock Price Learning Agent with Real-Time Data Collection

A hands-on project that brings together financial data scraping and reinforcement learning:

- **stock_data_download.py** – fetches stock data using Yahoo Finance.
- **sourabh_chauhan_rl.py** – reinforcement learning agent to simulate and learn trading strategies.

Think of it as a lightweight pipeline that goes from raw market data to RL-driven decisions – all in pure Python.

---

## Quick Start

### 1. Clone this Repo / Copy the Scripts

```bash
git clone https://github.com/your-username/rl-stock-agent.git
cd rl-stock-agent
```

2. Create a Clean Environment
```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

3. Install Dependencies
```
pip install -r requirements.txt
```

5. Run It
```

# Step 1: Download historical stock data
python stock_data_download.py
```
# Step 2: Train and evaluate RL trading agent


python sourabh_chauhan_rl.py
Requirements (PyPI names)
```
numpy
pandas
matplotlib
scikit-learn
tensorflow
yfinance
gym
```

To install manually:
```
pip install numpy pandas matplotlib scikit-learn tensorflow yfinance gym
```


File Layout


```
.
├── sourabh_chauhan_rl.py        # RL agent implementation
├── stock_data_download.py       # Stock data retrieval script
├── requirements.txt             # Python dependencies
└── README.md
```                   
Agent Highlights

Reinforcement Learning Strategy (sourabh_chauhan_rl.py)
Architecture: Likely uses a Dense DQN / custom logic to learn buy-hold-sell behavior.
Environment: Simulated trading environment (or Gym-style wrapper).
Reward Signal: Profit/loss per action or cumulative return.
Why it’s here: Models real-world stock dynamics and adapts to price trends.
Data Retrieval Tool (stock_data_download.py)

Source: Yahoo Finance via yfinance.
Customizable: Change tickers, date ranges, and intervals.
Output: Saves data as CSV or loads into pandas DataFrames.
Why it’s here: Keeps your model fed with the latest market information.
Sample Output

Training Loss dropping over episodes, Profit Curve rising
Final evaluation on test data shows learned behavior over random guessing.
Your results may vary depending on model architecture, learning rate, and exploration-exploitation balance.

Extend This Project

* Try different stocks – S&P 500 tickers, crypto, or ETFs.
* Use technical indicators – add SMA, RSI, or Bollinger Bands as state inputs.
* Switch agents – upgrade to DDPG, PPO, or TD3 using Stable-Baselines3.
* Deploy it – connect to a mock trading API or build a dashboard in Streamlit.

References

Yahoo Finance API – yfinance
OpenAI Gym Docs
Deep RL Trading Papers
Author – Sourabh Chauhan (2025)
