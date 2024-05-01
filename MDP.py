import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from pypfopt import expected_returns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sqlite3
conn = sqlite3.connect('dbb.db')
c = conn.cursor()


def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS usertable(username TEXT, password TEXT)')

def add_userdata(username, password):
    c.execute('INSERT INTO usertable(username,password) VALUES (?,?)',(username, password))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM usertable WHERE username=? AND password=?',(username, password))
    data = c.fetchall()
    return data


def calculate_annual_return(series):
    return series.pct_change().mean() * 252

def calculate_annual_volatility(series):
    return series.pct_change().std() * np.sqrt(252)

def calculate_sharpe_ratio(series):
    volatility = calculate_annual_volatility(series)
    if volatility == 0:
        return np.inf  
    return calculate_annual_return(series) / volatility


def calculate_maximum_drawdown(series):
    return series.pct_change().min()

@st.cache_data
def calculate_optimal_weights(train_df, risk_aversion):
  def objective(weights):
    expected_ret = expected_returns.mean_historical_return(train_df)
    return -np.dot(weights.T, expected_ret) + risk_aversion * np.dot(weights.T, np.dot(train_df.pct_change().cov() * 252, weights))

  constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
          {'type': 'ineq', 'fun': lambda x: 0.35 - np.max(x)})

  initial_guess = np.repeat(1 / train_df.shape[1], train_df.shape[1])
  bounds = [(0, 1) for _ in range(train_df.shape[1])]
  try:
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
    return result.x
  except Exception as e:
    st.error(f"Error calculating weights: {e}")
    return None  

    
def calculate_portfolio_return(df, weights):
    return np.sum(df.pct_change().mean() * weights) * 252

def calculate_portfolio_volatility(df, weights):
  return np.sqrt(np.dot(weights.T, np.dot(df.pct_change().cov() * 252, weights)))



def load_data():
  df = pd.read_csv('COMDX.csv')
  return df




def main():
  st.set_page_config(page_title="The MDP System", page_icon="HIC.png")
  df = load_data()
  train_size = int(0.6 * len(df))

  if True:
    house_icon = "house.png"
    graph_up_icon = "process.png"
    receipt_icon = "output.png"
    about = 'about.png'
    a,b,c, d = st.columns(4)
    a.image(house_icon, width = 100)
    b.image(graph_up_icon, width = 100)
    c.image(receipt_icon, width = 100)
    d.image(about, width = 100)

    selected_page = option_menu(
        menu_title=None,
        options=["Home", "MDP Process", "Evaluation Results", 'About?'],
        orientation="horizontal")

    if selected_page == 'Home':
      st.markdown("<h1 style='text-align: center; color: black;'>The MDP COMDX System</h1>", unsafe_allow_html=True)   
      st.image('HIC.png')

    elif selected_page == 'MDP Process':
        
        tickers = df.columns[1:-1].tolist()

        selected_tickers = st.multiselect("Commodity Selection for MDP Optimization:", options=tickers, format_func=lambda x: ' '.join(x),default = tickers)

        risk_aversion = st.number_input('Investor Risk Aversion Level:', min_value=0.0, max_value=1.0, step=0.01, value= 0.5)

        train_df = df[:train_size]

        weights = calculate_optimal_weights(df[selected_tickers], risk_aversion)
        with open('weights.txt', 'w') as f:
            for ticker, weight in zip(selected_tickers, weights):
                f.write(f'{ticker}: {weight}\n')

        st.write("Historical performance:")
        metrics_df = pd.DataFrame()
        for ticker in tickers:
            if ticker in selected_tickers:
                metrics = {
                    'Commodity': ticker,
                    'Annual Return': calculate_annual_return(df[ticker]),
                    'Annual Volatility': calculate_annual_volatility(df[ticker]),
                    'Sharpe': calculate_sharpe_ratio(df[ticker]),
                    'Maximum Drawdown': calculate_maximum_drawdown(df[ticker]),
                }
                new_df = pd.DataFrame(metrics, index=[0])  
                metrics_df = pd.concat([metrics_df, new_df], ignore_index=True)  

        st.write(metrics_df)
        
        st.write("***Optimal Funds Allocation Weights:***")
        weights_df = pd.DataFrame(weights, index=selected_tickers, columns=['Weight'])
        st.bar_chart(weights_df)

       
        st.write("***Portfolio Performance on Training Data:***")
        metrics = {
            'Annual Return': calculate_portfolio_return(train_df[selected_tickers], weights),
            'Annual Volatility': calculate_portfolio_volatility(train_df[selected_tickers], weights)
        }

        if not train_df[selected_tickers].empty:  
            portfolio_return = calculate_portfolio_return(train_df[selected_tickers], weights)
            portfolio_volatility = calculate_portfolio_volatility(train_df[selected_tickers], weights)

            metrics['Sharpe Ratio'] = portfolio_return/portfolio_volatility

        else:
            st.warning("Insufficient data for Sharpe Ratio calculation on training data.")

        col1, col2, col3 = st.columns(3)
        col1.metric(label="Annual Return", value=np.round(metrics['Annual Return'],3))
        col2.metric(label="Annual Volatility", value=np.round(metrics['Annual Volatility'],3))
        col3.metric(label="Sharpe Ratio", value=np.round(metrics['Sharpe Ratio'],3))

        
    elif selected_page == 'Evaluation Results':
        test_df = df[train_size:]
        
        with open('weights.txt', 'r') as f:
            weights_data = f.readlines()

        weights = {}
        for line in weights_data:
            ticker, weight_str = line.strip().split(':')
            weights[ticker] = float(weight_str)
        
        portfolio_returns = (test_df[list(weights.keys())] * list(weights.values())).sum(axis=1)
        portfolio_volatility = portfolio_returns.pct_change().std() * np.sqrt(252)
        portfolio_return = portfolio_returns.pct_change().mean() * 252
        portfolio_sharpe = portfolio_return/portfolio_volatility
        test_df['CS AGRIC INDEX'] = pd.to_numeric(test_df['CS AGRIC INDEX'], errors='coerce')

        top_ten_returns = test_df['CS AGRIC INDEX'].pct_change().mean() * 252  
        top_ten_volatility = test_df['CS AGRIC INDEX'].pct_change().std() * np.sqrt(252)
        top_sharpe = top_ten_returns/top_ten_volatility

        delta_return = portfolio_return - top_ten_returns
        delta_volatility = portfolio_volatility - top_ten_volatility
        delta_sharpe = portfolio_sharpe - top_sharpe

        performance_df = pd.DataFrame({
            'Metric': ['Return', 'Volatility', 'Sharpe Ratio'],
            'MDP Optimized Portfolio': [portfolio_return, portfolio_volatility, portfolio_sharpe],
            'CS AGRIC INDEX': [top_ten_returns, top_ten_volatility, top_sharpe]
        })

        st.write("***Portfolio Performance Evaluation:***")
        st.table(performance_df)
        st.write('Performance Differences')
        c1,c2,c3 = st.columns(3)
        c1.metric(label='Return Difference', value = np.round(delta_return,2))
        c2.metric(label = 'Volatility Difference', value = np.round(delta_volatility,2))
        c3.metric(label = 'Sharpe Difference', value = np.round(delta_sharpe,2))
        
        index_returns = (1 + test_df['CS AGRIC INDEX'].pct_change().dropna()).cumprod()  

        st.write("***Cumulative Returns (Test Period):***")
        fig, ax = plt.subplots()

        color_top_ten = 'tab:blue'
        color_optimized = 'tab:red'

        
        ax.plot(index_returns.index, index_returns, color=color_top_ten, label='CS AGRIC INDEX')

        portfolio_cumulative_returns = (1 + portfolio_returns.pct_change().dropna()).cumprod()
        ax.plot(index_returns.index, portfolio_cumulative_returns, color=color_optimized, label='MDP Optimized Portfolio')

        ax.set_xlabel('Testing Days')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Cumulative Returns Comparison')

        ax.legend(loc=0)

        st.pyplot(fig)
        
    elif selected_page == 'About?':
         st.info("""
           **MDP COMDX: A Markov Decision Process Based Approach to Commodities Portfolio Optimization**

           MDP COMDX is a software application that helps you optimize your investment portfolio using the Markov Decision Process Model. It empowers busy investors to achieve potentially better returns while minimizing risk.

           **Here's how it works:**

           1. **Data Gathering:** The app starts by collecting historical data on commodities and other assets in your chosen market (e.g., Zimbabwe Stock Exchange).

           2. **Markov Decision Process:** A MDP model is trained using this data. The model aims to choose the best asset allocation that maximizes projected returns while minimizing risk.

           3. **Performance Comparison:** MDP COMDX allows you to compare the performance of your existing portfolio to a benchmark index (like the Agricultural Index). This helps you understand how your portfolio stacks up against the benchmark.

           **Who should use MDP COMDX?**

           This tool is ideal for investors who:

           * Want to improve their commodities portfolio performance.
           * Don't have time for active portfolio management.
           * Are interested in exploring data-driven investment strategies.

           **Get Started Today!**

           Explore the app's functionalities to optimize your portfolio and potentially achieve your investment goals.
           """)
        
   
if __name__ == '__main__':
  main()

