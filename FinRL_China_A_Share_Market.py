#!/usr/bin/env python
# coding: utf-8

# ## Quantitative trading in China A stock market with FinRL

# <a href="https://colab.research.google.com/github/AI4Finance-Foundation/FinRL/blob/master/tutorials/3-Practical/FinRL_China_A_Share_Market.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Install FinRL

# In[ ]:

#from IPython import get_ipython

# get_ipython().system('conda activate finacial')
# get_ipython().system('pip install wrds')
# get_ipython().system('pip install swig')
# get_ipython().system('pip install -q condacolab')
# import condacolab
# condacolab.install()
# get_ipython().system('apt-get update -y -qq && apt-get install -y -qq cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig')
# get_ipython().system('pip install git+https://github.com/AI4Finance-Foundation/FinRL.git')


# Install other libraries

# In[2]:


# get_ipython().system('pip install stockstats')
# get_ipython().system('pip install tushare')
# #install talib
# get_ipython().system('wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz')
# get_ipython().system('tar xvzf ta-lib-0.4.0-src.tar.gz')
import os
# os.chdir('ta-lib') 
# get_ipython().system('./configure --prefix=/usr')
# get_ipython().system('make')
# get_ipython().system('make install')
# os.chdir('../')
# get_ipython().system('pip install TA-Lib')


# In[3]:


# get_ipython().run_line_magic('cd', '/')
# get_ipython().system('git clone https://github.com/AI4Finance-Foundation/FinRL-Meta')
# get_ipython().run_line_magic('cd', '/FinRL-Meta/')


# ### Import modules

# In[36]:


import warnings
import joblib
warnings.filterwarnings("ignore")

import pandas as pd
#from IPython import display
#display.set_matplotlib_formats("svg")

from meta import config
from meta.data_processors.tushare import Tushare, ReturnPlotter
from meta.env_stock_trading.env_stocktrading_China_A_shares import StockTradingEnv
#from meta.env_stock_trading.env_stocktrading_cashpenalty import get_transactions
from agents.stablebaselines3_models import DRLAgent
from plot import trx_plot

pd.options.display.max_columns = None
    
print("ALL Modules have been imported!")

 

# ### Create folders

# In[37]:


import os
if not os.path.exists("./datasets" ):
    os.makedirs("./datasets" )
if not os.path.exists("./trained_models"):
    os.makedirs("./trained_models" )
if not os.path.exists("./tensorboard_log"):
    os.makedirs("./tensorboard_log" )
if not os.path.exists("./results" ):
    os.makedirs("./results" )


# ### Download data, cleaning and feature engineering

# In[38]:


ticket_list=['600000.SH', '600009.SH', '600016.SH', '600028.SH', '600030.SH',
       '600031.SH', '600036.SH', '600050.SH', '600104.SH', '600196.SH',
       '600276.SH', '600309.SH', '600519.SH', '600547.SH', '600570.SH']

train_start_date='2015-01-01'
train_stop_date='2019-08-01'
val_start_date='2019-08-01'
val_stop_date='2021-01-03'

token='27080ec403c0218f96f388bca1b1d85329d563c91a43672239619ef5'


# In[39]:


# download and clean
ts_processor = Tushare(data_source="tushare", 
                                   start_date=train_start_date,
                                   end_date=val_stop_date,
                                   time_interval="1d",
                                   token=token)
# ts_processor.download_data(ticker_list=ticket_list)


ts_processor.load_data('./data/dataset.csv')

# In[40]:


ts_processor.clean_data()
ts_processor.fillna()
ts_processor.dataframe


# In[41]:


# add_technical_indicator
ts_processor.add_technical_indicator(config.INDICATORS)
ts_processor.fillna()
ts_processor.dataframe

# ### Split traning dataset

# In[42]:


train =ts_processor.data_split(ts_processor.dataframe, train_start_date, train_stop_date)       
len(train.tic.unique())


# In[43]:


train.tic.unique()


# In[44]:

train.head()


# In[45]:


train.shape


# In[46]:


stock_dimension = len(train.tic.unique())
state_space = stock_dimension*(len(config.INDICATORS)+2)+1
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


# ### Train

# In[47]:

env_kwargs = {
    "stock_dim": stock_dimension,
    "hmax": 100000, 
    "initial_amount": 1000000, 
    "buy_cost_pct":6.87e-5,
    "sell_cost_pct":1.0687e-3,
    "reward_scaling": 1e-4,
    "state_space": state_space, 
    "action_space": stock_dimension,
    "tech_indicator_list": config.INDICATORS, 
    "print_verbosity": 1,
    "initial_buy":True,
    "hundred_each_trade":True
}

e_train_gym = StockTradingEnv(df = train, **env_kwargs)


# ## DDPG

# In[48]:


env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))


# In[50]:


# agent = DRLAgent(env = env_train)
# DDPG_PARAMS = {
#                 "batch_size": 256, 
#                "buffer_size": 50000, 
#                "learning_rate": 0.001,
#                "action_noise":"normal",
#                 }
# POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))
# model_ddpg = agent.get_model("ddpg", model_kwargs = DDPG_PARAMS, policy_kwargs=POLICY_KWARGS)


# In[51]:


# trained_ddpg = agent.train_model(model=model_ddpg, 
#                               tb_log_name='ddpg',
#                               total_timesteps=50000)

# trained_ddpg.save('./ddpg')
# joblib.dump(trained_ddpg, "ddpg.pkl")
# ## A2C

# In[52]:

A2C_model_kwargs = {
                    'n_steps': 5,
                    'ent_coef': 0.005,
                    'learning_rate': 0.0007
                    }

POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))

agent = DRLAgent(env = env_train)
model_a2c = agent.get_model("a2c", model_kwargs=A2C_model_kwargs, policy_kwargs=POLICY_KWARGS)



trained_a2c = agent.train_model(model=model_a2c, 
                             tb_log_name='a2c',
                             total_timesteps=200000)

# agent = DRLAgent(env = env_train)
# model_ppo = agent.get_model("ppo")


# # In[53]:


# trained_ppo = agent.train_model(model=model_ppo, 
#                              tb_log_name='ppo',
#                              total_timesteps=2000000)



# trained_a2c.save('./a2c') 
# joblib.dump(trained_a2c, "a2c.pkl")
# ### Trade

# In[54]:


trade = ts_processor.data_split(ts_processor.dataframe, val_start_date, val_stop_date)
env_kwargs = {
    "stock_dim": stock_dimension,
    "hmax": 100000, 
    "initial_amount": 1000000, 
    "buy_cost_pct":6.87e-5,
    "sell_cost_pct":1.0687e-3,
    "reward_scaling": 1e-4,
    "state_space": state_space, 
    "action_space": stock_dimension,
    "tech_indicator_list": config.INDICATORS, 
    "print_verbosity": 1,
    "initial_buy":False,
    "hundred_each_trade":True
}
e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)


# In[55]:


# df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_ddpg,
#                        environment = e_trade_gym)


# # In[56]:


# #predict csv
# df_actions.to_csv("action_ddpg.csv",index=False)


#a2c
df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_a2c,
                       environment = e_trade_gym)
# In[56]:
#predict csv
df_actions.to_csv("action_a2c.csv",index=True)
#df_actions["transactions"] = get_transactions(df_actions)

trade.to_csv("trade.csv", index=True)

# ### Backtest

# In[57]:


# %matplotlib inline
plotter = ReturnPlotter(df_account_value, trade, val_start_date, val_stop_date)
# plotter.plot_all()


# In[58]:

trx_plot(trade, df_actions, ticket_list)


#get_ipython().run_line_magic('matplotlib', 'inline')
plotter.plot()

# In[59]:

# %matplotlib inline
# # ticket: SSE 50：000016
# plotter.plot("000016")

# #### Use pyfolio

# In[60]:

# CSI 300
baseline_df = plotter.get_baseline("399300")


# In[61]:


# import pyfolio
# from pyfolio import timeseries
# daily_return = plotter.get_return(df_account_value)
# daily_return_base = plotter.get_return(baseline_df, value_col_name="close")

# perf_func = timeseries.perf_stats
# perf_stats_all = perf_func(returns=daily_return, 
#                               factor_returns=daily_return_base, 
#                                 positions=None, transactions=None, turnover_denom="AGB")
# print("==============DRL Strategy Stats===========")
# perf_stats_all


# # In[62]:


# import pyfolio
# from pyfolio import timeseries
# daily_return = plotter.get_return(df_account_value)
# daily_return_base = plotter.get_return(baseline_df, value_col_name="close")

# perf_func = timeseries.perf_stats
# perf_stats_all = perf_func(returns=daily_return_base, 
#                               factor_returns=daily_return_base, 
#                                 positions=None, transactions=None, turnover_denom="AGB")
# print("==============Baseline Strategy Stats===========")
# perf_stats_all


# In[63]:


# with pyfolio.plotting.plotting_context(font_scale=1.1):
#         pyfolio.create_full_tear_sheet(returns = daily_return,
#                                        benchmark_rets = daily_return_base, set_context=False)


# ### Authors
# github username: oliverwang15, eitin-infant


# test







