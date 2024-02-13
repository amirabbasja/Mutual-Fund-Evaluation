import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime
from scipy.stats import norm
from binance.client import Client
from dateutil import parser
import os
import math
import pandas as pd
from datetime import timedelta, datetime
import matplotlib.pyplot as plt

binance_api_key = '0gi30wUbPZWGHf5f7gBbRqihWbMwvWZDOh4Vf9BW7NgcqwLLwnaEwpWFVKb53dEG'
binance_api_secret = '7YGtVX0XXzgX4lwoX1hKHZNy3rcasNqp4fDXTNGBKYTdGw5maNfip4z4ZgYCHwil'
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
binsizes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240,"1d": 1440}
batch_size = 750

def minutes_of_new_data(symbol, kline_size, data,f_time,e_time, source):
    if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1])
    elif source == "binance": old = datetime.strptime(f_time, '%d %b %Y')
    if source == "binance": new = datetime.strptime(e_time, '%d %b %Y')
    return old, new

def get_all_binance(symbol, kline_size, _f_time,_e_time, save = False):
    filename = '%s-%s-data.csv' % (symbol, kline_size)
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df,_f_time,_e_time, source = "binance")
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])
    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): print('Downloading all available %s data for %s. Be patient..!' % (kline_size, symbol))
    else: print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))
    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else: data_df = data
    if save: data_df.to_csv(filename)
    print('All caught up..!')
    return data_df



################################################################ Timimg and Stock-Selection Ability (Treynor and Mazuy (1966)) #######################################################################

def calcRollingTimimgAbility_TM(df:pd.DataFrame, interval:timedelta):
    """
    Calculates the market-timing regression for the passed dataframe df
    Perform classical market-timing regression as described by Treynor and Mazuy (1966). 
    r - rf = alpha + beta(rm - rf) + gamma (rm - rf)^2+ 𝜀𝑖

    Args:
        df: pd.Dataframe: A dataframe containing the fund return, benchmark returns and riskfree rate
            located at first, second and thirds columns
        interval: timedelta: The interval for the rolling calculation, use days, months
             or years as an argument
    
    Returns:
        A dataframe containing the rolling results
    """

    def _TimimgAbility(x:pd.Series, _df:pd.DataFrame):
        _df = _df.loc[x.index]        
        # Vectorized calculation
        
        ri = _df.iloc[:,0] # Fund return 
        rm = _df.iloc[:,1] # benchmark return 
        rf = _df.iloc[:,2] # riskfree rate 

        excess_ri = ri - rf
        excess_rm = rm - rf
        data =pd.DataFrame([excess_ri,excess_rm,excess_rm ** 2]).T
        # Add a constant term for the intercept
        data = sm.add_constant(data)
        # Fit the market-timing regression model
        model = sm.OLS(data[0], data[['const', 1, 2]])
        results = model.fit()
        alpha,beta, gamma = results.params
        return gamma
    
    # Calculate the rolling Information
    dfOut = df.rolling(interval, min_periods = interval.days).apply(lambda x: _TimimgAbility(x, df)).iloc[:,0]
    dfOut = pd.DataFrame(dfOut)
    if interval.days == 180:
        label = "6 month"
    elif interval.days == 365:
        label = "1 year"
    elif interval.days == 365*3:
        label = "3 years"
    else:
        label = f"{interval.days} days"
    dfOut.columns = [f"TimimgAbility_{label}"]
    return dfOut


def calcRollingStockSelection_TM(df:pd.DataFrame, interval:timedelta):
    """
    Calculates the market-timing regression for the passed dataframe df
    Perform classical market-timing regression as described by Treynor and Mazuy (1966). 
    r - rf = alpha + beta(rm - rf) + gamma (rm - rf)^2+ 𝜀𝑖

    Args:
        df: pd.Dataframe: A dataframe containing the fund return, benchmark returns and riskfree rate
            located at first, second and thirds columns
        interval: timedelta: The interval for the rolling calculation, use days, months
             or years as an argument
    
    Returns:
        A dataframe containing the rolling results
    """

    def _StockSelectionAbility(x:pd.Series, _df:pd.DataFrame):
        _df = _df.loc[x.index]        
        # Vectorized calculation
        
        ri = _df.iloc[:,0] # Fund return 
        rm = _df.iloc[:,1] # benchmark return 
        rf = _df.iloc[:,2] # riskfree rate 

        excess_ri = ri - rf
        excess_rm = rm - rf
        data =pd.DataFrame([excess_ri,excess_rm,excess_rm ** 2]).T
        # Add a constant term for the intercept
        data = sm.add_constant(data)
        # Fit the market-timing regression model
        model = sm.OLS(data[0], data[['const', 1, 2]])
        results = model.fit()
        alpha,beta, gamma = results.params
        return alpha
    
    # Calculate the rolling Information
    dfOut = df.rolling(interval, min_periods = interval.days).apply(lambda x: _StockSelectionAbility(x, df)).iloc[:,0]
    dfOut = pd.DataFrame(dfOut)
    if interval.days == 180:
        label = "6 month"
    elif interval.days == 365:
        label = "1 year"
    elif interval.days == 365*3:
        label = "3 years"
    else:
        label = f"{interval.days} days"
    dfOut.columns = [f"StockSelectionAbility_{label}"]
    return dfOut


################################################################ Timimg and Stock-Selection Ability (Merton and Henriksson (1981)) #######################################################################

def calcRollingTimimgAbility_MH(df:pd.DataFrame, interval:timedelta):
    """
    Calculates the market-timing regression for the passed dataframe df
    Perform classical market-timing regression as described by Merton and Henriksson (1981). 
    r - rf = alpha + beta(rm - rf) + gamma [max(0, rm - rf)]^2+ 𝜀𝑖

    Args:
        df: pd.Dataframe: A dataframe containing the fund return, benchmark returns and riskfree rate
            located at first, second and thirds columns
        interval: timedelta: The interval for the rolling calculation, use days, months
             or years as an argument
    
    Returns:
        A dataframe containing the rolling results
    """

    def _TimimgAbility(x:pd.Series, _df:pd.DataFrame):
        _df = _df.loc[x.index]        
        # Vectorized calculation
        
        ri = _df.iloc[:,0] # Fund return 
        rm = _df.iloc[:,1] # benchmark return 
        rf = _df.iloc[:,2] # riskfree rate 

        excess_ri = ri - rf
        excess_rm = rm - rf
        data = pd.DataFrame([excess_ri,excess_rm,(excess_rm[excess_rm>=0])**2]).T
        data = data.fillna(0)
        # Add a constant term for the intercept
        data = sm.add_constant(data)
        # Fit the market-timing regression model
        model = sm.OLS(data[0], data[['const', 1, 2]])
        results = model.fit()
        alpha,beta, gamma = results.params
        return gamma
    
    # Calculate the rolling Information
    dfOut = df.rolling(interval, min_periods = interval.days).apply(lambda x: _TimimgAbility(x, df)).iloc[:,0]
    dfOut = pd.DataFrame(dfOut)
    if interval.days == 180:
        label = "6 month"
    elif interval.days == 365:
        label = "1 year"
    elif interval.days == 365*3:
        label = "3 years"
    else:
        label = f"{interval.days} days"
    dfOut.columns = [f"TimimgAbility_{label}"]
    return dfOut


def calcRollingStockSelection_MH(df:pd.DataFrame, interval:timedelta):
    """
    Calculates the market-timing regression for the passed dataframe df
    Perform classical market-timing regression as described by Merton and Henriksson (1981). 
    r - rf = alpha + beta(rm - rf) + gamma [max(0, rm - rf)]^2+ 𝜀𝑖

    Args:
        df: pd.Dataframe: A dataframe containing the fund return, benchmark returns and riskfree rate
            located at first, second and thirds columns
        interval: timedelta: The interval for the rolling calculation, use days, months
             or years as an argument
    
    Returns:
        A dataframe containing the rolling results
    """

    def _StockSelectionAbility(x:pd.Series, _df:pd.DataFrame):
        _df = _df.loc[x.index]        
        # Vectorized calculation
        
        ri = _df.iloc[:,0] # Fund return 
        rm = _df.iloc[:,1] # benchmark return 
        rf = _df.iloc[:,2] # riskfree rate 

        excess_ri = ri - rf
        excess_rm = rm - rf
        data = pd.DataFrame([excess_ri,excess_rm,(excess_rm[excess_rm>=0])**2]).T
        data = data.fillna(0)
        # Add a constant term for the intercept
        data = sm.add_constant(data)
        # Fit the market-timing regression model
        model = sm.OLS(data[0], data[['const', 1, 2]])
        results = model.fit()
        alpha,beta, gamma = results.params
        return alpha
    
    # Calculate the rolling Information
    dfOut = df.rolling(interval, min_periods = interval.days).apply(lambda x: _StockSelectionAbility(x, df)).iloc[:,0]
    dfOut = pd.DataFrame(dfOut)
    if interval.days == 180:
        label = "6 month"
    elif interval.days == 365:
        label = "1 year"
    elif interval.days == 365*3:
        label = "3 years"
    else:
        label = f"{interval.days} days"
    dfOut.columns = [f"StockSelectionAbility_{label}"]
    return dfOut


################################################################ Value At Risk #######################################################################

class VaR:
    def __init__(self,df:pd.DataFrame, ConfidenceLevel: float, TimeHorizon: int) -> None:
        """
        Calculates the value at the risk for the passed dataframe df
        VaR = Portfolio-Return - Portfolio-Variance * Z-value * sqrt(TimeHorizon)

        Args:
            df: pd.Dataframe: A dataframe containing the fund trades
            TickersReturn: pd.Dataframe: A dataframe containing the returns of all tickers
            ConfidenceLevel: float
            TimeHorizon: int: 1 is for daily VaR, 7 is for weekly VaR, 365 is for yearly VaR,
        
        Returns:
            A dataframe containing the rolling results
        """
        self.df = df
        self.ConfidenceLevel = ConfidenceLevel
        self.TimeHorizon = TimeHorizon

    def __get_symbols_return(self):
        tickers = self.df["symbol"].unique()
        start_date = (self.df["buy_date"].min().date() - timedelta(days=365)).strftime("%d %b %Y")    
        end_date = (self.df["sell_date"].max().date() + timedelta(days=1)).strftime("%d %b %Y")    
        data_frame = "1d"
        df_return = []
        for ticker in tickers:
            df = get_all_binance(ticker, data_frame, start_date,end_date, save = False)
            df = df[["timestamp","close"]].set_index("timestamp")
            df = df.astype("float")
            df.columns = [ticker]
            df = df.pct_change().dropna()
            df_return.append(df)
            
        df_return = pd.concat(df_return,axis=1)
        return df_return
    


    def calcVaR(self):
        TickersReturn = self.__get_symbols_return()
        ts = pd.concat([self.df["buy_date"],self.df["sell_date"]])
        ts = sorted(ts.unique())
        result = {}
        for t in ts:
            start = pd.to_datetime(t) - timedelta(days = 365)
            end = pd.to_datetime(t)
            df_ = self.df[(self.df["buy_date"]<=t)&(self.df["sell_date"]>t)]
            df_["value"] = df_["buy_price"] * df_["lot"]
            df_ = df_.groupby("symbol")["value"].sum()
            tickers = df_.index
            weights = (df_/df_.sum()).values
            
            df_return_history = TickersReturn[tickers]
            df_return_history = df_return_history[(df_return_history.index>=start)&(df_return_history.index<=end)]
            df_return_history = df_return_history.fillna(0)
            cov_martix = df_return_history.cov()
            avg_returns = df_return_history.mean()
            portfo_mean = avg_returns @ weights
            portfo_std = np.sqrt(weights.T @ cov_martix @ weights)
            VaR = portfo_mean - portfo_std * norm.ppf(self.ConfidenceLevel)* np.sqrt(self.TimeHorizon)
            result[t] = VaR * 100
        df_VaR = pd.DataFrame(index=result.keys(), data=result.values())
        df_VaR.columns = [f"{self.TimeHorizon}-day VaR"]
        return df_VaR

################################################################ Loss Aversion #######################################################################

class Loss_Aversion:
    def __init__(self,df:pd.DataFrame) -> None:
        """
        Loss Aversion (La)
        --------------------------------------------------
        The Loss aversion Attribute (La) evaluates whether a trading strategy exhibits symmetric \n
        behaviour independently of whether a position is winning or losing.\n
        The Loss aversion Attribute is defined as the percentage of trades that are \n
        profitable.
        --------------------------------------------------
        Parameters :
        ------------
            - df : Trading Statment pandas-DataFrame
                - This DataFrame has six columns: symbol,buy_date,sell_date,buy_price,sell_price,return
        --------------------------------------------------
        attributes : ->
        -----------
        """
        self.df = df
    
    def __get_symbols_price(self):
        tickers = self.df["symbol"].unique()
        data_frame = "1h"
        df_prices = []
        for ticker in tickers:
            df_ = self.df[self.df["symbol"]==ticker]
            start_date = df_["buy_date"].min().date().strftime("%d %b %Y")
            end_date = (df_["sell_date"].max().date() + timedelta(days=1)).strftime("%d %b %Y")    
            df = get_all_binance(ticker, data_frame, start_date,end_date, save = False)
            df = df[["timestamp","low","high"]].set_index("timestamp")
            df = df.astype("float")
            df["ticker"] = ticker
            df_prices.append(df)
        df_prices = pd.concat(df_prices)
        return df_prices
    
    def __get_max_min_price(self,df_prices,symbol:str ,buy_date,sell_date)-> tuple:
        
        df = df_prices[df_prices["ticker"]==symbol]
        df = df[(df.index>=buy_date)&(df.index<=sell_date)]
        return df["high"].max(), df["low"].min()
    
    def Loss_Aversion_data(self):
        df = self.df
        df_prices = self.__get_symbols_price()

        df["position"] = df.apply(lambda x: "long" if ((x["return"]>0) and (x["buy_price"]<=x["sell_price"])) or ((x["return"]<0) and (x["buy_price"]>=x["sell_price"])) else "short",axis=1)
        df[['max', 'min']] = df.apply(lambda x: self.__get_max_min_price(df_prices,x["symbol"],x["buy_date"],x["sell_date"]), axis=1, result_type='expand')

        df["max_return"] = df.apply(lambda x: (x["max"]/x["buy_price"]-1)*100 if x["position"]=="long" else (x["max"]/x["buy_price"]-1)*-100 , axis=1)
        df["min_return"] = df.apply(lambda x: (x["min"]/x["buy_price"]-1)*100 if x["position"]=="long" else (x["min"]/x["buy_price"]-1)*-100 , axis=1)
        df["open"] = [0] * df.shape[0]
        return df

    def get_score(self, df:pd.DataFrame,interval=100):
            df = df.reset_index(drop=True)
            for row in range(interval-1,df.shape[0]):
                df_ = df[row-99:row+1]
                df_["diff"] = df_["max_return"] + (df_["min_return"])
                average_diff = (df_["diff"]).mean()
                score = 1/(1+np.exp(-3*(average_diff)))*10 
                num_8_12 = df_[(df_["min_return"]<=-8)&(df_["min_return"]>-12)].shape[0]
                num_12_20 = df_[(df_["min_return"]<=-12)&(df_["min_return"]>-20)].shape[0]
                num_20_30 = df_[(df_["min_return"]<=-20)&(df_["min_return"]>-30)].shape[0]
                num_30_ = df_[(df_["min_return"]<=-30)].shape[0]
                score = score - num_8_12 * 0.1 - num_12_20 * 0.5 - num_20_30 * 1.5 - num_30_ * 3
                df.loc[row,"la_score"] = np.max(score,0) 
            return df
    def get_plot(self, df:pd.DataFrame,intervals:int):

        df_ = (df.iloc[-intervals:]).reset_index(drop=True)
        df_.index = df_.index+1

        green_df = df_[df_["return"] >= 0].copy()
        green_df["Height"] = np.abs(green_df["return"] - green_df["open"])

        red_df =  df_[df_["return"] < 0].copy()
        red_df["Height"] = -1 * np.abs(red_df["return"] - red_df["open"])

        fig = plt.figure(figsize=(15,7))

        ##Grey Lines
        plt.vlines(x=green_df.index, ymin=green_df["min_return"], ymax=green_df["max_return"],
                color="green")
        plt.vlines(x=red_df.index, ymin=red_df["min_return"], ymax=red_df["max_return"],
                color="orangered")

        ##Green Candles
        plt.bar(x=green_df.index, height=green_df["Height"], bottom=green_df["open"], color="green")

        ##Red Candles
        plt.bar(x=red_df.index, height=red_df["Height"], bottom=red_df["open"], color="orangered")


        plt.xlabel("Trade")
        plt.ylabel("Return")
        plt.title("Lost Aversion")

        return plt.show()