import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import calendar
from utils import *
import quantstats as qs
import scipy
from sklearn.linear_model import LinearRegression

class performanceEval:
    """
    A class to help acquire various metrics for a performance dataset.
    """
    def __init__(self, returns, riskFreeRate, benchmarkReturns, AuM, MAR, confLevel, decimals = 3, metrics = None) -> None:
        """
        Constructs the object.

        returns: pd.series: The fund/portfolio returns
        riskFreeRate: pd.series: A series indicating the risk-free
            rate of returns at the end of each month.
        benchmarkReturns: pd.series: A series indicating the risk-free
            rate of returns in each year. The risk-free returns should be in
            a monthly manner (use: (avg yearly rf return + 1)^(1/12) - 1).
        AuM: pd.series: A pandas series containing the dollar value of assets
            under management of the fund at the end of each day. If you only
            have the gross return of each month, take teh initial AuM to
            be $1 and calculate the AuM accordingly.
        MAR: float: Minimum acceptable return for the fund. It is used in some 
            metrics such as Downside deviation, etc.
        confLevel: float: Confidence level to calculate value at risk (Between 
            1 and 0)
        decimals: int: The number of decimal point for representing the results
        metrics: list of strings: Name of the specific metrics you want to 
            be calculated. If None is passed (Default) then all the metrics 
            will be calculated which may be process intensive. 
        
        * Note: The data should be indexed with datetime and date of end of 
            each month should be entered for each entry. The entries should 
            be in a ascending order (Starting from earliest months).
        
        * Note: None of the arguments "returns", "riskFreeRate" and "benchmarkReturns" 
            should contain Nan.
        """

        # Disregarded metrics: 
        # Turnover rate, Payoff ratio, Hit rate, Average holding time (Need trade data)
        # Sterling ratio (Need daily data to calculate drawdowns) 

        # Note that in the risk-adjusted measures, the expected return and risk-free return  
        # is calculated via calculating the mean of the returns and risk-free rate of returns
        # respectively.
        __FundCumulativeReturn = returns.add(1).prod() - 1
        __BenchmarkCumulativeReturn = benchmarkReturns.add(1).prod() - 1
        __FundExpectedReturn = returns.mean()
        __benchmarkExpectedReturn = benchmarkReturns.mean()
        __avgRiskFreeReturn = riskFreeRate.mean()
        
        # Acquire the necessary calculations 
        df_Returns_Barclay = pd.concat([returns, benchmarkReturns], axis = 1)

        # Calculate Net growth rate of fund
        self.netGrowthRate = netGrowthRate(AuM.iloc[-1], AuM.iloc[0])

        # Calculate the Up/Down capture rate
        self.upDownCaptureRate = calcCaptureRate(df_Returns_Barclay)

        # Calculate the win ratio
        self.winRate = returns[0<returns].shape[0]/returns.shape[0]

        # Calculate the Jensen's alpha
        # Using the formula calcBeta in the utils file calculates the beta of 
        # the fund as well
        y = np.array(returns-riskFreeRate)
        x = np.array(benchmarkReturns-riskFreeRate[returns.index.max()<=riskFreeRate.index].iloc[0])
        x = np.expand_dims(x, axis=1)
        model = LinearRegression()
        model.fit(x, y)
        self.JensenAlpha = model.intercept_
        self.beta = model.coef_[0]

        # Calculate Calmar ratio
        self.calmarRatio = qs.stats.calmar(returns)
        
        # Calculate Treynor ratio
        self.treynorRatio = calcTreynorRatio(__FundExpectedReturn, __avgRiskFreeReturn, self.beta)

        # Calculate Sortino ratio
        self.sortinoRatio = calcSortinoRatio(__FundExpectedReturn, __avgRiskFreeReturn, returns[returns<0])

        # Calculate the Sharpe ratio
        self.sharpe = qs.stats.sharpe(returns, rf = __avgRiskFreeReturn, annualize = False)
        
        # Calculate information ratio
        self.informationRatio = calcInformationRatio(returns, benchmarkReturns)

        # Calculate return std
        self.stdReturns = returns.std()

        # Calculate downside deviation
        self.downsideDeviation = calcDownsideDeviation(returns, MAR)

        # Calculate downside deviation
        self.maxDrawDown = calcMaxDrawdown(returns)

        # Calculate tracking error
        self.trackingError = (returns-benchmarkReturns).std()

        # Calculate MPPM
        # It is best to calculate this metric for 1 year (12 months of data)
        self.MPPM = calcMPPM(returns, benchmarkReturns, riskFreeRate)
        self.manipulationProofPerformanceMetric = self.MPPM

        # Calculate cross product ratio
        self.crossProductRatio = calcCRP(returns)

        # Calculate Excess return on Var
        self.excessReturnOnVar = calcExcessReturnOnVaR(returns, __avgRiskFreeReturn, confLevel)
        self.generalizedSharpeRatio = self.excessReturnOnVar

        # Calculate conditional Sharpe ratio
        self.conditionalSharpeRatio = calcConditionalSharpeRatio(returns, __avgRiskFreeReturn, confLevel)

        # Calculate modified Sharpe ratio
        self.modifiedSharpeRatio = calcModifiedSharpeRatio(returns, __avgRiskFreeReturn, confLevel)

        # Calculate upside potential ratio
        self.upsidePotentialRatio = calcUpsidePotentialRatio(returns, MAR)

        # Calculate Omega ratio
        self.omegaRatio = calcOmegaRatio(returns, MAR)

        # Calculate d ratio
        self.dRatio = calcdRatio(returns)

        # Calculate kappa 3 ratio
        self.kappa3Ratio = calcKappa3Ratio(returns, MAR)

        # Calculate Sterling ratio
        # Calculated for 5 max drawdowns
        self.sterlingRatio = calcSterlingRatio(returns, __avgRiskFreeReturn,5)

        # Calculate Sterling-calmar ratio
        self.sterlingCalmarRatio = calcSterlingCalmarRatio(AuM, returns, __avgRiskFreeReturn)

        # Calculate Burke ratio
        self.burkeRatio = calcBurkeRatio(returns, __avgRiskFreeReturn)

        # Calculate Ulcer index
        self.ulcerIndex = calcUlcerIndex(returns)

        # Calculate adjusted Sharpe ratio
        self.adjustedSharpeRatio = calcAdjustedSharpeRatio(returns, __avgRiskFreeReturn)

        # Calculate prospect ratio
        self.prospectRatio = calcProspectRatio(returns, MAR)



        self.metrics = {
            "net growth rate" : np.round(self.netGrowthRate, decimals),
            "up/down capture rate" : np.round(self.upDownCaptureRate, decimals),
            "win rate" : np.round(self.winRate, decimals),
            "Jensen's alpha" : np.round(self.JensenAlpha, decimals),
            "calmar ratio" : np.round(self.calmarRatio, decimals),
            "treynor ratio" : np.round(self.treynorRatio, decimals),
            "sortino ratio" : np.round(self.sortinoRatio, decimals),
            "sharpe ratio" : np.round(self.sharpe, decimals),
            "information ratio" : np.round(self.informationRatio, decimals),
            "standard deviation of returns" : np.round(self.stdReturns, decimals),
            "downside deviation" : np.round(self.downsideDeviation, decimals),
            "max drawdown" : np.round(self.maxDrawDown, decimals),
            "tracking error" : np.round(self.trackingError, decimals),
            "cross product ratio" : np.round(self.crossProductRatio, decimals),
            "excess return on VaR (Generalized Sharpe ratio)" : np.round(self.excessReturnOnVar, decimals),
            "conditional Sharpe ratio" : np.round(self.conditionalSharpeRatio, decimals),
            "modified Sharpe ratio" : np.round(self.modifiedSharpeRatio, decimals),
            "upside potential ratio" : np.round(self.upsidePotentialRatio, decimals),
            "omega ratio" : np.round(self.omegaRatio, decimals),
            "d ratio" : np.round(self.dRatio, decimals),
            "kappa 3 ratio" : np.round(self.kappa3Ratio, decimals),
            "sterling ratio" : np.round(self.sterlingRatio, decimals),
            "sterling calmar ratio" : np.round(self.sterlingCalmarRatio, decimals),
            "burke ratio" : np.round(self.burkeRatio, decimals),
            "ulcer index" : np.round(self.ulcerIndex, decimals),
            "adjusted Sharpe ratio" : np.round(self.adjustedSharpeRatio, decimals),
            "prospect ratio" : np.round(self.prospectRatio, decimals),
        }

