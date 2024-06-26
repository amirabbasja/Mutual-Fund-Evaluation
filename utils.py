import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy
from dateutil.relativedelta import relativedelta
import calendar
import quantstats as qs
from scipy.stats import norm, t

# Necessary methods
def getDataVisionTrack(loc):
    """
    acquires fund performance data from Vision Track platform

    Args: 
        loc: str: The location of the html file containing the data
    """
    df = pd.read_html(loc)
    df = df[1]
    cols = df.columns.tolist()
    colsToSelect = [cols[1]]+[cols[2]]+[cols[10]]+cols[25:-1]
    df = df[colsToSelect]
    df = df.dropna(axis=1,how="all")
    
    return df

def divideDateRange(rangeStart, rangeEnd, interval, count):
    """
    Divides a date range to specific intervals. The process starts from the end date and
    goes backwards to reach the start date.

    Args:
        rangeStart, rangeEnd: datetime.datetime: Start and end of the desired date range.
        interval: str: One of the following items [months, days, years]
        count: int: The length of each interval 
    
    Returns: 
        A list of dates.
    """
    lstOutput = []
    temp = rangeEnd
    while rangeStart < temp:
        lstOutput.append(temp)
        if interval == "days":
            temp -= relativedelta(days=count)
        elif interval == "months":
            temp -= relativedelta(months=count)
        elif interval == "years":
            temp -= relativedelta(years=count)
    
    return lstOutput

def barclayHedgeIndex(fileLoc):
    """
    Gets the Barclay Hedge Index data and returns it as a dataframe.
    """
    # Map the months to numbers
    mon = {month: index for index, month in enumerate(calendar.month_abbr) if month}

    # Read and process the data
    data = pd.read_excel(fileLoc, header=1)
    data.dropna(inplace=True)
    data.columns = ["year"]+ [str(i) for i in range(1,13)] + ["YTD"]
    
    tmpMonth = []
    tmpReturn = []
    for _, row in data.iterrows():
        for i in range(1,13):
            tmpMonth.append(datetime.datetime(year=int(row.year),month=i,day=1))
            tmpReturn.append(float(row[str(i)]))
    
    df = pd.DataFrame(list(zip(tmpMonth,tmpReturn)), columns=["Date", "barclayReturn"])
    return df

def getMonthlyReturns(input, outCols = None):
    """
    Returns a dataframe containing monthly returns of the fund.

    Args:
        input: pd.Dataframe: A dataframe containing AuM of the fund at the end 
            of each day. Should contain columns named ["Date", "Value"]
        outCols: list: A list of strings, containing the output dataframe's 
            column names (Optional).
    """
    input = input.sort_values(["Date"], ascending = True)

    cols = input.columns
    endDate = input.iloc[-1][cols[0]] 
    startMonth = input.iloc[0][cols[0]].month
    startYear = input.iloc[0][cols[0]].year

    rangeStart = datetime.datetime(year = startYear, month=startMonth, day=1)
    rangeEnd = rangeStart
    monthReturn, monthStart, monthEnd = [], [], []

    while True:
        # Setting the difference between start and end of each month as its return
        rangeEnd += relativedelta(months=1)

        # Get the dataframe for each month
        tmpDataframe = input[(rangeStart <= input[cols[0]])&(input[cols[0]] < rangeEnd)]

        # Calculate the return

        monthReturn.append((tmpDataframe.iloc[-1][cols[1]]-tmpDataframe.iloc[0][cols[1]])/tmpDataframe.iloc[0][cols[1]])
        monthStart.append(rangeStart)
        monthEnd.append(rangeStart + relativedelta(months=1))

        if endDate <= rangeEnd:
            break
        rangeStart += relativedelta(months=1)

    fundMonthlyReturns = pd.DataFrame(list(zip(monthEnd,monthReturn)), columns=["Date", "fundReturn"] if outCols == None else outCols)
    return fundMonthlyReturns

def calcCum(returns):
    """
    Calculates the cumulative return of a pandas series

    Args: 
        returns: pd.Series: A pandas series or a column
    
    Returns:
        A float number
    """
    __cumReturn = 1
    for ret in returns:
        __cumReturn = __cumReturn * (1 + np.float32(ret))
    
    return __cumReturn - 1

def calcCumReturnInRange(df, interval, count):
    """
    Calculates the cumulative return in a range based manner. 

    Args: 
        df: pd.Dataframe: A pandas dataframe containing monthly returns. 
            First columns should be end of the month in datetime, The rest 
            of the columns should be returns. This columns should be denoted 
            by "Date" label.
        interval: str: One of the following items [months, days, years]
        count: int: The length of each interval 
    
    Returns:
        A pandas dataframe
    """
    # Sorting the dataframe
    df = df.sort_values(["Date"], ascending=False, ignore_index=True)
    __lastDate, __firstDate = df.iloc[0].Date, df.iloc[-1].Date

    dateRange = divideDateRange(__firstDate, __lastDate, interval, count)
    
    # The output dataframe
    outDf = pd.DataFrame(columns=["intervalStart", "intervalEnd"] + list(df.columns)[1:])
    
    __tmpLst = []
    for i in range(len(dateRange)-1):
        __df = df[(dateRange[i+1]<=df.Date)&(df.Date<dateRange[i])]
        __tmpLst = []

        for column in __df:
            # Calculate cumulative return for each columns except Date
            if column == "Date": continue
            __tmpLst.append(calcCum(__df[column]))
        
        # Add the new row to the output dataframe
        outDf.loc[i] = [dateRange[i+1],dateRange[i]-relativedelta(days=1)] + __tmpLst

    return outDf

def netGrowthRate(AuMLastDay, AuMFirstDay):
    """
    Returns the net growth rate of the fund's AuM 
    The formula: (Last day value - First day value)/First day value = growth rate

    Args:
        AuMLastDay, AuMFirstDay: float: AuM of the fund at the beginning and end of the interval
    
    Returns: 
        float.
    """
    return (AuMLastDay-AuMFirstDay)/AuMFirstDay

def getRollingGrowth(dfReturn, interval):
    """
    Gets the rolling growth rate of the dfReturn using netGrowthRate function

    Args:
        dfReturn: pd.Dataframe: The index should be date and the first columns the property you want to get
        interval: timedelta: The interval for the rolling calculation, use days, months or years as an argument
    
    Returns:
        A dataframe containing the rolling results in percentage
    """
    
    outDf = dfReturn.rolling(interval, min_periods = interval.days).apply(lambda x: netGrowthRate(x.iloc[-1], x.iloc[0]))

    # Get percentage returns
    outDf.iloc[:,0] = outDf.iloc[:,0] * 100

    return outDf

def plotMainAndSubPlot(main, sub, titles, eqAxisRange):
    """
    Plots the main chart with separate subplots below it

    Args:
        main: pd.Dataframe: A dataframe to plot as the main chart. indexes should be datetime
        sub: list: a list of lists,  dataframes to be plotted on a subplot each. indexes should be
             datetime. All the columns will be plotted on the same subplot
        titles: list: A list containing titles for each plot
        eqAxisRange: bool: If true, all plots on the figure will have same axis range
    """
    # Get axis range
    __xlim = (main.index.min(),main.index.max())
    if eqAxisRange:
        for df in sub:
            if __xlim[1] < df.index.max(): __xlim[1] = df.index.max()
            if df.index.min() < __xlim[0]: __xlim[0] = df.index.max()

    plotCount = 1 + len(sub)

    fig, ax = plt.subplots(plotCount, 1, gridspec_kw={'height_ratios': [3]+[1]*len(sub), 'hspace': .4})

    # The main plot has a 6 inches height, the rest of the subplots have a 2 inches height
    # Width of all charts are set to 10 inches
    fig.set_size_inches(10, 2 * (2+plotCount))


    # Plot the main chart
    ax[0].plot(main)
    ax[0].set_xlim(__xlim) if eqAxisRange else ax[0].set_xlim((main.index.min(),main.index.max()))
    ax[0].axhline(0, color='black', linewidth = .5)
    ax[0].set_title(titles[0])
    # ax[0].figure.set_size_inches(10,15)

    for i in range(len(sub)):
        ax[i+1].plot(sub[i], label = list(sub[i].columns) if sub[i].shape[1]!=1 else list(sub[i].columns)[0])
        ax[i+1].axhline(0, color='black', linewidth = .5) # Add the zero line (May not be seen)
        ax[i+1].set_title(titles[i+1])# Add title
        ax[i+1].legend(prop = { "size": 8 }, loc ="upper left") # Add legend

        # Set the Y axis limit for each subplot
        # The Y limit of each subplot is acquired by the min/max of the data it is plotting (Excluding the index)
        # A 2% padding to top and bottom of the chart is added for better UX
        subP_maxY =  sub[i].iloc[:,0:].max().max() * 1.02
        subP_minY =  sub[i].iloc[:,0:].min().min() * 0.98
        ax[i+1].set_ylim((subP_minY,subP_maxY)) 

        # Set the X axis limit for each subplot. If eqAxisRange = False, matplotlib will determine the X limits.
        ax[i+1].set_xlim(__xlim) if eqAxisRange else ax[i+1].set_xlim((sub[i].index.min(),sub[i].index.max()))

def calcCaptureRate(df:pd.DataFrame):
    """
    Calculates the capture rate of fund relative to benchmark.

    Args: 
        df: pd.Dataframe: A dataframe containing two columns. First the 
            fund returns and the second, the benchmark returns.
    """
    
    numerator, denumerator = 1, 1
    numerator = (df.iloc[:,0]+1).product(axis = 0) - 1 # cumulative 
    denumerator = (df.iloc[:,1]+1).product(axis = 0) - 1 # cumulative
    
    return (numerator - 1) / (denumerator - 1) if denumerator != 1 else 0

def calcRollingCaptureRate(df:pd.DataFrame, interval:datetime.timedelta):
    """
    Calculates the rolling capture ratio using calcCaptureRate

    Args:
        df: pd.Dataframe: A dataframe containing the fund return and benchmark return
            fund and benchmark columns should be located at first and second column 
            respectively.
        interval: timedelta: The interval for the rolling calculation, use days, months
             or years as an argument
    
    Returns:
        Two dataframe containing up and down capture rates respectively
    """
    # Internal functions to get the days that benchmark was up or down respectively
    def __upDays(x:pd.Series, _df:pd.DataFrame):
        # Assuming benchmark is at the second column
        _df = _df.loc[x.index]
        return  _df[0<_df.Benchmark]
    def __dnDays(x:pd.Series, _df:pd.DataFrame):
        # Assuming benchmark is at the second column
        _df = _df.loc[x.index]
        return _df[_df.iloc[:,1]<0]

    rollingUpCapture = df.rolling(interval, min_periods = interval.days).apply(lambda x: calcCaptureRate(__upDays(x, df)))
    rollingDownCapture = df.rolling(interval, min_periods = interval.days).apply(lambda x: calcCaptureRate(__dnDays(x, df)))

    return rollingUpCapture, rollingDownCapture

def calcRollingWinRate(df: pd.DataFrame, interval:datetime.timedelta):
    """
    Win rate refers to the probability of a fund being purchased at any time and held for a
    certain period of time before making a profit. For example, if a fund has been running
    for a year and has made profits in 9 months, the investment success rate of the fund is
    75%. This function calculates win ratio of dataframe df, in the required interval

    Args:
        df: pd.Dataframe: A dataframe containing the fund return on the first column
        interval: timedelta: The interval for the rolling calculation, use days, months
             or years as an argument
    
    Returns:
        A dataframe containing the rolling win rate for the end of specified date
    """
    def __internalFcn(__x):
        """
        Calculates win rate
        """
        __tmp = __x[0<__x].shape[0]/__x.shape[0]
        return __tmp

    dfOut = df.rolling(interval, min_periods = interval.days).apply(lambda x: __internalFcn(x))
    return dfOut

def loadTrades(path):
    """
    Loads the trades

    Args:
        path: str: the path for a csv file
    
    Returns:
        A pandas dataframe
    """
    df = pd.read_csv(path)

    # Cast the dates
    df["buy_date"] = pd.to_datetime(df["buy_date"], dayfirst = True, format = "%d/%m/%Y %H:%M")
    df["sell_date"] = pd.to_datetime(df["sell_date"], dayfirst = True, format = "%d/%m/%Y %H:%M")

    # Cast the numbers to float
    df[["buy_price","sell_price","lot","pnl","return"]] = df[["buy_price","sell_price","lot","pnl","return"]].astype(float)

    # Cast the symbols to str
    df["symbol"] = df["symbol"].astype(str)
    return df

def loadRiskFreeRate(path):
    """
    Loads the risk-free rate

    Args:
        path: str: the path for a csv file
    
    Returns:
        A pandas dataframe
    """
    df = pd.read_csv(path)
    df = df[["Date", "Price"]]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst = True, format = "%m/%d/%Y")
    df["Price"] = df["Price"]/100.
    return df

def calcMPPM(ri, rb, rf):
    """
    Calculates the manipulation proof performance metric.
    Although there is no limit to the amount of months to calculate this
    metric for, It is best to calculate this metric in a yearly basis (
    Pass only 12 months of data).

    Args:
        ri: pd.Series: Portfolio return.
        rb: pd.Series: Benchmark return.
        rf: pd.Series: Risk-free return.
    """
    # Calculate 1+rb
    E_rb = np.log(rb.add(1).mean())

    # Calculate 1+rf
    rfPlusOne = np.log(rf.add(1).mean())

    # Calculate Variance of 1+rb
    var = np.log(rb.add(1)).var()

    # Calculate rho
    rho = ((E_rb) - (rfPlusOne))/var

    MPPM_fund = 1/(1-rho)* np.log(((ri.add(1)/rf.add(1)).pow(1-rho)).mean())

    return MPPM_fund

def calcCRP(ri):
    """
    Calculates the cross product ratio

    Args:
        ri: pd.Series: Portfolio return.
    """
    ri.index = range(0, ri.shape[0],1)
    dftmp = pd.DataFrame({"FundReturn":ri})
    dftmp["Result"] = "-" # FIlling the rows with dummy values
    dftmp["DoubleResult"] = "-" # FIlling the rows with dummy values 

    median = dftmp.FundReturn.median()
    for i,row in dftmp.iterrows():
        dftmp.loc[i,"Result"] = "W" if median<=row["FundReturn"] else "L" 

        if i!=0:
            if (dftmp.loc[i-1,"Result"] == "L") and (dftmp.loc[i,"Result"] == "W"): dftmp.loc[i,"DoubleResult"] = "LW"
            if (dftmp.loc[i-1,"Result"] == "W") and (dftmp.loc[i,"Result"] == "L"): dftmp.loc[i,"DoubleResult"] = "WL"
            if (dftmp.loc[i-1,"Result"] == "L") and (dftmp.loc[i,"Result"] == "L"): dftmp.loc[i,"DoubleResult"] = "LL"
            if (dftmp.loc[i-1,"Result"] == "W") and (dftmp.loc[i,"Result"] == "W"): dftmp.loc[i,"DoubleResult"] = "WW"
    return (dftmp[dftmp.DoubleResult == "WW"].shape[0]+dftmp[dftmp.DoubleResult == "LL"].shape[0])/(dftmp[dftmp.DoubleResult == "LW"].shape[0]+dftmp[dftmp.DoubleResult == "WL"].shape[0])

def calcRollingTrackingError(df: pd.DataFrame, interval:datetime.timedelta):
    """
    Calculates the tracking error of the fund relative to a specific 
    Benchmark. The dataframe should have two columns, portfolio and 
    benchmark returns.

    Args:   
        df: pd.Dataframe: The fund's returns
        interval: datetime.timedelta:The window size in days
    """
    
    def __internalFcn(x:pd.Series, _df:pd.DataFrame):
        _df = _df.loc[x.index]
        diff = _df.iloc[:,0]-_df.iloc[:,1]

        # Return the tracking error
        return diff.std()
    
    # Calculate the Downside deviation
    dfOut = pd.DataFrame(df.rolling(interval, min_periods = interval.days).apply(lambda x: __internalFcn(x, df)).iloc[:,0])
    dfOut.columns = ["Tracking error"]
    
    return pd.DataFrame(dfOut)

def loadBTCReturn(path):
    """
    Loads the risk-free return rate

    Args:
        path: str: the path for a csv file
    
    Returns:
        A pandas dataframe
    """
    df = pd.read_csv(path)
    df = df[["Date", "Price"]]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst = True, format = "%m/%d/%Y")
    df.Price = df.Price.str.replace(",","").astype(float)
    df = df.sort_values("Date", ascending=True, ignore_index=True)
    return df

def calcJensenAlpha(ri, rm, rf, beta):
    """
    Calculates Jensen's alpha

    Args:
        ri: float: the realized return of the portfolio or investment
        rm: float: the realized return of the appropriate market index
        rf: float: the risk-free rate of return for the time period
        beta: float: the beta of the portfolio of investment with respect to the chosen market index

    Returns: A float number
    """
    return ri - (rf+beta*(rm-rf))

def calcTreynorRatio(ri, rf, beta):
    """
    Calculates Jensen's alpha

    Args:
        ri: float: The portfolio return
        rf: float: the risk-free rate of return for the time period
        beta: float: the beta of the portfolio of investment with respect to the chosen market index

    Returns: A float number
    """
    return (ri-rf)/beta

def rollingJensenAlpha(df: pd.DataFrame, interval:datetime.timedelta):
    """
    Calculates the rolling Jensen's ratio for the passed dataframe df

    Args:
        df: pd.Dataframe: A dataframe containing the fund return, benchmark return
            and the risk free return. Fund, benchmark and risk free columns should 
            be located at first, second and third columns respectively.
        interval: timedelta: The interval for the rolling calculation, use days, months
             or years as an argument
    
    Returns:
        A dataframe containing the rolling results
    """
    def __internalFcn(x:pd.Series, df:pd.DataFrame):
        df = df.loc[x.index]
        beta = calc_beta(df.iloc[:,0], df.iloc[:,1])

        # Vectorized calculation
        rf = (df.iloc[:,2]+1).prod() - 1 # cumulative Risk free return of interval
        rm = (df.iloc[:,1]+1).prod() - 1 # cumulative Market benchmark return
        ri = (df.iloc[:,0]+1).prod() - 1 # cumulative Fund return 

        # Calculate jensen's alpha
        return calcJensenAlpha(ri, rm, rf, beta)
    
    # Calculate the rolling alpha
    dfOut = df.rolling(interval, min_periods = interval.days).apply(lambda x: __internalFcn(x, df))

    # All three columns are identical. Choose first column and change the its label
    dfOut = pd.DataFrame(dfOut.iloc[:,0])
    dfOut.columns = ["Jensen alpha"]

    return dfOut # Change series to dataframe

def calcRollingCalmarRatio(df:pd.DataFrame, interval:datetime.timedelta):
    """
    Calculates the rolling Calmar ratio for the passed dataframe df

    Args:
        df: pd.Dataframe: A dataframe containing the fund return located at the first 
            column
        interval: timedelta: The interval for the rolling calculation, use days, months
             or years as an argument
    
    Returns:
        A dataframe containing the rolling results
    """
    def __internalFcn(x:pd.Series, _df:pd.DataFrame):
        _df = _df.loc[x.index]
        
        # Get calmar for chosen df
        return qs.stats.calmar(pd.DataFrame(_df))
    
    # Calculate the rolling calmar
    dfOut = df.rolling(interval, min_periods = interval.days).apply(lambda x: __internalFcn(x, df))
    dfOut.columns = ["calmar"]

    return pd.DataFrame(dfOut)

def calcSortinoRatio(rp, rf, nr):
    """
    Calculates Sortino ratio
    Ref: https://www.wallstreetprep.com/knowledge/sortino-ratio/

    Args:
        rp: float: The portfolio return
        rf: float: the risk-free rate of return for the time period
        nr: pd.Series: A dataframe containing the negative returns

    Returns: A float number
    """
    return (rp-rf)/np.std(nr,axis = 0)

def calc_beta(dfFund, dfBenchmark):
    """
    Calculates beta of an asset with respect to a benchmark.

    Args:
        dfFund, dfBenchmark: pd.Series: Two series of pandas data 
        representing the fund and the benchmark's returns.
    
    Returns: 
        A float number representing the beta
    """
    
    m = dfBenchmark # market returns are column zero from numpy array
    s = dfFund # stock returns are column one from numpy array
    covariance = np.cov(s,m) # Calculate covariance between stock and market
    beta = covariance[0,1]/covariance[1,1]
    return beta

def calcInformationRatio(rp, rb):
    """
    Calculates the Information ratio for the passed series

    Args:
        rp: pd.Series: Portfolio return
        rb: pd.Series: Benchmark return
    
    Returns:
        A dataframe containing the rolling results
    """
    diff = rp - rb
    return diff.mean()/diff.std()

def calcRollingInformationRatio(df:pd.DataFrame, interval:datetime.timedelta):
    """
    Calculates the rolling Information ratio for the passed dataframe df

    Args:
        df: pd.Dataframe: A dataframe containing the fund return and benchmark returns 
            located at first and second columns
        interval: timedelta: The interval for the rolling calculation, use days, months
             or years as an argument
    
    Returns:
        A dataframe containing the rolling results
    """
    def __internalFcn(x:pd.Series, _df:pd.DataFrame):
        _df = _df.loc[x.index]
        ri = _df.iloc[:,0] # Benchmark return
        rm = _df.iloc[:,1] # Portfolio return

        diff = ri - rm

        return diff.mean()/diff.std()
    
    # Calculate the rolling Information
    dfOut = df.rolling(interval, min_periods = interval.days).apply(lambda x: __internalFcn(x, df)).iloc[:,0]
    dfOut.columns = ["Information"]

    return pd.DataFrame(dfOut)

def calcRollingTreynorRatio(df:pd.DataFrame, interval:datetime.timedelta):
    """
    Calculates the rolling Treynor ratio for the passed dataframe df

    Args:
        df: pd.Dataframe:  A dataframe containing the fund return, benchmark return
            and the risk free return. Fund, benchmark and risk free columns should 
            be located at first, second and third columns respectively.
        interval: timedelta: The interval for the rolling calculation, use days, months
             or years as an argument
    
    Returns:
        A dataframe containing the rolling results
    """
    def __internalFcn(x:pd.Series, _df:pd.DataFrame):
        _df = _df.loc[x.index]
        beta = calc_beta(_df.iloc[:,0], _df.iloc[:,1])
        
        # Vectorized calculation
        rf = (_df.iloc[:,2]+1).prod() - 1 # Risk free return of interval
        rm = (_df.iloc[:,1]+1).prod() - 1 # Market benchmark return
        ri = (_df.iloc[:,0]+1).prod() - 1 # Fund return 

        return calcTreynorRatio(ri,rf,beta)
    
    # Calculate the rolling Treynor
    dfOut = df.rolling(interval, min_periods = interval.days).apply(lambda x: __internalFcn(x, df)).iloc[:,0]
    dfOut.columns = ["treynor"]

    return pd.DataFrame(dfOut)

def calcRollingSTD(df: pd.DataFrame, interval:datetime.timedelta):
    """
    Calculates the rolling standard deviation of the returns. Only one
    column for portfolio's daily return is needed.
    """
    dfOut = df.rolling(interval, interval.days).apply(lambda x: x.std())
    dfOut.columns = ["Standard deviation"]
    return dfOut

def calcMaxDrawdown(ri):
    """
    Calculates the maximum drawdown
    
    Args:
        ri: pd.Series: Fund's returns in desired intervals. Index should be 
            datetime.
        
        * Note: Avoid passing fund's AuM as ri.
    """
    ri = (ri.dropna()+1).cumprod() # Take fund's starting AuM to be $1 and calculate its AuM with respect to returns
    Roll_Max = ri.cummax()
    Daily_Drawdown = ri/Roll_Max - 1.0
    drawDown = Daily_Drawdown.cummin()
    return drawDown[-1]

def calcDownsideDeviation(ri, MAR):
    """
    Calculates the downside deviation of the fund relative to a specific 
    MAR (Minimum acceptable return). 

    Args:
        ri: pd.series: The portfolio return.
        MAR: float: Minimum acceptable return (example: .3 for 30%)
    """
    # Calculate min(0, ri - MAR)
    ri = ri - MAR
    ri = ri[ri<0] 

    return np.sqrt((ri**2).sum()/ri.shape[0])

def calcRollingDownsideDeviation(df: pd.DataFrame, interval:datetime.timedelta, MAR: float):
    """
    Calculates the downside deviation of the fund relative to a specific 
    MAR (Minimum acceptable return). The dataframe should have one column 
    which is the fund's return at a specific date.

    Args:   
        df: pd.Dataframe: The fund's returns
        interval: datetime.timedelta: The window size
        MAR: float: Minimum acceptable return (Not in percentages)
    """

    def __internalFcn(x:pd.Series, _df:pd.DataFrame):
        _df = _df.loc[x.index]
        _df = _df.iloc[:,0] - MAR
        _df = _df[_df<0]
        _df=np.sqrt((_df**2).sum()/_df.shape[0])
        return _df
    
    # Calculate the Downside deviation
    dfOut = pd.DataFrame(df.rolling(interval, min_periods = interval.days).apply(lambda x: __internalFcn(x, df)).iloc[:,0])
    dfOut.columns = ["Downside Deviation"]
    
    return pd.DataFrame(dfOut)

def calcPayoffRatio(df:pd.DataFrame):
    """
    Calculates payoff ratio by dividing the average profit to average loss trades.
    formula: Payoff ratio = average profit per winning trade / average loss per 
    losing trade.
    
    Args:
        df: pd.Dataframe: A dataframe containing PNLs for each individual trade. 
            The index should be in datetime 
    """
    return pd.DataFrame(df[0<df.PNL].mean()/df[df.PNL<0].mean().abs()).iloc[0,0]


def calcHitRate(df:pd.DataFrame):
    """
    Calculates the hit rate from a track record by dividing number of winning 
    trades by the total number of executed trades
    
    Args:
        df: pd.Dataframe: A dataframe containing PNLs for each individual trade. 
            The index should be in datetime 
    """
    return df[0<df.PNL].shape[0]/(df.shape[0])*100

def calcTurnOverRate(df:pd.DataFrame):
    """
    Calculates the turnover rate of the fund/algorithm with the formula:
    Turnover Rate = MAX (total cost of buying stocks during the reporting 
    period, total revenue from selling stocks during the reporting period
    )/AVG (market value of stocks)

    Args: 
        df: pd.Dataframe: The dataframe containing the following
            columns: buy_price, sell_price, the trade volume, and
            also the trade's closing date as index
    
    Returns: A float as the turnover rate
    """
    buyCost = df.buy_price*df.lot
    sellRevenue = df.sell_price*df.lot
    numerator = np.sum(np.maximum(buyCost, sellRevenue))
    denumerator = np.mean(df.AuM)
    
    return float(numerator/denumerator)

def calcAvgHoldingTime(df:pd.DataFrame):
    """
    Calculates the average holding time for the fund manager. The 
    formula denoted below: The average holding time of stocks in 
    a fund = Interval duration /Turnover rate

    Note that the turnover rate is calculated by the calcTurnOverRate 
    function.

    Args: 
        df: pd.Dataframe: The dataframe containing the following
            columns: buy_price, sell_price, the trade volume, and
            also the trade's closing date as index
    
    Returns: A float as holding time 
    """
    # Calculate turnover rate
    turnoverRate = calcTurnOverRate(df)

    #Calculate the interval duration
    intervalDuration = (np.abs(df.index[0] - df.index[-1])).days
    
    return float(intervalDuration/turnoverRate)


def calcVarCVar(df, confLevel, method = "historical", distribution = "normal", dof = 6):
    """
    Calculates value at risk (VAR) and conditional value at risk (CVAR) in 
    both historical and parametric methods. Both these values are known as 
    the loss incurred on the investment, so the absolute values of calculations 
    are returned. CVAR is calculated by assuming a normal or t-distribution.

    Historical method: We assume that future returns will follow a similar 
        distribution to historical returns.
    Parametric method:  The parametric method looks at the price movements 
        of investments over a look-back period and uses probability theory
        to compute a portfolio's maximum loss. This method for the value 
        at risk calculates the standard deviation of price movements of an
        investment or security. Assuming stock price returns and volatility
        follow a normal distribution, the maximum loss within the specified
        confidence level is calculated.

    Args:
        df: pd.Dataframe: A pandas dataframe or series containing returns in each
            desired interval (daily, monthly, etc.)
        confLevel: float: A float between 0 and 1, indicating the confidence level
            for VAR calculation
        method: str: The method to calculate var, can be historical or parametric
        distribution: str: The distribution to assume when calculating parametric 
            VAR and CVAR. Two distributions are acceptable, normal and t-distribution.
        dof: int: Degrees of freedom. Used in t-distribution formula. Disregard if you
            chose normal distribution
    
    Returns:
        Two floats, the first is VAR and teh second is CVAR.
    """

    if method == "historical":
        var = df.quantile( 1 - confLevel)
        cvar = df[df>= var].mean()
    elif method == "parametric":

        # Assuming normal distribution
        if distribution == "normal":
            z_score = norm.ppf(q= 1 - confLevel)
            var = df.mean() - (norm.ppf(confLevel) * df.std())
            cvar = df.mean() - df.std() * (norm.pdf(z_score)/(1-confLevel))
        elif distribution == "t-distribution":
            xdof = t.ppf(1-confLevel, dof)
            var = np.sqrt((dof-2)/2)*t.ppf(confLevel, dof) * df.std() - df.mean()
            cvar = -1/(1-confLevel) * (1-dof)**(-1) * (dof-2+xdof**2) * t.pdf(xdof, dof) * df.std() - df.mean()
        else:
            raise "The distribution argument should should be normal or t-distribution"


    else:
        raise "The method argument should be either historical or parametric"

    return abs(var), abs(cvar)

def calcMVaR(df, confLevel):
    """
    Calculates modified VaR (Favre and Galeano, 2002). This method adjusts VAR
    for kurtosis and skewness using a Cornish-Fisher expansion.

    Args:
        df: pd.Dataframe: A pandas dataframe or series containing returns in each
            desired interval (daily, monthly, etc.)
        confLevel: float: A float between 0 and 1, indicating the confidence level
            for VAR calculation
    """
    z = abs(norm.ppf(1-confLevel))
    s = scipy.stats.skew(df)
    k = scipy.stats.kurtosis(df)
    t = z + 1/6*(z**2-1)*s+1/24*(z**3-3*z)*k-1/36*(2*z**3-5*z)*s**2

    MVaR = df.mean() - t * df.std()
    return MVaR


def calcModifiedSharpeRatio(df, rf, confLevel, method = "parametric", distribution = "normal", dof = 6):
    """
    Calculates the modified Sharpe ratio.
    The formula: (Rp - Rf) / MVAR

    Args:
        df: pd.dataframe: A dataframe containing portfolio returns
        rf: Risk Free return
        Rest of the args are adopted from calcVarCVar function
    """

    mvar = calcMVaR(df, confLevel)
    return (df.mean() - rf) / mvar

def calcConditionalSharpeRatio(df, rf, confLevel, method = "parametric", distribution = "normal", dof = 6):
    """
    Calculates the conditional Sharpe ratio (Agarwal 2003).
    The formula: (Rp - Rf) / CVAR

    Args:
        df: pd.dataframe: A dataframe containing portfolio returns
        rf: Risk Free return
        Rest of the args are adopted from calcVarCVar function
    """

    _, cvar = calcVarCVar(df, confLevel, method, distribution, dof)
    return (df.mean() - rf) / cvar

def calcExcessReturnOnVaR(df, rf, confLevel, method = "parametric", distribution = "normal", dof = 6):
    """
    Calculates the excess return on VaR (Dowd 2000), Also known as 
    generalized Sharpe ratio.
    The formula: (Rp - Rf) / VaR

    Args:
        df: pd.dataframe: A dataframe containing portfolio returns
        rf: Risk Free return
        Rest of the args are adopted from calcVarCVar function
    """

    var, _ = calcVarCVar(df, confLevel, method, distribution, dof)
    return (df.mean() - rf) / var


def calcUpsidePotentialRatio(df, MVAR):
    """
    Calculates upside potential ratio (Sortino, 1999)

    Args:
        df: pd.dataframe: A pandas dataframe/series containing the returns
            in desired timeframe.
        MVAR: float: Minimum acceptable return (e.g. 0.1 for a 10% return). 

    Returns:
        A float, indicating the upside potential ratio
    """
    upside_returns = df[df > MVAR] - MVAR
    downside_returns = df[df < MVAR] - MVAR
    upside_potential = upside_returns.mean()
    downside_risk = downside_returns.std()

    return upside_potential / downside_risk if downside_risk != 0 else float('inf')

def calcOmegaRatio(df, MAR):
    """
    Calculates omega ratio (Shadwick and Keating, 2002). The omega ratio 
    can be used as a ranking statistic; the higher the better. It equals 
    1 when rT is the mean return.

    Args:
        df: pd.dataframe: A pandas dataframe/series containing the returns
            in desired timeframe.
        MAR: float: Minimum acceptable return (e.g. 0.1 for a 10% return). 

    Returns:
        A float, indicating the upside potential ratio
    """
    numerator =  np.sum(np.max(df-MAR, 0))
    denumerator =  np.sum(np.max(MAR-df, 0))

    return numerator / denumerator

def calcBernadoLedoitRatio(df):
    """
    Calculates Bernardo and Ledoit (1996) ratio which is a special case of 
    the omega ratio with MAR = 0.

    Args:
        df: pd.dataframe: A pandas dataframe/series containing the returns
            in desired timeframe.
        MAR: float: Minimum acceptable return (e.g. 0.1 for a 10% return). 

    Returns:
        A float, indicating the upside potential ratio
    """
    return calcOmegaRatio(df,0)


def calcdRatio(df):
    """
    Calculates d ratio (Lavinio, 1999) which is  is similar to the Bernado 
    Ledoit ratio but inverted and taking into account the frequency of 
    positive and negative returns.

    Args:
        df: pd.dataframe: A pandas dataframe/series containing the returns
            in desired timeframe.

    Returns:
        A float, indicating the upside potential ratio
    """
    numerator =  np.sum(np.max(0-df, 0))
    denumerator =  np.sum(np.max(df, 0))
    nu = df[df>=0].shape[0]
    nd = df[df<0].shape[0]


    return numerator * nd / denumerator / nu

def calcLPMn(df, threshold, n):
    """
    Calculates the lower partial moment (Harlow, 1991)

    Args:
        df: pd.dataframe: A pandas dataframe/series containing the returns
            in desired timeframe.
        threshold: float: The target return.
        n: int: The power to use in formula.

    Returns:
        A float, indicating the upside potential ratio
    """
    
    return np.sum(np.power(np.max(threshold-df,0),n)) / df.shape[0]


def calcKappa3Ratio(df, MAR):
    """
    Calculates kappa3 ratio (Kaplan and Knowles, 2004)

    Args:
        df: pd.dataframe: A pandas dataframe/series containing the returns
            in desired timeframe.
        MAR: float: Minimum acceptable return (e.g. 0.1 for a 10% return). 

    Returns:
        A float, indicating the upside potential ratio
    """
    LPM3 = calcLPMn(df, MAR, 3)
    kappa = (df.mean() - MAR)/np.power(LPM3,1/3)
    return kappa

def calcSterlingRatio(dfReturns, rfReturns, nd):
    """
    Calculates the Sterling ratio (McCafferty, 2003). This function uses 
    quantstats library to get the drawdowns in the dataset.

    Args:
        dfReturns: pd.Dataframe: A pandas dataframe/series containing the 
            returns.
        rfReturns: float: Average risk-free rate of return.
        nd: int: Number of the biggest draw downs to compute
    """
    # Get the drawdown dataset
    drawdowns = qs.stats.to_drawdown_series(dfReturns).sort_values()
    drawdowns = drawdowns[drawdowns!=0.0]
    
    if drawdowns.shape[0] < nd:
        drawdowns = drawdowns
        print(f"Sterling ratio - warning: Only {drawdowns.shape[0]} exists while nd = {nd}")
    else:
        drawdowns = drawdowns.iloc[:nd]
    
    if drawdowns.shape[0] == 0:
        # No drawdowns found
        print("Sterling ratio: No draw downs found in the provided dataset")
        return np.inf
    else:
        return (dfReturns.mean() - rfReturns) / abs(drawdowns.mean() )

def calcSterlingCalmarRatio(dfAuM, dfReturns, rfReturns):
    """
    Calculates the Sterling-calmar ratio. Perhaps the most common variation of
    the Sterling ratio uses the average annual maximum drawdown in the denominator
    over 3 years. A combination of both Sterling and Calmar concepts

    Args:
        dfAuM: pd.Series: A pandas series indicating portfolio size
        dfReturns: pd.Dataframe: A pandas dataframe/series containing the 
            returns.
        rfReturns: float: Average risk-free rate of return.
    """
    # Get the max drawdown
    Roll_Max = dfAuM.cummax()
    Daily_Drawdown = dfAuM/Roll_Max - 1.0
    Max_Drawdown = Daily_Drawdown.cummin()
    
    if Max_Drawdown.shape[0] == 0:
        # No drawdowns found
        print("Sterling-calmar ratio: No draw downs found in the provided dataset")
        return np.inf
    else:
        return (dfReturns.mean() - rfReturns) / abs(Max_Drawdown).max()

def calcUlcerIndex(dfReturns, n = 14):
    """
    Calculates the Ulcer index (Martin and McCann, 1987)

    Args:
        dfReturns: pd.Dataframe: A pandas dataframe/series containing AuM in 
            the required periods. Using daily timeframe is advised.
        n: int: The look back period.
    """

    if n < dfReturns.shape[0]:
        ulcerIndex = dfReturns.rolling(n, min_periods=n).apply(lambda x: (x.iloc[-1]-np.max(x))/np.max(x)*100)
        ulcerIndex = ulcerIndex.pow(2).sum()/n
        ulcerIndex = np.sqrt(ulcerIndex)
        return ulcerIndex
    else:
        print(f"Ulcer index warning: Passed period length ({n}) in shorter than inputted dataset ({dfReturns.shape[0]}).")
        return np.nan
    
def calcBurkeRatio(dfReturns, rfReturns):
    """
    Calculates the Burke ratio (Burke, 1994). This function uses 
    quantstats library to get the drawdowns in the dataset.

    Args:
        dfReturns: pd.Dataframe: A pandas dataframe/series containing the 
            returns.
        rfReturns: float: Average risk-free rate of return.
        nd: int: Number of the biggest draw downs to compute
    """
    # Get the drawdown dataset
    drawdowns = qs.stats.to_drawdown_series(dfReturns).sort_values()
    drawdowns = drawdowns[drawdowns!=0.0]
    
    if drawdowns.shape[0] == 0:
        # No drawdowns found
        print("Burke ratio: No drawdowns found in the provided dataset")
        return np.inf
    else:
        return (dfReturns.mean() - rfReturns) / np.sqrt(drawdowns.pow(2).sum())

def calcAdjustedSharpeRatio(dfReturns, rfReturns):
    """
    Calculates the adjusted Sharpe ratio (Pezier and White, 2006).

    Args:
        dfReturns: pd.Dataframe: A pandas dataframe/series containing the 
            returns.
        rfReturns: float: Average risk-free rate of return.
    """

    SR = (dfReturns.mean() - rfReturns)/np.std(dfReturns - rfReturns)
    S = scipy.stats.skew(dfReturns)
    K = scipy.stats.kurtosis(dfReturns)
    return SR*(1+S/6*SR-(K-3)/24*SR**2)

    
def calcProspectRatio(dfReturns, threshold):
    """
    Calculates the Prospect ratio (Wantable, 2014)

    Args:
        dfDD: pd.Series: A pandas series with indexes as dates. The draw
            downs are calculated using this dataframe. It is suggested that
            this dataframe have a maximum timeframe of daily entries. Using
            weekly/monthly entries may lead to null dataframes. Note that this 
            data series should contain the returns, not the AuM or NAV.
        dfReturns: pd.Dataframe: A pandas dataframe/series containing the 
            returns.
        threshold: float: Minimum acceptable return threshold.
    """
    # Get the drawdown dataset
    # Get the drawdown dataset
    drawdowns = qs.stats.to_drawdown_series(dfReturns).sort_values()
    drawdowns = drawdowns[drawdowns!=0.0]
    
    if drawdowns.shape[0] == 0:
        # No drawdowns found
        print("Prospect ratio: No drawdowns found in the provided dataset")
        return np.inf
    else:
        return (1/dfReturns.shape[0]*(np.max(dfReturns,0)+2.25*np.min(dfReturns,0))-threshold)/np.std(drawdowns)