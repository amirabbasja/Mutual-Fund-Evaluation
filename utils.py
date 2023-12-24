import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import calendar
import quantstats as qs

# Necessary methods
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
            tmpMonth.append(datetime.datetime(year=int(row.year),month=i,day=1)+relativedelta(months=1))
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

def loadRiskFreeReturn(path):
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
    return df

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