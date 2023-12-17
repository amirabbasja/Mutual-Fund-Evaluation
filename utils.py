import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import calendar

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

def getMonthlyReturns(fundAuM):
    """
    Returns a dataframe containing monthly returns of the fund.

    Args:
        fundAuM: pd.Dataframe: A dataframe containing AuM of the fund at the end 
            of each day. SHould contain columns named ["Date", "AuM"]
    """
    endDate = fundAuM.iloc[-1].Date 
    startMonth = fundAuM.iloc[0].Date.month
    startYear = fundAuM.iloc[0].Date.year

    rangeStart = datetime.datetime(year = startYear, month=startMonth, day=1)
    rangeEnd = rangeStart
    monthReturn, monthStart, monthEnd = [], [], []

    while True:
        # Setting the difference between start and end of each month as its return
        rangeEnd += relativedelta(months=1)

        # Get the dataframe for each month
        tmpDataframe = fundAuM[(rangeStart <= fundAuM.Date)&(fundAuM.Date < rangeEnd)]

        # Calculate the return
        monthReturn.append((tmpDataframe.iloc[-1].AuM-tmpDataframe.iloc[0].AuM)/tmpDataframe.iloc[0].AuM)
        monthStart.append(rangeStart)
        monthEnd.append(rangeStart + relativedelta(months=1))

        if endDate <= rangeEnd:
            break
        rangeStart += relativedelta(months=1)

    fundMonthlyReturns = pd.DataFrame(list(zip(monthEnd,monthReturn)), columns=["Date", "fundReturn"])
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

def calcCaptureRate(df):
    """
    Calculates the capture rate of fund relative to benchmark.

    Args: 
        df: pd.Dataframe: A dataframe containing two columns. First the 
            fund returns and the second, the benchmark returns.
    """
    numerator, denumerator = 1, 1
    numerator = (df.iloc[:,0]+1).product(axis = 0)
    denumerator = (df.iloc[:,1]+1).product(axis = 0)
    
    return (numerator - 1) / (denumerator - 1)

def netGrowthRate(AuMLastDay, AuMFirstDay, annualized):
    """
    Returns the net growth rate of the fund's AuM 
    The formula: (Last day value - First day value)/First day value = growth rate

    Args:
        AuMLastDay, AuMFirstDay: float: AuM of the fund at the beginning and end of the interval
    
    Returns: 
        float.
    """
    if annualized:
        return np.power((AuMLastDay-AuMFirstDay)/AuMFirstDay + 1, 1/3) - 1
    else:
        return (AuMLastDay-AuMFirstDay)/AuMFirstDay