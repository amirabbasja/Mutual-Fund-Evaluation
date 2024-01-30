# Mutual-Fund-Evaluation

This repository contains the evaluations for multiple funds using various metrics.
This repo is an ongoing project.

Metrics used in this repository:

1. Net Growth Rate of the funds
2. Up/Down Capture Ratio
3. Win Ratio
4. Jensen's alpha
5. Calmar Ratio
6. Treynor Ratio
7. Sortino Ratio
8. Sharpe Ratio
9. Information Ratio
10. Standard deviation of returns
11. Downside deviation
12. Max drawdown
13. Tracking error
14. Alpha and beta of the fund
15. MPPM (Manipulation-Proof Performance Measure)
16. CRP (Cross Product Ratio)
17. Hit rate
18. Payoff ratio
19. Turnover rate
20. Average holding time
21. VaR and CVaR (Normal and t-distribution)
22. Excess return on VaR (Generalized Sharpe ratio)
22. Conditional Sharpe ratio

--------

The calculation and interpretation of each metric is explained in this section:

### 1. Net Growth Rate of the funds

Fund return, also known as the net growth rate of unit net worth is the most intuitive and clear evaluation indicator.

The formula can be calculated as follows:

![alt text](https://latex.codecogs.com/svg.image?Growth%5C;rate=%5Cfrac%7BUnit%5C;AuM%5C;of%5C;the%5C;last%5C;day%5C;of%5C;interval-Unit%5C;AuM%5C;of%5C;the%5C;first%5C;day%5C;of%5C;interval%7D%7BUnit%5C;AuM%5C;of%5C;the%5C;first%5C;day%5C;of%5C;interval%7D)

### 2. Up/Down capture rate

The up-capture return represents the ratio of fund return to market index return when the market rises. The higher the upward capture rate, the stronger the fund's ability to keep up with the market. The down-capture return represents the ratio of fund returns to market index returns when the market falls. The smaller the downward capture rate, the stronger the fund's resilience to falls.

The formula can be calculated as follows:

![alt text](https://latex.codecogs.com/svg.image?Up%5C;capture=%5Cfrac%7B%5Cprod_%7Bk=1%7D%5E%7Bn_%7Bup%7D%7D(1&plus;r_%7Bk%7D)-1%7D%7B%5Cprod_%7Bj=1%7D%5E%7Bn_%7Bup%7D%7D(1&plus;r_%7Bmj%7D)-1%7D)

![alt text](https://latex.codecogs.com/svg.image?Down%5C;capture=%5Cfrac%7B%5Cprod_%7Bk=1%7D%5E%7Bn_%7Bdown%7D%7D(1&plus;r_%7Bk%7D)-1%7D%7B%5Cprod_%7Bj=1%7D%5E%7Bn_%7Bdown%7D%7D(1&plus;r_%7Bmj%7D)-1%7D)

Where ![alt text](https://latex.codecogs.com/svg.image?r_%7Bk%7D) represents the k-th return of the fund during the benchmark rise period  ![alt text](https://latex.codecogs.com/svg.image?r_%7Bmj%7D) represents the j-th yield of the benchmark during the period of benchmark rise.

### 3. Win ratio

Win rate refers to the probability of a fund being purchased at any time and held for a certain period of time before making a profit. For example, if a fund has been running for a year and has made profits in 9 months, the investment success rate of the fund is 75%.

### 4. Jensen's alpha

The Jensen's measure, or Jensen's alpha, is a risk-adjusted performance measure that represents the average return on a portfolio or investment, above or below that predicted by the capital asset pricing model (CAPM), given the portfolio's or investment's beta and the average market return. This metric is also commonly referred to as simply alpha.

The formula can be calculated as follows:

![alt text](https://latex.codecogs.com/svg.image?%5Calpha=R_%7Bi%7D-(R_%7Bf%7D&plus;%5Cbeta(R_%7Bm%7D-R_%7Bf%7D)))

### 5. Calmar Ratio

The Calmar ratio is a gauge of the performance of fund. It is a function of the fund's average compounded annual rate of return versus its maximum drawdown. The higher the Calmar ratio, the better it performed on a risk-adjusted basis during the given time, which is mostly commonly set at 36 months. One strength of the Calmar ratio is its use of the maximum drawdown as a measure of risk. For one thing, it's more understandable than other, more abstract risk gauges, and this makes it preferable for some investors. In addition, even though it is updated monthly, the Calmar ratio's standard three-year time frame makes it more reliable than other gauges with shorter time frames that might be more affected by natural market volatility. On the flip side, the Calmar ratio's focus on drawdown means it's view of risk is rather limited compared to other gauges, and it ignores general volatility. This makes it less statistically significant and useful.

The formula can be calculated as follows:

![alt text](https://latex.codecogs.com/svg.image?Calmar%5C;ratio=%5Cfrac%7BAnnual%5C;rate%5C;of%5C;return%5C;of%5C;portfolio%7D%7BMaximum%5C;drawdown%5C;in%5C;the%5C;period%7D)

### 6. Treynor Ratio

The Treynor Ratio calculates a portfolio's excess returns as a percentage of its systematic risk, which is denoted by beta. This metric is helpful for determining how much more than a risk-free rate the fund has generated for each unit of market risk it has taken (Treynor, 1965).

The formula can be calculated as follows:

![alt text](https://latex.codecogs.com/svg.image?Sortino%5C;ratio=%5Cfrac%7BPortfolio%5C;return-Risk%5C;free%5C;return%7D%7B%5Cbeta%7D)

### 7. Sortino ratio

It is similar to the Sharpe ratio, except that it does not use standard deviation as the standard, but uses downside deviation, which is the degree to which the investment portfolio deviates from its average decline, to distinguish between good and bad volatility. The higher the ratio, the higher the excess return that the fund can achieve by taking on the same unit of downside risk.

![alt text](https://latex.codecogs.com/svg.image?Sortino%5C;ratio=%5Cfrac%7BExcess%5C;return%20of%5C;portfolio%7D%7BDownside%5C;risk%7D=%5Cfrac%7BAverage%5C;return%5C;in%5C;the%5C;desired%5C;period-Risk%5C;free%5C;return%7D%7BDownside%5C;risk%7D)

### 8. Sharpe Ratio

In 1966, American economist William Sharp proposed an indicator that comprehensively considers returns and risks, using standard deviation to measure the total risk borne by a fund and evaluating fund performance at a premium per unit of
total risk. The Sharpe index represents how much excess return will be generated for each unit of risk taken.

* When the Sharpe index is positive, it indicates that the fund's return rate is higher than the volatility risk. The larger the Sharpe index, the higher the fund's unit risk premium and performance.
* When the Sharpe index is negative, it indicates that the fund's operational risk is greater than the fund's return rate. The smaller the Sharpe index, the smaller the fund's unit risk premium, The poorer the performance of the fund

The ratio is calculated a follows:

![alt text](https://latex.codecogs.com/svg.image?Sharpe%5C;ratio=%5Cfrac%7BR_%7Bp%7D-R_%7Bf%7D%7D%7B%5Csigma%20_%7Bp%7D%7D)

Where:

* ![alt text](https://latex.codecogs.com/svg.image?R_%7Bp%7D) is the portfolio return
* ![alt text](https://latex.codecogs.com/svg.image?R_%7Bf%7D) is the risk-free return
* ![alt text](https://latex.codecogs.com/svg.image?%5Csigma_%7Bp%7D) standard deviation of the portfolio's excess return

### 9. Information Ratio

The information ratio characterizes the excess returns brought by unit active risk. From the perspective of active management, the risk adjusted portfolio returns are examined. The larger the information ratio, the higher the excess returns obtained by fund managers' unit tracking errors. The calculation formula for information ratio is as follows:

![alt text](https://latex.codecogs.com/svg.image?Information%5C;ratio%5C;=%5Cfrac%7BExcess%5C;return%5C;of%20the%5C;protfolio%7D%7BStandard%5C;deviation%5C;of%5C;exces%5C;return%7D)

### 10. Standard deviation of returns

The standard deviation of return measures the degree of deviation between the daily return of a fund and the average return. It is used to measure the volatility of fund returns. The larger the standard deviation of a fund, the greater the corresponding risk

### 11. Downside deviation

The downward standard deviation is an improvement on the traditional standard deviation indicator. The downward standard deviation establishes a certain critical or minimum value, and only calculates the "bad" observations below that critical value. Observations above the cut-off value are considered 'good' and will be ignored. The downward deviation refers to the risk of a decline in securities returns when the market environment deteriorates, reflecting the magnitude of the decline in returns when the index goes down. The larger the decline, the higher the risk is

![alt text](https://latex.codecogs.com/svg.image?E(%5C!r_%7Bi%7D)=r_%7Bf%7D&plus;%5Cbeta%20_%7Bim%7D(E(%5C!r_%7Bm%7D-r_%7Bm%7D)))

Where ![alt text](https://latex.codecogs.com/svg.image?r_%7Bi%7D) is the yield of the i - th period of the fund, ![alt text](https://latex.codecogs.com/svg.image?r_%7Bt%7D) is the target yield, and 𝑛 represents the number of periods where the fund yield is less than the target yield

### 12. Maximum drawdown

The maximum drawdown is used to describe the worst-case scenario that may occur after buying a fund product. It is calculated from any historical point in a selected period, and the return rate when the product's net value drops to the lowest point, that is, the maximum pullback amplitude. It is an important risk indicator for investment portfolios and reflects the maximum loss that may be faced in the investment.

### 13. Tracking error

Tracking Error refers to the standard deviation of the return between the portfolio return and the benchmark return, which is an important indicator of the deviation between fund returns and target index returns. The larger the value, the greater the risk of active investment by the fund manager, and a tracking error of more than 2% indicates a significant difference.

### 14. Alpha and beta of the fund

The definition of Beta originated from the classic financial theory proposed by William Sharp, John Lintner, Jack Treynor, and Jan Mossin in 1964- the Capital Asset Pricing Model. The purpose of this model is to calculate the reasonable return on an investment product or portfolio, which is simply to calculate the return on assets. The formula for the capital asset pricing model is:

![alt text](https://latex.codecogs.com/svg.image?E(%5C!r_%7Bi%7D)=r_%7Bf%7D&plus;%5Cbeta%20_%7Bim%7D(E(%5C!r_%7Bm%7D-r_%7Bm%7D)))

Where E(ri) represents the expected return rate of asset r, rf is the risk-free interest rate, and the risk-free interest rate represents the time value of the asset. ![alt text](https://latex.codecogs.com/svg.image?%5Cbeta%20_%7Bim%7D) can be derived from the above formula. Therefore, ![alt text](https://latex.codecogs.com/svg.image?%5Cbeta%20_%7Bim%7D) actually compares the ratio between the expected excess return of asset i relative to the risk-free interest rate and the expected excess return of bearing market risk, reflecting the sensitivity of asset i's price to overall market fluctuations Therefore ![alt text](https://latex.codecogs.com/svg.image?%5Cbeta%20_%7Bim%7D) is also known as risk coefficient.

* If ![alt text](https://latex.codecogs.com/svg.image?%5Cbeta%20_%7Bim%7D) > 1, it indicates that the expected return volatility of asset 𝑖 is greater than the overall market.
* if ![alt text](https://latex.codecogs.com/svg.image?%5Cbeta%20_%7Bim%7D) < 1, it indicates that the expected return volatility of asset 𝑖 is lower than the overall market.
* if ![alt text](https://latex.codecogs.com/svg.image?%5Cbeta%20_%7Bim%7D) = 1, it indicates that the volatility of expected returns on asset 𝑖 is the same as the overall market.

The capital asset model actually calculates the theoretical expected return of the investment portfolio, as it assumes that investors are rational and that the capital market is a completely efficient market without any friction hindering investment. The difference between the actual expected return and the  theoretical expected return is called 𝛼, which represents the portion of the investment that exceeds the market or benchmark return, also known as excess return. The calculation formula for ![alt text](https://latex.codecogs.com/svg.image?%5Calpha%20) is:

![alt text](https://latex.codecogs.com/svg.image?%5Calpha=E(r_%7Bi%7D)-r_%7Bf%7D&plus;%5Cbeta%20_%7Bim%7D(E(r_%7Bm%7D)-r_%7Bf%7D))

![alt text](https://latex.codecogs.com/svg.image?%5Calpha%20) may not always be a positive number. When ![alt text](https://latex.codecogs.com/svg.image?%5Calpha%20) is less than 0, it indicates that the active management strategy of a fund manager is not successful. Therefore, when judging a fund manager's historical performance, we cannot only evaluate it based on the absolute return it has obtained. In an upward market, a fund manager with a high beta but a negative ![alt text](https://latex.codecogs.com/svg.image?%5Calpha%20) may also periodically obtain a seemingly good absolute return, but his active management ability may not be outstanding, When the market falls, it will also fall more, so finding a positive ![alt text](https://latex.codecogs.com/svg.image?%5Calpha%20) is the goal that every fund manager and fund investor has been striving for

### 15. Manipulation-Proof Performance Measure (MPPM)

MPPM is used to assess the performance of an investment fund. It compares the fund's returns with both the broader market and a "risk-free rate". It also compares the fund's performance to that of the market, not merely to a risk-free rate. This enables you to determine whether the fund is actually doing well or is simply being fortunate.

The metric is calculated below:

![alt text](https://latex.codecogs.com/svg.image?MPPM%5C;=%5Cfrac%7B1%7D%7B1-%5Crho%7DLn%5Cleft(%5Cfrac%7B1%7D%7BT%7D%5Csum_%7BT%7D%5E%7Bt=1%7D%5Cfrac%7B1&plus;rt%7D%7B1&plus;rft%7D%5Cright))

Where:

* ρ: This parameter is crucial for the measure. It adjusts the sensitivity of the MPPM to the returns' distribution tails. A different value of ρ will give different weight to higher or lower returns Eq (2).
* t: This represents a specific month within the period of observation. In the context of the study, t varies from 1 to T, where T is twelve months.
* rt: This is the monthly return of the crypto fund for month t.
* rft: This is the monthly risk-free rate for month t. This is a standard measure used in finance to represent the return on an investment that is considered risk free, such as a short-term government bond. In this case, it is sourced from the DFO data file.
* T: This represents the total number of months being considered in the study, which is twelve months.

Also, rho can be calculated as below:

![alt text](https://latex.codecogs.com/svg.image?%5Crho=%5Cfrac%7BLn(E(1&plus;rb))-Ln(E(1&plus;rft))%7D%7BVar(Ln(1&plus;rb))%7D)

Where:

* ρ: This symbol typically represents a parameter or coefficient. In the context of the previous MPPM formula, ρ adjusts the sensitivity of the MPPM to the returns' distribution tails. In this equation, it's being defined in relation to expected returns, the risk-free rate, and the variability (variance) of returns.
* E(1+rb): This represents the expected value (or mean) of the quantity 1+rb. Here, E denotes the expectation operator, which, in finance, typically means taking the average expected outcome for a random variable.
* rb: This is the return of the crypto fund (or benchmark return) for a given period. In financial analyses, rb often stands for the return of a benchmark against which other returns are compared.
* rf: This is the risk-free rate. In finance, the risk-free rate represents the return on an investment that's considered devoid of risk, such as a short-term government bond. It's a foundational concept in modern finance, acting as a baseline against which other returns are evaluated.
* Var[(1+rb)]: This represents the variance of the quantity 1+rb. Variance is a statistical measure that captures the dispersion or spread of a set of data points. In the context of returns, it provides a measure of the risk or volatility of the investment. The higher the variance, the more spread out or volatile the returns are.

### 16. Cross product ratio (CRP)

The Cross Product Ratio (CPR) method is used to analyze the persistence in a fund's performance. Persistence in performance means that a fund which performed well (or poorly) in one period is likely to continue that trend in subsequent periods. The CPR method quantifies this persistence by looking at the ratio of funds that maintain their performance (either good or bad) to those that switch their performance. In this method the entire sample period is divide into equal intervals (e.g. 1 month). For each interval, if the fund/manager had a return in the top 50% are classified as winners (W), on the other hand, if the fund's return is in the low 50%, it will be classified as a loser in the said interval (L). If the fund/manager is a winner/loser in two consecutive intervals, it will be known as double winner/loser in that interval (WW/LL). On the other hand, if the manager has won in current interval and lost the prevous interval, it will be counted as winner-loser (WL) in that interval (The similar logic goes for loser-winner intervals - LW).

The metric is calculated below:

![alt text](https://latex.codecogs.com/svg.image?CRP%5C;=%5Cfrac%7BWW&plus;LL%7D%7BWL&plus;LW%7D)

### 17. Hit rate

The Hit Rate represents the percentage of trades that are profitable over a specified period. It provides insights into the consistency of a trading strategy.

* A Hit Rate of 50% means that 50% of all trades were profitable, and it's equivalent to random chance.
* A Hit Rate greater than 50% suggests that more than half the trades were profitable, indicating a potentially effective trading strategy.
* Conversely, a Hit Rate less than 50% indicates that less than half of the trades were profitable.

The metric is calculated below:

![alt text](https://latex.codecogs.com/svg.image?Hit%5C;rate%5C;=%5Cfrac%7BNumber%5C;of%5C;losing%5C;trades%7D%7BNumber%5C;of%5C;losing%5C;trades%7D)

### 18. Payoff ratio

The Payoff Ratio, also known as the Profit/Loss Ratio, measures the relationship between the average profit from winning trades and the average loss from losing trades. It essentially gives an idea about the reward-to-risk ratio of a trading strategy.

* A Payoff Ratio greater than 1 indicates that the average profit from winning trades is greater than the average loss from losing trades.
* A Payoff Ratio less than 1 suggests the opposite: that losses are on average bigger than wins.
* A Payoff Ratio equal to 1 means that the average profit and average loss are the same.

The metric is calculated below:

![alt text](https://latex.codecogs.com/svg.image?Payoff%5C;ratio%5C;=%5C;%5Cfrac%7BAerage%5C;profit%5C;per%5C;winning%5C;trade%7D%7BAverage%5C;Loss%5C;per%5C;losing%5C;trade%7D)

### 19. Turnover rate

The turnover rate of a fund is an indicator that reflects the frequency of stock trading in a fund. It is used to measure the frequency of changes in the fund's investment portfolio and the average length of time a fund manager holds stocks. It is an important indicator for evaluating the investment style of a fund manager.

studies have found that compared to passive index funds, active equity funds have a relatively high turnover rate, which can bring certain transaction impact costs and drag on fund performance. Under the same yield, the lower the turnover rate of the fund, the better the fund is.

The metric is calculated below:

![alt text](https://latex.codecogs.com/svg.image?Turnover%5C;rate%5C;=%5C;%5Cfrac%7BMAX(Total%5C;cost%5C;of%5C;buying%5C;stocks,Totla%5C;revenue%5C;from%5C;selling%5C;stocks)%7D%7BAVG(Fund's%5C;AuM%5C;in%5C;the%5C;required%5C;period)%7D)

### 20. Average holding time

The longer the holding period of a fund manager, the higher the comprehensive return on fund performance. In addition, fund managers with long-term holdings hold a more concentrated investment portfolio and a lower turnover rate, which allows them to have more time and energy to collect information about the companies they invest in, thereby supervising the opportunistic behavior of the company's management and effectively ensuring the improvement of the company's value.

The metric is calculated below:

![alt text](https://latex.codecogs.com/svg.image?Average%5C;holding%5C;time%5C;=%5C;%5Cfrac%7BInterval%5C;duration%7D%7BTurnover%5C;rate%7D)

### 21. VAR and CVAR

Value at Risk (VaR) and Conditional Value at Risk (CVaR) are both risk measures used to quantify the potential loss of an investment portfolio or financial instrument. However, they differ in their approach to measuring risk. Value at Risk (VaR) is a statistical measure that estimates the maximum potential loss that a portfolio or financial instrument will incur with a specified probability over a given time horizon. For example, a VaR of 1% at a one-day horizon means that there is a 1% chance that the portfolio will lose more than 1% of its value over the next day. Conditional Value at Risk (CVaR), also known as Expected Shortfall (ES), is a measure of the average loss that will occur if the VaR is breached. In other words, it measures the expected loss beyond the VaR threshold.

Both parametric and historical methods for VaR and CVaR calculations are provided in this library. Both methods are explained briefly below:

* Historical method: In this method, we assume that future returns will follow a similar distribution to historical returns.
* Parametric method:  The parametric method looks at the price movements of investments over a look-back period and uses probability theory to compute a portfolio's maximum loss. This method for VaR calculates the standard deviation of price movements of an investment or security. Assuming stock price returns and volatility follow a normal distribution, the maximum loss within the specified confidence level is calculated. In this library, Student's t-distribution is provided as an alternative to the common normal distribution as well.

### 22. Excess return on VaR (Generalized Sharpe ratio)

It is superior to the standard Sharpe ratio because it is valid regardless of the correlations of the investments being considered with the rest of our portfolio. Some illustrative numerical examples also suggest that generalized and traditional Sharpe rules can generate very different required returns, and hence lead to very different decisions.

The metric is calculated below:

![alt text](https://latex.codecogs.com/svg.image?%20Generalized%5C;Sharpe%5C;ratio%5C;=%5Cfrac%7BR_%7Bp%7D-R_%7Bf%7D%7D%7BVaR%7D)

Where VaR is the value at risk of returns with a predefined confidence level. For more information, refer to Dowd (2000).

### 23. Conditional Sharpe ratio

Conditional Sharpe ratio replaces VaR with conditional VaR in the denominator of the reward to VaR ratio. Clearly, if expected shortfall is the major concern of the investor then the conditional Sharpe ratio is demonstrably favorable to the reward to VaR ratio:

The metric is calculated below:

![alt text](https://latex.codecogs.com/svg.image?%20Conditional%5C;Sharpe%5C;ratio%5C;=%5Cfrac%7BR_%7Bp%7D-R_%7Bf%7D%7D%7BCVaR%7D)

Where CVaR is the conditional value at risk of returns with a predefined confidence level. For more information, refer to Agarwal (2003).
