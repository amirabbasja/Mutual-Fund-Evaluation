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

--------

The calculation and interpretation of each metric is explained in this section:

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

Also rho can be calculated as below:
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
