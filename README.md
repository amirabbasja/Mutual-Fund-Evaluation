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
