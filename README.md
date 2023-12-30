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

### 19. Turnover rate
The turnover rate of a fund is an indicator that reflects the frequency of stock trading in a fund. It is used to measure the frequency of changes in the fund's investment portfolio and the average length of time a fund manager holds stocks. It is an important indicator for evaluating the investment style of a fund manager.

studies have found that compared to passive index funds, active equity funds have a relatively high turnover rate, which can bring certain transaction impact costs and drag on fund performance. Under the same yield, the lower the turnover rate of the fund, the better the fund is.

The metric is calculated below:

![alt text](https://latex.codecogs.com/svg.image?Turnover\;rate=\frac{MAX(Total\;cost\;of\;buying\;stocks,Total\;revenue\;from\;selling\;stocks)}{Fund's\;AuM\;in\;the\;required\;period})

### 20. Average holding time
The longer the holding period of a fund manager, the higher the comprehensive return on fund performance. In addition, fund managers with long-term holdings hold a more concentrated investment portfolio and a lower turnover rate, which allows them to have more time and energy to collect information about the companies they invest in, thereby supervising the opportunistic behavior of the company's management and effectively ensuring the improvement of the company's value.

The metric is calculated below:

![alt text](https://latex.codecogs.com/svg.image?Average%5C;holding%5C;time%5C;=%5C;%5Cfrac%7BInterval%5C;duration%7D%7BTurnover%5C;rate%7D)
