# Problem definition

The portfolio optimization problem involves distributing an investment over a set of assets to maximize returns while minimizing risk. The goal is to find the optimal allocation of capital across different assets to achieve the best risk-adjusted returns. This problem is often formulated as a mathematical optimization problem, where the objective is to maximize the Sharpe Ratio, which measures the excess return per unit of risk.

The Sharpe Ratio is calculated as follows: Sharpe Ratio = (Expected Return of Portfolio - Risk-Free Rate) / Standard Deviation of Portfolio Returns

# Related work
The following papers were considered for this project:
- Research on intelligent algorithms for solving portfolio problems, Hongwei Wang, Lin Huo* and Jinhao Feng, https://www.matec-conferences.org/articles/matecconf/abs/2021/05/matecconf_cscns20_09017/matecconf_cscns20_09017.html
- Z. X. Loke, S. L. Goh, G. Kendall, S. Abdullah and N. R. Sabar, "Portfolio Optimization Problem: A Taxonomic Review of Solution Methodologies," in IEEE Access, vol. 11, pp. 33100-33120, 2023, doi: 10.1109/ACCESS.2023.3263198.
- Fifty years of portfolio optimization, Ahti Salo, Michalis Doumpos, Juuso Liesiö, Constantin Zopounidis, https://www.sciencedirect.com/science/article/pii/S0377221723009827
- A Comparative Study of Hybrid Quantum and Classical Genetic Algorithms in Portfolio Optimization, Romeu Rossi Junior, José Augusto Miranda Nacif, Leonardo Antônio Mendes Souza, Marcus Henrique Soares Mendes, https://arxiv.org/abs/2604.11667
- Doering, Jana & Kizys, Renatas & Juan, Angel & Fitó-Bertran, Àngels & Polat, Onur. (2019). Metaheuristics for rich portfolio optimisation and risk management: Current state and future trends. Operations Research Perspectives. 6. 10.1016/j.orp.2019.100121. 
- Abdallah, Ameni & Bedoui, Rihab & Boubaker, Heni. (2025). Metaheuristics for Portfolio Optimization: Application of NSGAII, SPEA2, and PSO Algorithms. Risks. 13. 227. 10.3390/risks13110227. 

From these papers we concluded that while mathematical optimization algorithms like Markowitz's always finds the optimal solution because it fails to scale up with the number of stocks machine learning and metaheuristics are preferred. Classical metaheuristics or hybrid ones seem to dominate the field, with PSO and GA being the most popular ones.
Based on these considerations we decided to implement the following algorithms:

# Algorithms implemented
- Memetic Genetic Algorithm (HGA)
- Particle Swarm Optimization (PSO)
- Sine Cosine Algorithm hybridized with Beetle Antennae Search (SCA-BAS)

# Parameter setting

For our experiments we decided to implement three versions for each algorithm, a fast one, a standard one and an aggressive one. 
They have the following characteristics:
- the fast one has a smaller population and fewer iterations, which results in a faster runtime but also it tries to find a good solution faster
- the standard one has a larger population and more iterations and is supposed to balance out runtime and performance. They are based on the setting found in the reference papers.
- the aggressive one has a much larger population and more iterations, which results in a longer runtime but also it tries to find the best solution possible.

# The results

| Experiment Name                     | Algorithm Settings                                   | Sharpe Ratio performance in Train | Sharpe Ratio performance in Test | Evolution of performance (Test vs Train) | Runtime |
|-------------------------------------|------------------------------------------------------|-----------------------------------|----------------------------------|------------------------------------------|---------|
| Benchmark Markowitz (on train data) | Mathematical Optimization                            | 1.8847                            | 1.4583                           | -0.4264                                  | 0.6046s |
| Benchmark Markowitz (on test data)  | Mathematical Optimization                            |                                   | 2.8702                           |                                          |         |
| HGA Fast                            | Population:30 Generations:50 HC_Probability:0.2      | 1.7101                            | 1.4161                           | -0.294                                   | 0.0235s |
| HGA Standard                        | Population:50 Generations:100 HC_Probability:0.3     | 1.8248                            | 1.4576                           | -0.3671                                  | 0.1347s |
| HGA Aggressive                      | Population:60 Generations:150 HC_Probability:0.5     | 1.8654                            | 1.4522                           | -0.4132                                  | 0.3089s |
| PSO Fast                            | Pop:20 Iter:50 w:0.7 c1:1.2                          | 1.8847                            | 1.4583                           | -0.4264                                  | 0.0259s |
| PSO Standard                        | Pop:40 Iter:100 w:0.7298 c1:1.4962                   | 1.8847                            | 1.4583                           | -0.4264                                  | 0.0899s |
| PSO Aggressive                      | Pop:70 Iter:200 w:0.8 c1:2.0                         | 1.8847                            | 1.4583                           | -0.4264                                  | 0.1846s |
| SCA-BAS Fast                        | Agents:15   Iterations:60   Sensing:0.15   Step:0.15 | 1.801                             | 1.5069                           | -0.2941                                  | 0.0722s |
| SCA-BAS Standard                    | Agents:25   Iterations:100   Sensing:0.1   Step:0.1  | 1.8116                            | 1.4568                           | -0.3548                                  | 0.2261s |
| SCA-BAS Aggressive                  | Agents:40   Iterations:150   Sensing:0.2   Step:0.2  | 1.8841                            | 1.4545                           | -0.4296                                  | 0.4129s |

# Discussion of the results

The results show that the algorithms performed as expected (except for PSO which always found the same solution as Markowitz), with the aggressive versions performing better than the standard ones, which in turn performed better than the fast ones.
Also the fundamental problem of PO is present where that best solution found on the historical data doesn't translate into the best solution on the test dataset with one of the algorithms actually finding a better solution on the test dataset than Markowitz.

# Conclusion

The stockmarket is a complex system and is very volatile, which makes it very difficult to predict. The best solution found so far is using historical data to find the best allocation of capital across different assets, but this solution is not guaranteed to perform well in the future. The algorithms implemented in this project are able to find good solutions on the historical data, but they are not able to find the best solution on the test dataset. This is a common problem in portfolio optimization and it is still an open research question. Future work could add more restriction on the portfolio to try to increase the performance on the test dataset.