# Problem definition

The portfolio optimization problem involves distributing an investment over a set of assets to maximize returns while minimizing risk. The goal is to find the optimal allocation of capital across different assets to achieve the best risk-adjusted returns. This problem is often formulated as a mathematical optimization problem, where the objective is to maximize the Sharpe Ratio, which measures the excess return per unit of risk.

The Sharpe Ratio is calculated as follows: Sharpe Ratio = (Expected Return of Portfolio - Risk-Free Rate) / Standard Deviation of Portfolio Returns


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


