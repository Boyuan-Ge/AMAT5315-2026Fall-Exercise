# Week 1: Monte Carlo π Estimator Specification

`estimate_pi(n, seed)` 会通过在单位正方形里随机生成点，统计落在四分之一圆内的比例，再乘以 4 来估算 π。

`seed` 的作用是固定随机数生成过程，让测试可以重复，保证相同输入得到相同结果。

正确性的判断标准是：

```python
abs(estimate_pi(1_000_000, seed=2026) - math.pi) < 1e-2
```
