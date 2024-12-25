import numpy as np
from .config import eps

def sharpe(returns, freq=252, rfr=0.0):
    """Given a set of returns, calculates naive (rfr=0) sharpe (eq 28) """
    return (np.sqrt(freq) * np.mean(returns - rfr)) / (np.std(returns - rfr) + eps)

def sortino(returns, freq=252, rfr=0.0):
    df = returns.copy()
    downside_returns = df[df < rfr]
    expected_return = returns.mean()
    downside_stdev = np.std(downside_returns - rfr)
    sortino_ratio = ((expected_return - rfr) * np.sqrt(freq)) / (downside_stdev + eps)
    return sortino_ratio

def MDD(X):
    """By nicktids, see issue 15."""
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return mdd


def softmax(w, t=1.0):
    """softmax implemented in numpy."""
    log_eps = np.log(eps)
    w = np.clip(w, log_eps, -log_eps)  # avoid inf/nan
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

def relu(w, t=1.0):
    """softmax implemented in numpy."""
    log_eps = np.log(eps)
    w = np.clip(w, log_eps, -log_eps)  # avoid inf/nan
    dist = (abs(w) + w) / 2
    return dist

def MDD1(X):
    dd = 1 - X/X.cummax()
    mdd = max(dd)  #输入为 df.portfolio_value
    return mdd

def calmar(returns, freq=252, rfr=0.0):
    # MDD function:
    prices = np.exp(np.cumsum(returns))
    mdd = 0
    peak = prices[0]
    for x in prices:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    # get calmar:
    calmar_ratio = (freq * np.mean(returns - rfr)) / mdd
    return calmar_ratio

def other_metrics(returns):
    # annual returns:
    print("平均年化收益率（annual average return）：", np.mean(returns)*252)
    print("年化波动率（annual variance）：", np.std(returns)*np.sqrt(252))
    print("投资胜率（percentage of positive trade returns）：", len(returns[returns>0])/len(returns))
    print("平均益损比（ratio between positive and negative returns）:", \
    returns[returns>0].mean()/np.abs(returns[returns<0].mean()))
    return None
