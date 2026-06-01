"""
A股机器学习预测策略 - 最小可行原型
=====================================
用随机森林模型预测股票次日涨跌，并进行简单回测。

使用方法：
    python stock_predictor.py

修改配置：
    改下面「配置区」的 STOCK_CODE 可以换股票
    改 STOCK_NAME 可以换个好认的名字
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import akshare as ak

# ============================================================
# 配置区 —— 新手只需要改这里
# ============================================================
STOCK_CODE = "000300"   # 股票/指数代码（000300=沪深300ETF）
STOCK_NAME = "沪深300"   # 显示用的名字
PERIOD = "daily"         # 日线数据
START_DATE = "20200101"  # 数据起始日期
TRAIN_RATIO = 0.8        # 训练集比例（0.8 = 前80%训练，后20%测试）

# ============================================================
# 第1步：下载数据
# ============================================================
print("=" * 60)
print(f"  正在下载 {STOCK_NAME}({STOCK_CODE}) 的历史数据...")
print("=" * 60)

try:
    # akshare 获取A股指数日线数据
    df = ak.stock_zh_index_daily(symbol=f"sh{STOCK_CODE}")
    df.rename(columns={
        "date": "日期",
        "open": "开盘",
        "high": "最高",
        "low": "最低",
        "close": "收盘",
        "volume": "成交量",
    }, inplace=True)
    print(f"  数据下载成功！共 {len(df)} 个交易日")
    print(f"  日期范围：{df['日期'].iloc[0]} 到 {df['日期'].iloc[-1]}")
except Exception as e:
    print(f"  指数数据获取失败 ({e})，尝试获取个股数据...")
    try:
        # 如果指数获取失败，尝试获取个股
        df = ak.stock_zh_a_hist(symbol=STOCK_CODE, period=PERIOD,
                                start_date=START_DATE, adjust="qfq")
        df.rename(columns={
            "日期": "日期",
            "开盘": "开盘",
            "最高": "最高",
            "最低": "最低",
            "收盘": "收盘",
            "成交量": "成交量",
        }, inplace=True)
        print(f"  个股数据下载成功！共 {len(df)} 个交易日")
    except Exception as e2:
        print(f"  数据下载失败：{e2}")
        print("  网络问题或akshare版本过旧，请稍后重试")
        exit(1)

# 如果列名是英文的，统一映射
if "收盘" not in df.columns:
    col_map = {"date": "日期", "open": "开盘", "high": "最高",
               "low": "最低", "close": "收盘", "volume": "成交量"}
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns},
              inplace=True)

# 确保日期列是 datetime 类型
df["日期"] = pd.to_datetime(df["日期"])
df.sort_values("日期", inplace=True)
df.reset_index(drop=True, inplace=True)

# ============================================================
# 第2步：特征工程
# ============================================================
print("\n" + "=" * 60)
print("  正在计算技术指标（特征工程）...")
print("=" * 60)

# --- 价格特征 ---
df["收益率_1日"] = df["收盘"].pct_change(1)            # 今天涨跌幅
df["收益率_5日"] = df["收盘"].pct_change(5)            # 过去5天涨跌幅
df["收益率_20日"] = df["收盘"].pct_change(20)          # 过去20天涨跌幅

# --- 均线偏离度（价格相对于均线是高还是低）---
for window in [5, 10, 20, 60]:
    ma = df["收盘"].rolling(window).mean()
    df[f"偏离度_MA{window}"] = df["收盘"] / ma - 1     # >0 表示在均线上方

# --- 成交量特征 ---
df["成交量_5日均值"] = df["成交量"].rolling(5).mean()
df["成交量变化_5日"] = df["成交量"] / df["成交量_5日均值"] - 1
df["成交量变化_20日"] = df["成交量"] / df["成交量"].rolling(20).mean() - 1

# --- RSI（相对强弱指标）---
# RSI = 100 - (100 / (1 + 平均涨幅 / 平均跌幅))
delta = df["收盘"].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["RSI_14"] = 100 - (100 / (1 + rs))

# --- MACD ---
ema12 = df["收盘"].ewm(span=12, adjust=False).mean()
ema26 = df["收盘"].ewm(span=26, adjust=False).mean()
df["MACD_DIF"] = ema12 - ema26
df["MACD_DEA"] = df["MACD_DIF"].ewm(span=9, adjust=False).mean()
df["MACD_HIST"] = 2 * (df["MACD_DIF"] - df["MACD_DEA"])  # MACD柱

# --- 布林带位置 ---
df["BB_MID"] = df["收盘"].rolling(20).mean()
bb_std = df["收盘"].rolling(20).std()
df["BB_upper"] = df["BB_MID"] + 2 * bb_std
df["BB_lower"] = df["BB_MID"] - 2 * bb_std
df["BB_position"] = (df["收盘"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
# BB_position: 0=下轨, 0.5=中轨, 1=上轨

# --- 波动率 ---
df["波动率_20日"] = df["收益率_1日"].rolling(20).std()

# --- 振幅 ---
df["振幅"] = (df["最高"] - df["最低"]) / df["收盘"].shift(1)

# --- 目标变量：明天涨了吗？---
df["明日涨跌"] = (df["收盘"].shift(-1) > df["收盘"]).astype(int)

# 删除包含 NaN 的行（滚动计算初期会产生空值）
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"  特征计算完成！清洗后共 {len(df)} 行数据")

# ============================================================
# 第3步：准备训练数据
# ============================================================
# 特征列清单
FEATURES = [
    "收益率_1日", "收益率_5日", "收益率_20日",
    "偏离度_MA5", "偏离度_MA10", "偏离度_MA20", "偏离度_MA60",
    "成交量变化_5日", "成交量变化_20日",
    "RSI_14",
    "MACD_DIF", "MACD_DEA", "MACD_HIST",
    "BB_position",
    "波动率_20日",
    "振幅",
]

X = df[FEATURES].values      # 特征矩阵
y = df["明日涨跌"].values     # 标签：明天涨(1)还是跌(0)

# 按时间切分（不是随机打乱，因为时间序列有先后关系）
split_idx = int(len(df) * TRAIN_RATIO)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

print(f"\n  训练集：{len(X_train)} 天  |  测试集：{len(X_test)} 天")
print(f"  涨跌分布：涨 {y.sum()} 天 / 跌 {len(y)-y.sum()} 天")
print(f"  特征数量：{len(FEATURES)} 个")
print(f"\n  特征清单：")
for i, f in enumerate(FEATURES, 1):
    print(f"    {i:2d}. {f}")

# ============================================================
# 第4步：训练模型
# ============================================================
print("\n" + "=" * 60)
print("  正在训练随机森林模型...")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=200,       # 200棵树
    max_depth=8,            # 限制深度防止过拟合
    min_samples_leaf=10,    # 叶子节点最少10个样本
    random_state=42,        # 固定随机种子，保证结果可复现
    n_jobs=-1,              # 使用所有CPU核心
)
model.fit(X_train, y_train)

# 在测试集上评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n  测试集准确率：{accuracy:.2%}")
print(f"  （基准准确率：{max(y_test.mean(), 1-y_test.mean()):.2%}，即总是猜涨/跌的正确率）")
print(f"\n  分类报告：")
print(f"    {'':>10}  precision    recall  f1-score")
for label, name in [(0, "明日跌"), (1, "明日涨")]:
    tp = ((y_pred == label) & (y_test == label)).sum()
    fp = ((y_pred == label) & (y_test != label)).sum()
    fn = ((y_pred != label) & (y_test == label)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    support = (y_test == label).sum()
    print(f"    {name:>10}  {p:>8.2%}  {r:>8.2%}  {f1:>8.2%}  ({support}天)")

# ============================================================
# 第5步：特征重要性
# ============================================================
print("\n" + "=" * 60)
print("  特征重要性排名（Top 10）")
print("=" * 60)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
for rank, idx in enumerate(indices[:10], 1):
    bar = "█" * int(importances[idx] * 100)
    print(f"  {rank:2d}. {FEATURES[idx]:<20s}  {importances[idx]:.4f}  {bar}")

# ============================================================
# 第6步：回测
# ============================================================
print("\n" + "=" * 60)
print("  正在回测...")
print("=" * 60)

# 获取模型预测概率（涨的概率）
y_prob = model.predict_proba(X_test)[:, 1]

# 策略收益：预测涨则持仓，预测跌则空仓
strategy_returns = []
benchmark_returns = []
positions = []  # 记录持仓状态

prev_close = None
test_close = df_test["收盘"].values
test_dates = df_test["日期"].values

for i in range(len(X_test)):
    if i == 0:
        # 测试集第一天，用前一天收盘价算基准
        prev_close = df_train["收盘"].iloc[-1]

    # 当日收益率（相对于前一天收盘）
    daily_ret = (test_close[i] - prev_close) / prev_close

    # 策略：预测概率 > 0.5 就持仓
    if y_prob[i] > 0.5:
        strategy_returns.append(daily_ret)
        positions.append(1)
    else:
        strategy_returns.append(0)  # 空仓，不赚不亏
        positions.append(0)

    benchmark_returns.append(daily_ret)
    prev_close = test_close[i]

# 累计收益
strategy_cum = np.cumprod(1 + np.array(strategy_returns)) - 1
benchmark_cum = np.cumprod(1 + np.array(benchmark_returns)) - 1

# 计算回测指标
total_days = len(X_test)
trade_days = sum(positions)
win_trades = sum(1 for i in range(total_days) if positions[i] == 1 and strategy_returns[i] > 0)
total_return = strategy_cum[-1]
benchmark_return = benchmark_cum[-1]
win_rate = win_trades / trade_days if trade_days > 0 else 0

# 年化收益率
years = total_days / 252
annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
benchmark_annual = (1 + benchmark_return) ** (1 / years) - 1 if years > 0 else 0

# 最大回撤
peak = np.maximum.accumulate(strategy_cum + 1)
drawdown = (strategy_cum + 1 - peak) / peak
max_drawdown = drawdown.min()

# 夏普比率（简化版，假设无风险利率为0）
if len(strategy_returns) > 0:
    sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
else:
    sharpe = 0

print(f"\n  {'指标':<20s}  {'策略':>12s}  {'买入持有':>12s}")
print(f"  {'-'*46}")
print(f"  {'总收益率':<20s}  {total_return:>11.2%}  {benchmark_return:>11.2%}")
print(f"  {'年化收益率':<20s}  {annual_return:>11.2%}  {benchmark_annual:>11.2%}")
print(f"  {'胜率':<20s}  {win_rate:>11.2%}  {'--':>12s}")
print(f"  {'交易天数/总天数':<20s}  {f'{trade_days}/{total_days}':>12s}  {f'{total_days}/{total_days}':>12s}")
print(f"  {'夏普比率':<20s}  {sharpe:>11.2f}  {'--':>12s}")
print(f"  {'最大回撤':<20s}  {max_drawdown:>11.2%}  {'--':>12s}")

# ============================================================
# 第7步：画图
# ============================================================
print("\n" + "=" * 60)
print("  正在生成收益曲线图...")
print("=" * 60)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 上图：收益曲线
ax1 = axes[0]
ax1.plot(test_dates, strategy_cum * 100, label="ML策略", color="#e74c3c", linewidth=1.5)
ax1.plot(test_dates, benchmark_cum * 100, label="买入持有", color="#3498db", linewidth=1.5, alpha=0.7)
ax1.fill_between(range(len(test_dates)), 0, strategy_cum * 100,
                  where=(np.array(strategy_cum) > np.array(benchmark_cum)),
                  color="#e74c3c", alpha=0.1)
ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
ax1.set_title(f"{STOCK_NAME}({STOCK_CODE}) ML策略回测 — 测试集", fontsize=14, fontweight="bold")
ax1.set_ylabel("累计收益率 (%)", fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 标注最终收益
ax1.annotate(f"策略: {total_return:.2%}", xy=(len(test_dates)-1, strategy_cum[-1]*100),
             xytext=(-80, 10), textcoords="offset points", fontsize=10, color="#e74c3c",
             fontweight="bold")
ax1.annotate(f"基准: {benchmark_return:.2%}", xy=(len(test_dates)-1, benchmark_cum[-1]*100),
             xytext=(-80, -20), textcoords="offset points", fontsize=10, color="#3498db",
             fontweight="bold")

# 下图：持仓热力图 + 每日信号
ax2 = axes[1]
colors = ["#2ecc71" if p == 1 else "#ecf0f1" for p in positions]
ax2.scatter(range(len(positions)), [1] * len(positions),
            c=colors, marker="s", s=1, alpha=0.8)
ax2.set_ylabel("持仓", fontsize=11)
ax2.set_yticks([])
ax2.set_xlabel("交易日序号", fontsize=11)
ax2.set_title("每日持仓状态（绿色=持仓，灰色=空仓）", fontsize=12)

# 设置 x 轴标签
step = max(1, len(test_dates) // 10)
tick_positions = list(range(0, len(test_dates), step))
tick_labels = [test_dates[i].strftime("%Y-%m") if isinstance(test_dates[i], pd.Timestamp)
               else str(test_dates[i])[:7] for i in tick_positions]
ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels, rotation=45, fontsize=8)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "收益曲线.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"  图表已保存到：{output_path}")
print("\n" + "=" * 60)
print("  运行完毕！")
print(f"  策略最终收益: {total_return:+.2%}  |  基准收益: {benchmark_return:+.2%}")
print("=" * 60)

# 提示：如何换股票
print(f"\n  [提示] 修改脚本开头的 STOCK_CODE 可以换股票")
print(f"     如改成 '600519' 就是贵州茅台")
print(f"     在配置区改 STOCK_NAME 可以换个显示名字")
