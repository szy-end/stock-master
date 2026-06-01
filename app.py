"""
炒股大王施大师（亏了别找我） - 本地桌面App
====================================
基于 Streamlit 的本地应用，支持：
  - 市场概览（指数行情、板块走势）
  - 个股深度分析（K线、技术指标、基本面）
  - AI智能推荐（ML预测 + 技术面 + 资金面 + 新闻）
  - 新闻雷达（联网搜索 + 情感分析）

启动方式：streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak
from datetime import datetime, timedelta
import time

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="炒股大王施大师（亏了别找我）",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 响应式CSS — 手机端自适应
st.markdown("""
<style>
/* 整体容器边距优化 */
.main .block-container {
    padding: 1rem 0.5rem !important;
    max-width: 100% !important;
}

/* 小屏手机 (&lt;= 640px) */
@media screen and (max-width: 640px) {
    /* 标题缩小 */
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 1rem !important; }

    /* 卡片文字缩小 */
    .prediction-card { padding: 10px !important; }
    .prediction-card .big-num { font-size: 32px !important; }
    .prediction-card .label { font-size: 12px !important; }

    /* 表格字体缩小 */
    .stDataFrame { font-size: 12px !important; }

    /* 指标卡片居中 */
    [data-testid="stMetric"] {
        text-align: center !important;
    }
}

/* 平板 (641-1024px) */
@media screen and (min-width: 641px) and (max-width: 1024px) {
    h1 { font-size: 1.8rem !important; }
}

/* 防止内容溢出 */
.stDataFrame, .stTable {
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch;
}

/* 按钮触摸友好 */
button, .stButton > button {
    min-height: 44px !important;
    min-width: 44px !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 缓存函数 —— 避免重复请求数据
# ============================================================

@st.cache_data(ttl=600)
def get_index_data():
    """获取主要指数最新行情（使用历史数据接口，更稳定）"""
    index_map = {
        "上证指数": "sh000001",
        "深证成指": "sz399001",
        "沪深300": "sh000300",
        "创业板指": "sz399006",
        "科创50": "sh000688",
        "中证500": "sh000905",
        "上证50": "sh000016",
    }
    results = []
    for name, symbol in index_map.items():
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df is not None and len(df) >= 2:
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                change_pct = (latest["close"] - prev["close"]) / prev["close"] * 100
                results.append({
                    "指数": name,
                    "最新价": round(float(latest["close"]), 2),
                    "涨跌幅": round(change_pct, 2),
                    "涨跌额": round(float(latest["close"] - prev["close"]), 2),
                    "成交量": int(latest["volume"]),
                    "成交额": "N/A",
                })
        except Exception:
            continue

    if results:
        return pd.DataFrame(results)
    return None


@st.cache_data(ttl=600)
def get_sector_data():
    """获取行业板块行情（深交所行业统计）"""
    try:
        df = ak.stock_szse_sector_summary(symbol="按行业分")
        if df is None or len(df) == 0:
            return None
        # 列名映射（深交所数据可能有中英文混杂）
        col_map = {}
        for c in df.columns:
            if "项目" in c or "类别" in c:
                col_map[c] = "板块名称"
            elif "公司" in c and "家数" in c:
                col_map[c] = "公司家数"
            elif "市值" in c and "总值" in c:
                col_map[c] = "总市值"
            elif "成交" in c and "金额" in c:
                col_map[c] = "成交额"
            elif "成交" in c and "数量" in c:
                col_map[c] = "成交量"
        df = df.rename(columns=col_map)
        # 过滤掉"合计"行
        if "板块名称" in df.columns:
            df = df[~df["板块名称"].str.contains("合计", na=False)]
        df = df.head(19)  # 排除合计行后取top 19
        return df
    except Exception:
        return None


@st.cache_data(ttl=600)
def get_north_flow():
    """获取北向资金流向"""
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        if df is not None and len(df) > 0:
            df = df.tail(30)
            # 兼容不同列名格式
            date_col = "日期" if "日期" in df.columns else df.columns[0]
            net_col = "当日成交净买额" if "当日成交净买额" in df.columns else \
                      [c for c in df.columns if "净买" in str(c)][0] if any("净买" in str(c) for c in df.columns) else None
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
            return df
        return None
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_stock_daily(code, period="daily", start_date="20200101"):
    """获取个股历史日线数据"""
    try:
        if code.startswith(("60", "68")):
            symbol = f"sh{code}"
        else:
            symbol = f"sz{code}"

        df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date, adjust="qfq")
        if df is not None and len(df) > 0:
            df = df.rename(columns={
                "date": "日期", "open": "开盘", "high": "最高",
                "low": "最低", "close": "收盘", "volume": "成交量",
                "amount": "成交额", "turnover": "换手率",
            })
        return df
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_stock_info(code):
    """获取个股基本信息（从日线数据推导 + 尝试EM接口）"""
    info = {}
    # 尝试获取EM基本信息
    try:
        df = ak.stock_individual_info_em(symbol=code)
        for _, row in df.iterrows():
            info[row["item"]] = row["value"]
    except Exception:
        pass

    # 如果EM失败，从日线数据推导基本信息
    if not info:
        try:
            if code.startswith(("60", "68")):
                symbol = f"sh{code}"
            else:
                symbol = f"sz{code}"
            daily = ak.stock_zh_a_daily(symbol=symbol, start_date="20250101", adjust="qfq")
            if daily is not None and len(daily) > 0:
                latest = daily.iloc[-1]
                info["最新"] = str(latest.get("close", ""))
                info["股票简称"] = code
                info["总市值"] = "N/A"
                info["市盈率-动态"] = "N/A"
                info["市净率"] = "N/A"
                info["流通市值"] = "N/A"
                info["换手率"] = f"{latest.get('turnover', 0):.2f}%" if "turnover" in daily.columns else "N/A"
        except Exception:
            pass

    return info


@st.cache_data(ttl=600)
def get_stock_news(keyword, limit=15):
    """通过akshare获取个股相关新闻"""
    try:
        df = ak.stock_news_em()
        if keyword:
            mask = df["标题"].str.contains(keyword, na=False)
            df = df[mask]
        df = df.head(limit)
        return df
    except Exception:
        return None


@st.cache_data(ttl=600)
def web_search_news(query, max_results=10):
    """通过DuckDuckGo搜索新闻"""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, region="cn-zh", max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "url": r.get("url", ""),
                    "source": r.get("source", ""),
                    "date": r.get("date", ""),
                })
        return results
    except Exception:
        # DuckDuckGo可能因网络问题不可用
        return []


# ============================================================
# ML 分析函数
# ============================================================

def compute_features(df):
    """计算技术指标特征"""
    close = df["收盘"].values
    volume = df["成交量"].values
    high = df["最高"].values
    low = df["最低"].values

    features = {}

    # 价格特征
    features["收益率_1日"] = pd.Series(close).pct_change(1).values
    features["收益率_5日"] = pd.Series(close).pct_change(5).values
    features["收益率_20日"] = pd.Series(close).pct_change(20).values

    # 均线偏离
    for w in [5, 10, 20, 60]:
        ma = pd.Series(close).rolling(w).mean().values
        features[f"偏离_MA{w}"] = close / ma - 1

    # 成交量
    vol_ma5 = pd.Series(volume).rolling(5).mean().values
    vol_ma20 = pd.Series(volume).rolling(20).mean().values
    features["量变_5日"] = volume / vol_ma5 - 1
    features["量变_20日"] = volume / vol_ma20 - 1

    # RSI
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    features["RSI_14"] = 100 - (100 / (1 + rs)).values

    # MACD
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    dif = (ema12 - ema26).values
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    features["MACD_DIF"] = dif
    features["MACD_HIST"] = 2 * (dif - dea)

    # 布林带
    bb_mid = pd.Series(close).rolling(20).mean().values
    bb_std = pd.Series(close).rolling(20).std().values
    features["BB位置"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)

    # 波动率
    features["波动率_20日"] = pd.Series(close).pct_change().rolling(20).std().values

    # 振幅
    features["振幅"] = (high - low) / pd.Series(close).shift(1).values

    # 换手率（如果有的话）
    if "换手率" in df.columns:
        features["换手率"] = df["换手率"].values

    return pd.DataFrame(features)


def train_ml_model(df):
    """训练ML模型并返回预测"""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    feat_df = compute_features(df)
    feat_df["target"] = (df["收盘"].shift(-1) > df["收盘"]).astype(int).values

    feat_df = feat_df.dropna()
    if len(feat_df) < 200:
        return None, None, None, None

    feature_cols = [c for c in feat_df.columns if c != "target"]
    X = feat_df[feature_cols].values
    y = feat_df["target"].values

    # 80/20 切分
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 训练两个模型
    rf = RandomForestClassifier(n_estimators=150, max_depth=8,
                                 min_samples_leaf=10, random_state=42)
    rf.fit(X_train, y_train)

    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                     min_samples_leaf=10, random_state=42)
    gb.fit(X_train, y_train)

    # 测试集准确率
    rf_acc = rf.score(X_test, y_test)
    gb_acc = gb.score(X_test, y_test)

    # 对最新一天预测
    latest_features = X[-1:].copy()
    rf_prob = rf.predict_proba(latest_features)[0, 1]
    gb_prob = gb.predict_proba(latest_features)[0, 1]
    avg_prob = (rf_prob + gb_prob) / 2

    # 特征重要性
    importances = rf.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:5]

    # 基准准确率
    baseline = max(y_test.mean(), 1 - y_test.mean())

    results = {
        "rf_acc": rf_acc,
        "gb_acc": gb_acc,
        "avg_acc": (rf_acc + gb_acc) / 2,
        "baseline": baseline,
        "rf_prob": rf_prob,
        "gb_prob": gb_prob,
        "avg_prob": avg_prob,
        "prediction": "涨" if avg_prob > 0.5 else "跌",
        "feat_imp": feat_imp,
        "test_size": len(y_test),
    }

    return results, rf, gb, feat_df


def calc_technical_score(df):
    """计算技术指标综合评分"""
    if len(df) < 60:
        return 50

    close = df["收盘"].values
    score = 50  # 基准分

    # MA趋势（+/-10分）
    ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
    ma60 = pd.Series(close).rolling(60).mean().iloc[-1]
    if close[-1] > ma20:
        score += 5
    else:
        score -= 5
    if ma20 > ma60:
        score += 5
    else:
        score -= 5

    # RSI（+/-8分）
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    loss = (-delta).clip(lower=0).rolling(14).mean().iloc[-1]
    if loss > 0:
        rsi = 100 - (100 / (1 + gain / loss))
    else:
        rsi = 100
    if 30 < rsi < 70:
        score += 5
    elif rsi < 30:
        score += 8  # 超卖，看多
    else:
        score -= 5

    # 成交量（+/-5分）
    vol_ma20 = pd.Series(df["成交量"].values).rolling(20).mean().iloc[-1]
    vol_ratio = df["成交量"].iloc[-1] / vol_ma20
    if 1.2 < vol_ratio < 2.5:
        score += 5
    elif vol_ratio < 0.5:
        score -= 3

    # MACD（+/-7分）
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    dif = ema12.iloc[-1] - ema26.iloc[-1]
    dea = pd.Series(ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1]
    if dif > dea and dif > 0:
        score += 7
    elif dif < dea and dif < 0:
        score -= 7

    return max(0, min(100, score))


def calc_fund_score(info):
    """根据资金流向计算评分"""
    score = 50
    # 如果akshare返回了资金流向数据
    try:
        # 尝试从基本信息中提取
        if info:
            pe = float(info.get("市盈率-动态", 0))
            if 0 < pe < 20:
                score += 10
            elif pe > 100:
                score -= 10
            elif pe <= 0:
                score -= 5
    except Exception:
        pass
    return max(0, min(100, score))


def analyze_news_sentiment(news_list):
    """简单的新闻情感分析"""
    if not news_list:
        return 50, []

    positive_words = ["利好", "增长", "突破", "增持", "回购", "盈利", "上涨", "中标",
                      "签约", "创新高", "超预期", "政策支持", "补贴", "获批"]
    negative_words = ["利空", "下滑", "亏损", "减持", "处罚", "调查", "跌停", "退市",
                      "暴雷", "违约", "诉讼", "监管", "警告", "风险"]

    sentiments = []
    total_score = 0

    for news in news_list:
        text = str(news.get("title", "")) + str(news.get("body", ""))
        pos = sum(1 for w in positive_words if w in text)
        neg = sum(1 for w in negative_words if w in text)

        if pos > neg:
            sent = "positive"
            s_score = 60
        elif neg > pos:
            sent = "negative"
            s_score = 40
        else:
            sent = "neutral"
            s_score = 50

        sentiments.append(sent)
        total_score += s_score

    avg_score = total_score / len(sentiments) if sentiments else 50
    return avg_score, sentiments


# ============================================================
# 智能选股函数
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data_cached(code):
    """缓存股票数据，避免重复请求"""
    try:
        # 判断交易所前缀
        if code.startswith(("60", "68")):
            symbol = f"sh{code}"
        else:
            symbol = f"sz{code}"
        df = ak.stock_zh_a_daily(symbol=symbol, start_date="20240101", adjust="qfq")
        if df is not None and len(df) >= 60:
            # 统一列名为中文，兼容后续代码
            df = df.rename(columns={
                "date": "日期", "open": "开盘", "high": "最高",
                "low": "最低", "close": "收盘", "volume": "成交量",
                "amount": "成交额", "turnover": "换手率",
            })
            return df
    except Exception:
        pass
    return None


def quick_stock_score(code, name):
    """快速评估单只股票的上涨潜力（0-100分）"""
    df = fetch_stock_data_cached(code)
    if df is None:
        return None

    try:
        close = df["收盘"].values
        volume = df["成交量"].values
        high = df["最高"].values
        low = df["最低"].values
        n = len(close)

        # ---- 1. 技术面评分 (35%) ----
        tech_score = 50
        # MA多头排列
        ma5 = pd.Series(close).rolling(5).mean().iloc[-1]
        ma10 = pd.Series(close).rolling(10).mean().iloc[-1]
        ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
        ma60 = pd.Series(close).rolling(60).mean().iloc[-1]

        if close[-1] > ma5 > ma10 > ma20:
            tech_score += 15  # 完美多头排列
        elif close[-1] > ma20 and ma5 > ma20:
            tech_score += 10
        elif close[-1] > ma20:
            tech_score += 5
        elif close[-1] < ma60:
            tech_score -= 10

        # RSI
        delta = pd.Series(close).diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta).clip(lower=0).rolling(14).mean().iloc[-1]
        if loss > 0:
            rsi = 100 - (100 / (1 + gain / loss))
        else:
            rsi = 100
        if 30 <= rsi <= 70:
            tech_score += 10
        elif rsi < 30:
            tech_score += 15  # 超卖反弹
        else:
            tech_score -= 5

        # MACD
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        dif = ema12.iloc[-1] - ema26.iloc[-1]
        dea = pd.Series(ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1]
        if dif > dea and dif > 0:
            tech_score += 10
        elif dif < dea and dif < 0:
            tech_score -= 10

        tech_score = max(0, min(100, tech_score))

        # ---- 2. 动量评分 (30%) ----
        ret_5d = (close[-1] / close[-6] - 1) * 100 if n >= 6 else 0
        ret_10d = (close[-1] / close[-11] - 1) * 100 if n >= 11 else 0
        ret_20d = (close[-1] / close[-21] - 1) * 100 if n >= 21 else 0

        mom_score = 50
        mom_score += ret_5d * 2.0   # 短期动量权重高
        mom_score += ret_10d * 1.0
        mom_score += ret_20d * 0.5
        mom_score = max(0, min(100, mom_score))

        # ---- 3. ML预测评分 (35%) ----
        from sklearn.ensemble import RandomForestClassifier

        # 构建特征
        feats = pd.DataFrame({
            "ret_1": pd.Series(close).pct_change(1),
            "ret_5": pd.Series(close).pct_change(5),
            "ret_10": pd.Series(close).pct_change(10),
            "ma5_bias": close / ma5 - 1,
            "ma20_bias": close / ma20 - 1,
            "vol_ratio": volume / pd.Series(volume).rolling(20).mean().values,
            "rsi": 100 - (100 / (1 + pd.Series(close).diff().clip(lower=0).rolling(14).mean() /
                                 pd.Series(close).diff().clip(upper=0).abs().rolling(14).mean())),
            "bb_pos": (close - (ma20 - 2 * pd.Series(close).rolling(20).std().values)) /
                      (4 * pd.Series(close).rolling(20).std().values + 1e-10),
            "amplitude": (high - low) / pd.Series(close).shift(1).values,
        })
        feats["target"] = (pd.Series(close).shift(-1) > close).astype(int)
        feats = feats.dropna()
        if len(feats) < 80:
            ml_score = 50  # 数据不足
        else:
            X = feats[[c for c in feats.columns if c != "target"]].values
            y = feats["target"].values
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            rf = RandomForestClassifier(n_estimators=80, max_depth=6,
                                         min_samples_leaf=5, random_state=42)
            rf.fit(X_train, y_train)
            prob = rf.predict_proba(X_test[-1:])[0, 1]
            ml_score = prob * 100

        # ---- 综合 ----
        final = tech_score * 0.35 + mom_score * 0.30 + ml_score * 0.35

        return {
            "code": code,
            "name": name,
            "final": round(final, 1),
            "tech": round(tech_score, 1),
            "momentum": round(mom_score, 1),
            "ml": round(ml_score, 1),
            "ret_5d": round(ret_5d, 2),
            "ret_20d": round(ret_20d, 2),
            "price": round(float(close[-1]), 2),
        }
    except Exception:
        return None


# ============================================================
# Streamlit UI
# ============================================================

st.title("炒股大王施大师（亏了别找我）")
st.caption("本地运行的AI股票分析工具 — 仅供学习参考，不构成投资建议")

# 侧边栏：全局设置
with st.sidebar:
    st.header("设置")
    stock_code = st.text_input("股票代码", value="600519", placeholder="输入6位代码，如 600519")
    stock_name_input = st.text_input("股票名称（可选）", value="", placeholder="自动获取")

    st.divider()
    st.subheader("智能选股")
    scan_universe = st.selectbox("股票池", ["沪深300", "中证500", "全部A股(5525只)"], index=0)
    scan_count = st.slider("扫描数量", 10, 5000, 50, step=10,
                           help="越多越慢。500只≈8分钟，5000只≈80分钟。全部A股模式下随机抽取。")

    st.divider()
    st.caption("数据来源：东方财富、AKShare")
    st.caption("新闻搜索：东方财富 + DuckDuckGo")
    st.caption(f"更新时间：{datetime.now().strftime('%H:%M:%S')}")

# ============================================================
# 4个标签页
# ============================================================
tab_smart, tab_market, tab_stock, tab_recommend, tab_news = st.tabs(
    ["智能选股", "市场概览", "个股分析", "智能推荐", "新闻雷达"]
)

# ===== 标签1：市场概览 =====
with tab_market:
    st.subheader("主要指数行情")

    col1, col2 = st.columns([2, 1])

    with col1:
        index_df = get_index_data()
        if index_df is not None:
            # 用颜色标记涨跌
            def color_val(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return f'color: #e74c3c'
                    elif val < 0:
                        return f'color: #27ae60'
                return ''

            styled = index_df.style.map(
                color_val, subset=["涨跌幅", "涨跌额"]
            ).format({
                "涨跌幅": "{:.2f}%",
                "涨跌额": "{:+.2f}",
                "最新价": "{:.2f}",
            })
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.warning("指数数据获取失败，请检查网络连接")

    with col2:
        if index_df is not None:
            up = (index_df["涨跌幅"] > 0).sum()
            down = (index_df["涨跌幅"] < 0).sum()
            flat = (index_df["涨跌幅"] == 0).sum()
            st.metric("上涨指数", f"{up} 个", delta=None)
            st.metric("下跌指数", f"{down} 个", delta=None)

    st.divider()

    # 行业板块
    st.subheader("深交所行业板块统计")
    sector_df = get_sector_data()
    if sector_df is not None and len(sector_df) > 0:
        st.dataframe(sector_df, use_container_width=True, hide_index=True)
    else:
        st.info("板块数据暂不可用")

    st.divider()

    # 北向资金
    st.subheader("北向资金近期流向")
    north_df = get_north_flow()
    if north_df is not None and len(north_df) > 0:
        # 动态检测列名
        cols = north_df.columns.tolist()
        date_col = next((c for c in cols if "日期" in str(c) or "date" in str(c).lower()), cols[0])
        net_col = next((c for c in cols if "净买" in str(c)), None)

        if net_col:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=north_df[date_col],
                y=north_df[net_col],
                name="净买入",
                marker_color=["#e74c3c" if v > 0 else "#27ae60" for v in north_df[net_col]],
            ))
            fig.add_trace(go.Scatter(
                x=north_df[date_col],
                y=north_df[net_col].cumsum(),
                name="累计净买入",
                line=dict(color="#3498db", width=2),
                yaxis="y2",
            ))
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=0, b=0),
                yaxis=dict(title="日净买入（亿）"),
                yaxis2=dict(title="累计净买入（亿）", overlaying="y", side="right"),
                legend=dict(x=0, y=1.1, orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(north_df.tail(10), use_container_width=True, hide_index=True)
    else:
        st.info("北向资金数据暂不可用")

# ===== 标签2：个股分析 =====
with tab_stock:
    code = st.session_state.get("stock_code", "600519") if stock_code else "600519"
    code = stock_code or "600519"

    if len(code) != 6 or not code.isdigit():
        st.warning("请输入正确的6位股票代码")
    else:
        with st.spinner(f"正在加载 {code} 的数据..."):
            df = get_stock_daily(code, start_date="20220101")
            info = get_stock_info(code)

        if df is None or len(df) == 0:
            st.error(f"获取 {code} 数据失败，请检查代码是否正确")
        else:
            stock_name = stock_name_input or info.get("股票简称", code)
            st.subheader(f"{stock_name} ({code})")

            # 基本信息卡片
            if info:
                cols = st.columns(6)
                metrics = [
                    ("最新价", info.get("最新", "-")),
                    ("总市值", info.get("总市值", "-")),
                    ("市盈率", info.get("市盈率-动态", "-")),
                    ("市净率", info.get("市净率", "-")),
                    ("流通市值", info.get("流通市值", "-")),
                    ("换手率", f"{df['换手率'].iloc[-1]:.2f}%" if "换手率" in df.columns else "-"),
                ]
                for i, (label, val) in enumerate(metrics):
                    with cols[i]:
                        st.metric(label, val)

            st.divider()

            # K线图
            st.subheader("K线图 & 技术指标")

            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=("K线与均线", "成交量", "MACD"),
            )

            # 取近120天
            plot_df = df.tail(180).copy()
            dates = plot_df["日期"].values if "日期" in plot_df.columns else plot_df.index.values

            # K线
            fig.add_trace(go.Candlestick(
                x=dates,
                open=plot_df["开盘"],
                high=plot_df["最高"],
                low=plot_df["最低"],
                close=plot_df["收盘"],
                name="K线",
                increasing_line_color="#e74c3c",
                decreasing_line_color="#27ae60",
            ), row=1, col=1)

            # 均线
            for w, color in [(5, "#f39c12"), (20, "#3498db"), (60, "#9b59b6")]:
                ma = plot_df["收盘"].rolling(w).mean()
                fig.add_trace(go.Scatter(
                    x=dates, y=ma, name=f"MA{w}",
                    line=dict(color=color, width=1),
                ), row=1, col=1)

            # 成交量
            colors = ["#e74c3c" if plot_df["收盘"].iloc[i] >= plot_df["开盘"].iloc[i] else "#27ae60"
                      for i in range(len(plot_df))]
            fig.add_trace(go.Bar(
                x=dates, y=plot_df["成交量"], name="成交量",
                marker_color=colors, opacity=0.5,
            ), row=2, col=1)

            # MACD
            ema12 = plot_df["收盘"].ewm(span=12, adjust=False).mean()
            ema26 = plot_df["收盘"].ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9, adjust=False).mean()
            hist = 2 * (dif - dea)
            fig.add_trace(go.Scatter(x=dates, y=dif, name="DIF",
                                     line=dict(color="#3498db", width=1)), row=3, col=1)
            fig.add_trace(go.Scatter(x=dates, y=dea, name="DEA",
                                     line=dict(color="#e74c3c", width=1)), row=3, col=1)
            fig.add_trace(go.Bar(x=dates, y=hist, name="MACD柱",
                                 marker_color=["#e74c3c" if v > 0 else "#27ae60" for v in hist],
                                 opacity=0.5), row=3, col=1)

            fig.update_layout(
                height=450,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_rangeslider_visible=False,
                legend=dict(x=0, y=1.1, orientation="h"),
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(title_text="价格", row=1, col=1)
            fig.update_yaxes(title_text="成交量", row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # 基本面信息
            if info:
                st.divider()
                st.subheader("基本面信息")
                cols = st.columns(3)
                info_items = list(info.items())
                for i, (k, v) in enumerate(info_items):
                    with cols[i % 3]:
                        st.text(f"{k}: {v}")

# ===== 标签3：智能推荐 =====
with tab_recommend:
    code = stock_code or "600519"

    if len(code) != 6 or not code.isdigit():
        st.warning("请在左侧输入正确的6位股票代码")
    else:
        st.subheader("AI 智能分析推荐")

        with st.spinner("正在加载数据并训练模型..."):
            df = get_stock_daily(code, start_date="20180101")
            info = get_stock_info(code)

        if df is None or len(df) < 200:
            st.error(f"数据不足（需要至少200个交易日），当前仅{len(df) if df is not None else 0}条")
        else:
            stock_name = stock_name_input or info.get("股票简称", code) if info else code

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("ML 模型预测")
                with st.spinner("训练模型中..."):
                    ml_results, rf_model, gb_model, feat_df = train_ml_model(df)

                if ml_results is None:
                    st.error("模型训练失败")
                else:
                    # 预测结果
                    prob = ml_results["avg_prob"]
                    pred = ml_results["prediction"]
                    color = "#e74c3c" if pred == "涨" else "#27ae60"

                    st.markdown(f"""
                    <div style="
                        background: {color}15;
                        border: 2px solid {color};
                        border-radius: 12px;
                        padding: 20px;
                        text-align: center;
                        margin: 10px 0;
                    ">
                        <div style="font-size: 14px; color: #888;">模型预测次日</div>
                        <div style="font-size: clamp(28px, 8vw, 48px); font-weight: bold; color: {color};">{pred}</div>
                        <div style="font-size: 16px; color: #666;">
                            看涨概率 {prob:.1%} | 信心度 {abs(prob - 0.5) * 2:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 模型准确率
                    st.caption(f"随机森林测试准确率: {ml_results['rf_acc']:.2%} | "
                              f"GBDT: {ml_results['gb_acc']:.2%} | "
                              f"基准: {ml_results['baseline']:.2%}")

                    # 特征重要性
                    st.caption("**Top 5 重要特征**")
                    for name, imp in ml_results["feat_imp"]:
                        st.caption(f"  {name}: {imp:.4f}")

            with col2:
                st.subheader("综合评分")

                # 1. ML信号（权重40%）
                ml_score = ml_results["avg_prob"] * 100 if ml_results else 50

                # 2. 技术指标（权重30%）
                tech_score = calc_technical_score(df)

                # 3. 资金面（权重20%）
                fund_score = calc_fund_score(info)

                # 4. 新闻情感（权重10%）
                news_list = web_search_news(f"{stock_name} {code} A股")
                news_score, sentiments = analyze_news_sentiment(news_list)

                # 综合
                final_score = (
                    ml_score * 0.4 +
                    tech_score * 0.3 +
                    fund_score * 0.2 +
                    news_score * 0.1
                )

                # 推荐
                if final_score >= 65:
                    recommendation = "买入关注"
                    rec_color = "#e74c3c"
                    rec_desc = "多个维度发出积极信号，可重点关注"
                elif final_score >= 45:
                    recommendation = "观望持有"
                    rec_color = "#f39c12"
                    rec_desc = "信号中性，建议观察等待更明确的方向"
                else:
                    recommendation = "回避减仓"
                    rec_color = "#27ae60"
                    rec_desc = "多维度偏空，注意风险控制"

                st.markdown(f"""
                <div style="
                    background: {rec_color}15;
                    border: 2px solid {rec_color};
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                    margin: 10px 0;
                ">
                    <div style="font-size: 14px; color: #888;">综合建议</div>
                    <div style="font-size: clamp(24px, 6vw, 42px); font-weight: bold; color: {rec_color};">{recommendation}</div>
                    <div style="font-size: 16px; color: #666;">综合评分 {final_score:.1f}/100</div>
                    <div style="font-size: 13px; color: #999; margin-top: 8px;">{rec_desc}</div>
                </div>
                """, unsafe_allow_html=True)

                # 各维度得分
                st.divider()
                st.caption("各维度评分明细")
                dims = pd.DataFrame({
                    "维度": ["ML预测", "技术指标", "资金面", "新闻情感"],
                    "权重": ["40%", "30%", "20%", "10%"],
                    "得分": [f"{ml_score:.1f}", f"{tech_score:.1f}", f"{fund_score:.1f}", f"{news_score:.1f}"],
                })

                fig = go.Figure(data=[
                    go.Bar(
                        x=["ML预测", "技术指标", "资金面", "新闻情感"],
                        y=[ml_score, tech_score, fund_score, news_score],
                        marker_color=["#3498db", "#2ecc71", "#e67e22", "#9b59b6"],
                        text=[f"{ml_score:.1f}", f"{tech_score:.1f}", f"{fund_score:.1f}", f"{news_score:.1f}"],
                        textposition="outside",
                    )
                ])
                fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0),
                                  yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.warning("**免责声明**：以上分析完全基于历史数据和公开信息，不构成任何投资建议。"
               "股市有风险，投资需谨慎。ML模型在历史数据上表现不代表未来收益。")

# ===== 标签4：新闻雷达 =====
with tab_news:
    code = stock_code or "600519"

    if len(code) != 6 or not code.isdigit():
        st.warning("请在左侧输入正确的6位股票代码")
    else:
        st.subheader("新闻雷达 - 联网搜索")

        info = get_stock_info(code)
        stock_name = stock_name_input or info.get("股票简称", code) if info else code

        search_keyword = st.text_input("搜索关键词", value=f"{stock_name} {code}", key="news_search")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("东方财富新闻")
            with st.spinner("正在获取..."):
                news_df = get_stock_news(f"{stock_name} {code}", limit=15)

            if news_df is not None and len(news_df) > 0:
                for _, row in news_df.iterrows():
                    title = row.get("标题", "")
                    pub_time = row.get("发布时间", "")
                    source = row.get("来源", "")

                    # 简单情感着色
                    pos_words = ["利好", "增长", "突破", "增持", "回购", "盈利", "上涨"]
                    neg_words = ["利空", "下滑", "亏损", "减持", "处罚", "跌停", "风险"]
                    title_str = str(title)
                    pos_count = sum(1 for w in pos_words if w in title_str)
                    neg_count = sum(1 for w in neg_words if w in title_str)

                    if pos_count > neg_count:
                        emoji = "🟢"
                    elif neg_count > pos_count:
                        emoji = "🔴"
                    else:
                        emoji = "⚪"

                    with st.expander(f"{emoji} {title}", expanded=False):
                        st.caption(f"来源：{source} | 时间：{pub_time}")
                        content = row.get("内容", "")
                        if content:
                            st.text(str(content)[:300])
            else:
                st.info("暂无相关新闻")

        with col2:
            st.subheader("网络搜索")
            with st.spinner("正在联网搜索..."):
                web_news = web_search_news(search_keyword, max_results=10)

            if web_news:
                for i, news in enumerate(web_news):
                    title = news.get("title", "")
                    body = news.get("body", "")
                    url = news.get("url", "")
                    source = news.get("source", "")
                    pub_date = news.get("date", "")

                    # 情感
                    pos_words = ["利好", "增长", "突破", "增持", "回购", "盈利", "上涨", "创新高"]
                    neg_words = ["利空", "下滑", "亏损", "减持", "处罚", "跌停", "退市", "风险"]
                    text = title + body
                    pos_count = sum(1 for w in pos_words if w in text)
                    neg_count = sum(1 for w in neg_words if w in text)

                    if pos_count > neg_count:
                        emoji = "🟢"
                    elif neg_count > pos_count:
                        emoji = "🔴"
                    else:
                        emoji = "⚪"

                    with st.expander(f"{emoji} {title}", expanded=False):
                        st.caption(f"来源：{source} | 时间：{pub_date}")
                        st.text(body[:300] if body else "")
                        if url:
                            st.caption(f"[阅读原文]({url})")
            else:
                st.info("网络搜索暂无结果（可能网络波动或DuckDuckGo不可用）")

# ===== 标签5：智能选股 =====
with tab_smart:
    st.subheader("AI 智能选股 — 自动扫描推荐")

    if "scan_results" not in st.session_state:
        st.session_state.scan_results = None
    if "scan_running" not in st.session_state:
        st.session_state.scan_running = False

    col1, col2 = st.columns([1, 3])
    with col1:
        do_scan = st.button("开始扫描", type="primary", use_container_width=True,
                            disabled=st.session_state.scan_running)
    with col2:
        st.caption(f"将扫描 {scan_universe} 中的前 {scan_count} 只股票，请耐心等待")

    if do_scan:
        st.session_state.scan_running = True
        st.session_state.scan_results = None

    if st.session_state.scan_running:
        # 获取股票列表
        with st.spinner("获取股票列表..."):
            codes, names = [], []
            try:
                if scan_universe == "全部A股(5525只)":
                    all_df = ak.stock_info_a_code_name()
                    # 过滤掉ST、*ST、退市股
                    all_df = all_df[~all_df["name"].str.contains("ST|退市|退", na=False)]
                    # 排除北交所（8开头）、新三板
                    all_df = all_df[all_df["code"].str.match(r"^(00|30|60|68)\d{4}$", na=False)]
                    # 随机抽取
                    sample = all_df.sample(n=min(scan_count, len(all_df)), random_state=None)
                    codes = sample["code"].tolist()
                    names = sample["name"].tolist()
                else:
                    index_code = "000300" if scan_universe == "沪深300" else "000905"
                    cons_df = ak.index_stock_cons(symbol=index_code)
                    codes = cons_df["品种代码"].tolist()[:scan_count]
                    names = cons_df["品种名称"].tolist()[:scan_count]
            except Exception as e:
                st.error(f"获取股票列表失败：{e}")
                st.session_state.scan_running = False
                codes, names = [], []

        if codes:
            if len(codes) > 300:
                st.warning(f"即将扫描 {len(codes)} 只股票，预计需要 {len(codes)//60}-{len(codes)//30} 分钟。请耐心等待，不要关闭页面。")
            progress_bar = st.progress(0, text="准备扫描...")
            status_area = st.empty()
            results = []

            for i, (code, name) in enumerate(zip(codes, names)):
                pct = (i + 1) / len(codes)
                progress_bar.progress(pct, text=f"正在分析 {name}({code}) ... {i+1}/{len(codes)}")

                score = quick_stock_score(code, name)
                if score:
                    results.append(score)

                # 每10只显示一次中间结果
                if (i + 1) % 10 == 0 and results:
                    top_so_far = sorted(results, key=lambda x: x["final"], reverse=True)[:5]
                    status_text = "当前 Top 5:\n"
                    for r, item in enumerate(top_so_far, 1):
                        status_text += f"  {r}. {item['name']}({item['code']}) 综合{item['final']:.0f}分\n"
                    status_area.text(status_text)

            progress_bar.progress(1.0, text="扫描完成！")
            st.session_state.scan_results = sorted(results, key=lambda x: x["final"], reverse=True)
            st.session_state.scan_running = False
            st.rerun()

    # 显示结果
    if st.session_state.scan_results is not None and len(st.session_state.scan_results) > 0:
        results = st.session_state.scan_results

        st.success(f"扫描完成，共分析 {len(results)} 只股票，以下是排名结果")

        # Top 3 高亮卡片
        st.subheader("Top 3 推荐")
        top_cols = st.columns(3)
        for rank, (col, item) in enumerate(zip(top_cols, results[:3]), 1):
            with col:
                if item["final"] >= 65:
                    bg = "#27ae6015"
                    border = "#27ae60"
                elif item["final"] >= 50:
                    bg = "#f39c1215"
                    border = "#f39c12"
                else:
                    bg = "#e74c3c15"
                    border = "#e74c3c"

                st.markdown(f"""
                <div style="
                    background: {bg};
                    border: 2px solid {border};
                    border-radius: 12px;
                    padding: 16px;
                    text-align: center;
                ">
                    <div style="font-size: 12px; color: #888;">#{rank}</div>
                    <div style="font-size: clamp(16px, 4vw, 22px); font-weight: bold;">{item['name']}</div>
                    <div style="font-size: 14px; color: #666;">{item['code']}</div>
                    <div style="font-size: clamp(24px, 6vw, 36px); font-weight: bold; color: {border}; margin: 8px 0;">
                        {item['final']:.0f}
                    </div>
                    <div style="font-size: 12px; color: #999;">综合评分 / 100</div>
                    <div style="font-size: 13px; margin-top: 8px;">
                        现价 {item['price']} | 5日 {item['ret_5d']:+.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.subheader("完整排名")

        # 表格
        table_data = []
        for rank, item in enumerate(results, 1):
            if item["final"] >= 65:
                tag = "买入关注"
            elif item["final"] >= 50:
                tag = "观望"
            else:
                tag = "回避"
            table_data.append({
                "排名": rank,
                "代码": item["code"],
                "名称": item["name"],
                "综合": item["final"],
                "技术面": f"{item['tech']:.0f}",
                "动量": f"{item['momentum']:.0f}",
                "ML预测": f"{item['ml']:.0f}",
                "5日涨跌": f"{item['ret_5d']:+.1f}%",
                "建议": tag,
            })

        table_df = pd.DataFrame(table_data)

        # 颜色映射
        def highlight_score(val):
            try:
                v = float(val)
                if v >= 65:
                    return "background-color: #27ae6030; color: #27ae60; font-weight: bold"
                elif v >= 50:
                    return "background-color: #f39c1230; color: #f39c12"
                else:
                    return "background-color: #e74c3c30; color: #e74c3c"
            except (ValueError, TypeError):
                return ""

        def highlight_tag(val):
            if "买入" in str(val):
                return "background-color: #27ae6030; color: #27ae60; font-weight: bold"
            elif "回避" in str(val):
                return "background-color: #e74c3c30; color: #e74c3c"
            return ""

        styled_table = table_df.style.map(
            highlight_score, subset=["综合"]
        ).map(
            highlight_tag, subset=["建议"]
        )

        st.dataframe(styled_table, use_container_width=True, hide_index=True)

        st.caption(f"扫描范围：{scan_universe} | 已分析：{len(results)} 只 | "
                   f"平均评分：{np.mean([r['final'] for r in results]):.1f}")

        # 清除按钮
        if st.button("重新扫描", use_container_width=False):
            st.session_state.scan_results = None
            st.cache_data.clear()
            st.rerun()

    elif st.session_state.scan_results is not None and len(st.session_state.scan_results) == 0:
        st.warning("扫描未找到符合条件的股票，请扩大扫描范围或检查网络连接")

    st.divider()
    st.warning("**免责声明**：智能选股基于历史数据和技术指标，不构成投资建议。"
               "评分高不代表未来一定上涨。股市有风险，投资需谨慎。")


# ============================================================
# 底部
# ============================================================
st.divider()
st.caption("炒股大王施大师（亏了别找我） v1.0 | 本工具仅供学习参考，不构成投资建议 | 数据来源：AKShare、东方财富、DuckDuckGo")
