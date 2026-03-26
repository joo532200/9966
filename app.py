from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from io import BytesIO
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="Top-10号码预测", layout="wide")

NUM_COLS = ["平一", "平二", "平三", "平四", "平五", "平六", "特码"]
WAVE_COLS = ["平一波", "平二波", "平三波", "平四波", "平五波", "平六波", "特码波"]
ZODIAC_COLS = ["平一生肖", "平二生肖", "平三生肖", "平四生肖", "平五生肖", "平六生肖", "特码生肖"]

ALL_WAVES = ["红", "蓝", "绿"]
ALL_ZODIACS = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]


def normalize_wave(x: str) -> str:
    x = str(x).strip().replace("色", "").strip()
    if x in ["红", "蓝", "绿"]:
        return x
    raise ValueError(f"未知波色: {x}")


def normalize_zodiac(x: str) -> str:
    x = str(x).strip()
    mapping = {
        "龍": "龙",
        "馬": "马",
        "雞": "鸡",
        "豬": "猪",
    }
    x = mapping.get(x, x)
    if x not in ALL_ZODIACS:
        raise ValueError(f"未知生肖: {x}")
    return x


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValueError("仅支持 csv 或 xlsx 文件")


def load_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) 清理列名前后空格
    df.columns = [str(c).strip() for c in df.columns]

    # 2) 兼容“波色”列名
    rename_map = {
        "平一波色": "平一波",
        "平二波色": "平二波",
        "平三波色": "平三波",
        "平四波色": "平四波",
        "平五波色": "平五波",
        "平六波色": "平六波",
        "特码波色": "特码波",
        "特波": "特码波",
    }
    df = df.rename(columns=rename_map)

    required_cols = ["expect", "openTime"] + NUM_COLS + WAVE_COLS + ZODIAC_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    df["expect"] = pd.to_numeric(df["expect"], errors="coerce")
    df["openTime"] = pd.to_datetime(df["openTime"], errors="coerce")

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in WAVE_COLS:
        df[c] = df[c].apply(normalize_wave)

    for c in ZODIAC_COLS:
        df[c] = df[c].apply(normalize_zodiac)

    if df[["expect", "openTime"] + NUM_COLS].isna().any().any():
        raise ValueError("期号 / 时间 / 号码列存在空值或无法解析的值")

    return df.sort_values(["openTime", "expect"]).reset_index(drop=True)


def build_row_features(history_df: pd.DataFrame) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    if len(history_df) == 0:
        return feats

    tm_series = history_df["特码"].astype(int).tolist()
    all_num_matrix = history_df[NUM_COLS].astype(int).values.tolist()
    all_wave_matrix = history_df[WAVE_COLS].values.tolist()
    all_zodiac_matrix = history_df[ZODIAC_COLS].values.tolist()

    # 1. 特码 lag
    for lag in [1, 2, 3, 5, 10]:
        feats[f"tm_lag_{lag}"] = float(tm_series[-lag]) if len(tm_series) >= lag else -1.0

    # 2. 特码 rolling 统计
    for w in [5, 10, 20]:
        vals = tm_series[-w:]
        if len(vals) > 0:
            feats[f"tm_mean_{w}"] = float(np.mean(vals))
            feats[f"tm_std_{w}"] = float(np.std(vals))
            feats[f"tm_min_{w}"] = float(np.min(vals))
            feats[f"tm_max_{w}"] = float(np.max(vals))
            feats[f"tm_odd_ratio_{w}"] = float(np.mean([v % 2 == 1 for v in vals]))
            feats[f"tm_big_ratio_{w}"] = float(np.mean([v >= 25 for v in vals]))
        else:
            feats[f"tm_mean_{w}"] = -1.0
            feats[f"tm_std_{w}"] = -1.0
            feats[f"tm_min_{w}"] = -1.0
            feats[f"tm_max_{w}"] = -1.0
            feats[f"tm_odd_ratio_{w}"] = -1.0
            feats[f"tm_big_ratio_{w}"] = -1.0

    # 3. 最近几期整期统计
    for lag in [1, 2, 3, 5]:
        if len(history_df) >= lag:
            row = history_df.iloc[-lag]
            nums = [int(row[c]) for c in NUM_COLS]
            feats[f"draw_sum_lag_{lag}"] = float(sum(nums))
            feats[f"draw_span_lag_{lag}"] = float(max(nums) - min(nums))
            feats[f"draw_odd_ratio_lag_{lag}"] = float(np.mean([n % 2 == 1 for n in nums]))
            feats[f"draw_big_ratio_lag_{lag}"] = float(np.mean([n >= 25 for n in nums]))
        else:
            feats[f"draw_sum_lag_{lag}"] = -1.0
            feats[f"draw_span_lag_{lag}"] = -1.0
            feats[f"draw_odd_ratio_lag_{lag}"] = -1.0
            feats[f"draw_big_ratio_lag_{lag}"] = -1.0

    # 4. 号码频率与遗漏
    for num in range(1, 50):
        # 特码频率
        for w in [5, 10, 20]:
            vals = tm_series[-w:]
            feats[f"tm_freq_num_{num}_w{w}"] = float(np.mean([v == num for v in vals])) if len(vals) > 0 else 0.0

        # 全位置频率
        for w in [5, 10, 20]:
            recent_rows = all_num_matrix[-w:]
            if len(recent_rows) == 0:
                feats[f"all_freq_num_{num}_w{w}"] = 0.0
            else:
                cnt = sum(num in row for row in recent_rows)
                feats[f"all_freq_num_{num}_w{w}"] = float(cnt / len(recent_rows))

        # 特码遗漏
        last_tm_idx = None
        for i in range(len(tm_series) - 1, -1, -1):
            if tm_series[i] == num:
                last_tm_idx = i
                break
        feats[f"tm_omit_num_{num}"] = (
            float(len(tm_series) - 1 - last_tm_idx) if last_tm_idx is not None else float(len(tm_series))
        )

        # 全位置遗漏
        last_all_idx = None
        for i in range(len(all_num_matrix) - 1, -1, -1):
            if num in all_num_matrix[i]:
                last_all_idx = i
                break
        feats[f"all_omit_num_{num}"] = (
            float(len(all_num_matrix) - 1 - last_all_idx) if last_all_idx is not None else float(len(all_num_matrix))
        )

    # 5. 波色分布
    for w in [5, 10, 20]:
        recent_rows = all_wave_matrix[-w:]
        total = len(recent_rows) * 7
        for wave in ALL_WAVES:
            cnt = sum(v == wave for row in recent_rows for v in row) if total > 0 else 0
            feats[f"wave_ratio_{wave}_w{w}"] = float(cnt / total) if total > 0 else 0.0

        recent_tm = history_df["特码波"].tolist()[-w:]
        for wave in ALL_WAVES:
            feats[f"tm_wave_ratio_{wave}_w{w}"] = (
                float(np.mean([v == wave for v in recent_tm])) if len(recent_tm) > 0 else 0.0
            )

    # 6. 生肖分布
    for w in [5, 10, 20]:
        recent_rows = all_zodiac_matrix[-w:]
        total = len(recent_rows) * 7
        for z in ALL_ZODIACS:
            cnt = sum(v == z for row in recent_rows for v in row) if total > 0 else 0
            feats[f"zodiac_ratio_{z}_w{w}"] = float(cnt / total) if total > 0 else 0.0

        recent_tm = history_df["特码生肖"].tolist()[-w:]
        for z in ALL_ZODIACS:
            feats[f"tm_zodiac_ratio_{z}_w{w}"] = (
                float(np.mean([v == z for v in recent_tm])) if len(recent_tm) > 0 else 0.0
            )

    # 7. 时间特征
    last_time = history_df.iloc[-1]["openTime"]
    feats["last_weekday"] = float(last_time.weekday())
    feats["last_day"] = float(last_time.day)
    feats["last_month"] = float(last_time.month)

    return feats


def build_supervised_table(df: pd.DataFrame, min_history: int = 30) -> Tuple[pd.DataFrame, pd.Series]:
    rows = []
    targets = []
    expects = []
    open_times = []

    for i in range(min_history, len(df)):
        history = df.iloc[:i].copy()
        current = df.iloc[i]

        feat = build_row_features(history)
        rows.append(feat)
        targets.append(int(current["特码"]) - 1)  # 0~48
        expects.append(int(current["expect"]))
        open_times.append(current["openTime"])

    X = pd.DataFrame(rows)
    y = pd.Series(targets, name="target")
    meta = pd.DataFrame({"expect": expects, "openTime": open_times})
    X = pd.concat([meta, X], axis=1)
    return X, y


def get_feature_columns(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if c not in ["expect", "openTime"]]


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=49,
        n_estimators=220,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    return model


def walk_forward_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    train_start_size: int = 60
):
    feat_cols = get_feature_columns(X)

    top1_hits = 0
    top5_hits = 0
    top10_hits = 0
    total = 0
    detail_rows = []

    if len(X) <= train_start_size:
        raise ValueError("有效样本太少，建议上传更多历史期数。")

    for i in range(train_start_size, len(X)):
        X_train = X.iloc[:i][feat_cols]
        y_train = y.iloc[:i]
        X_test = X.iloc[[i]][feat_cols]
        y_true = int(y.iloc[i])

        model = train_xgb(X_train, y_train)
        proba = model.predict_proba(X_test)[0]
        rank_idx = np.argsort(proba)[::-1]

        top1 = rank_idx[:1] + 1
        top5 = rank_idx[:5] + 1
        top10 = rank_idx[:10] + 1
        true_num = y_true + 1

        top1_hit = int(true_num in top1)
        top5_hit = int(true_num in top5)
        top10_hit = int(true_num in top10)

        top1_hits += top1_hit
        top5_hits += top5_hit
        top10_hits += top10_hit
        total += 1

        detail_rows.append({
            "expect": int(X.iloc[i]["expect"]),
            "true_num": f"{true_num:02d}",
            "top1_pred": f"{int(top1[0]):02d}",
            "top5_pred": ",".join(f"{x:02d}" for x in top5),
            "top10_pred": ",".join(f"{x:02d}" for x in top10),
            "top1_hit": top1_hit,
            "top5_hit": top5_hit,
            "top10_hit": top10_hit,
        })

    detail_df = pd.DataFrame(detail_rows)
    metrics = {
        "top1": top1_hits / total if total else 0.0,
        "top5": top5_hits / total if total else 0.0,
        "top10": top10_hits / total if total else 0.0,
        "test_points": total,
    }
    return metrics, detail_df


def predict_next_top10(df: pd.DataFrame, min_history: int = 30):
    X, y = build_supervised_table(df, min_history=min_history)
    feat_cols = get_feature_columns(X)

    model = train_xgb(X[feat_cols], y)

    next_feat = build_row_features(df.copy())
    X_next = pd.DataFrame([next_feat])[feat_cols]

    proba = model.predict_proba(X_next)[0]
    rank_idx = np.argsort(proba)[::-1]

    result = pd.DataFrame({
        "号码": rank_idx + 1,
        "概率": proba[rank_idx]
    }).head(10).reset_index(drop=True)

    result["号码"] = result["号码"].apply(lambda x: f"{x:02d}")
    return result


def to_excel_bytes(top10_df: pd.DataFrame, detail_df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        top10_df.to_excel(writer, index=False, sheet_name="下一期Top10")
        detail_df.to_excel(writer, index=False, sheet_name="回测明细")
    return output.getvalue()


st.title("上传 Excel → 自动预测下一期 Top-10 号码")
st.caption("自动清理列名空格，兼容‘波’和‘波色’字段。")

with st.expander("支持的表头写法", expanded=False):
    st.write("expect, openTime, 平一, 平二, 平三, 平四, 平五, 平六, 特码")
    st.write("平一波 / 平一波色 都可以，其余同理")
    st.write("平一生肖, 平二生肖, 平三生肖, 平四生肖, 平五生肖, 平六生肖, 特码生肖")

uploaded = st.file_uploader("上传 Excel 或 CSV 文件", type=["xlsx", "csv"])
min_history = st.slider("最小历史期数", 20, 60, 30, 5)
train_start_size = st.slider("回测起始训练样本数", 40, 120, 60, 5)

if uploaded is not None:
    try:
        raw_df = read_uploaded_file(uploaded)
        st.subheader("数据预览")
        st.dataframe(raw_df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"文件读取失败：{e}")

if st.button("开始分析并生成 Top-10 号码", type="primary"):
    if uploaded is None:
        st.warning("请先上传文件")
    else:
        try:
            uploaded.seek(0)
            raw_df = read_uploaded_file(uploaded)
            df = load_data(raw_df)

            with st.spinner("正在训练与回测，请稍等..."):
                X, y = build_supervised_table(df, min_history=min_history)
                metrics, detail_df = walk_forward_backtest(
                    X, y, train_start_size=train_start_size
                )
                top10_df = predict_next_top10(df, min_history=min_history)

            st.success("分析完成")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Top-1", f"{metrics['top1']:.4f}")
            c2.metric("Top-5", f"{metrics['top5']:.4f}")
            c3.metric("Top-10", f"{metrics['top10']:.4f}")
            c4.metric("随机Top-10基准", f"{10/49:.4f}")

            st.subheader("下一期 Top-10 号码")
            st.dataframe(top10_df, use_container_width=True)

            st.subheader("推荐号码")
            st.markdown(
                f"""
                <div style="padding:16px;border-radius:12px;background:#f2f7ff;font-size:24px;font-weight:700;">
                {' / '.join(top10_df['号码'].tolist())}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("回测明细")
            st.dataframe(detail_df.tail(30), use_container_width=True)

            excel_bytes = to_excel_bytes(top10_df, detail_df)
            st.download_button(
                label="下载结果 Excel",
                data=excel_bytes,
                file_name="Top10预测结果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"运行失败：{e}")

st.markdown("---")
st.caption("说明：这是机器学习实验工具，用于学习时序特征、回测与概率排序。")
