import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO

# =========================
# 页面设置
# =========================
st.set_page_config(
    page_title="DPN vs Normal (SWE + Radiomics)",
    layout="wide",  # ✅ 横向宽屏
)

st.title("DPN vs Normal Prediction (SWE + Radiomics)")
st.caption("Based on duration, CSA, PCA features and radiomics features")

# =========================
# ✅ 基于脚本目录定位文件（兼容 Streamlit Cloud / Linux）
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "LGB.pkl"
DEFAULT_DATA_PATH = BASE_DIR / "剪切波数据_结合_val_select.xlsx"

# （可选）调试：查看云端目录文件
# st.write("BASE_DIR:", str(BASE_DIR))
# st.write("Files:", [p.name for p in BASE_DIR.iterdir()])

if not MODEL_PATH.exists():
    st.error(f"❌ 找不到模型文件：{MODEL_PATH.name}（请确认和 predictor.py 在同一目录）")
    st.stop()

if not DEFAULT_DATA_PATH.exists():
    st.error(f"❌ 找不到默认数据文件：{DEFAULT_DATA_PATH.name}（请确认和 predictor.py 在同一目录）")
    st.stop()

# =========================
# 载入模型 & 默认数据
# =========================
model = joblib.load(MODEL_PATH)
df_default = pd.read_excel(DEFAULT_DATA_PATH)

# 标签列名：如果是 target/label 二选一
label_col = None
for c in ["target", "label"]:
    if c in df_default.columns:
        label_col = c
        break

X_default = df_default.drop(columns=[label_col]) if label_col else df_default.copy()

# =========================
# 你的 22 个特征（必须与训练时一致）
# =========================
feature_names = [
    "duration",
    "CSA",
    "PCA_6",
    "PCA_61",
    "PCA_59",
    "PCA_27",
    "PCA_7",
    "PCA_12",
    "PCA_11",
    "PCA_14",
    "PCA_38",
    "PCA_2",
    "lbp-2D_firstorder_10Percentile",
    "original_firstorder_RootMeanSquared",
    "wavelet-LLH_firstorder_Variance",
    "original_glrlm_GrayLevelNonUniformity",
    "exponential_glrlm_RunEntropy",
    "original_gldm_LargeDependenceLowGrayLevelEmphasis",
    "wavelet-LLH_glrlm_ShortRunEmphasis",
    "exponential_glrlm_GrayLevelNonUniformity",
    "wavelet-LHL_glszm_SmallAreaHighGrayLevelEmphasis",
    "wavelet-HHL_glszm_LargeAreaEmphasis",
]

# ✅ 如果默认数据里列不全，直接提示（避免后面神秘报错）
missing = [c for c in feature_names if c not in X_default.columns]
if missing:
    st.error(f"❌ 默认数据缺少这些特征列：{missing}\n请确认 Excel 表头与模型训练一致。")
    st.stop()

# 只取这些列，并按 feature_names 排序（强制对齐列顺序）
X_default = X_default[feature_names]

# =========================
# 输入表单：✅ 3列横向布局
# =========================
with st.form("input_form"):
    st.subheader("请输入以下特征（可用默认中位数）")

    cols = st.columns(3)  # ✅ 3列横向
    inputs = {}

    for i, col in enumerate(feature_names):
        box = cols[i % 3]
        default_val = float(X_default[col].median())

        # 按特征设置更合理的输入格式
        if col == "duration":
            inputs[col] = box.number_input(
                col, value=float(default_val),
                min_value=0.0, max_value=1000.0, step=1.0
            )
        elif col == "CSA":
            inputs[col] = box.number_input(
                col, value=float(default_val),
                min_value=0.0, max_value=10.0, step=0.01, format="%.4f"
            )
        else:
            # PCA / Radiomics：可能正负都有
            inputs[col] = box.number_input(
                col, value=float(default_val),
                step=0.01, format="%.6f"
            )

    submitted = st.form_submit_button("Submit Prediction")

# =========================
# 工具函数：把 matplotlib 图保存到内存并展示
# =========================
def fig_to_bytesio(dpi=250):
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close()
    buf.seek(0)
    return buf

# =========================
# 预测 & 解释
# =========================
if submitted:
    # 强制列顺序对齐
    model_input = pd.DataFrame([inputs], columns=feature_names)

    # 顶部两栏：输入/结果
    left, right = st.columns([1.1, 1.0], gap="large")

    with left:
        st.subheader("Model Input Features")
        st.dataframe(model_input, use_container_width=True)

    with right:
        st.subheader("Prediction Result")

        # 预测概率（class=1）
        predicted_proba = model.predict_proba(model_input)[0]
        prob1 = float(predicted_proba[1])
        st.markdown(f"**Estimated probability (class=1 / DPN):** {prob1*100:.1f}%")

        # 分层（用默认数据分位数做阈值：三分法）
        y_probs = model.predict_proba(X_default)[:, 1]
        low_th = np.percentile(y_probs, 50)
        mid_th = np.percentile(y_probs, 88.07)

        st.caption(f"Thresholds (based on default data): 50%={low_th:.3f}, 88.07%={mid_th:.3f}")

        if prob1 <= low_th:
            st.success("🟢 Low risk / predicted as Normal tendency")
        elif prob1 <= mid_th:
            st.warning("🟡 Moderate risk")
        else:
            st.error("🔴 High risk / predicted as DPN tendency")

    st.divider()

    # ===== SHAP Force Plot（静态图）=====
    st.subheader("SHAP Force Plot (Local Explanation)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(model_input)

    # 二分类兼容：取 class=1 的 SHAP
    if isinstance(shap_values, list):
        shap_value_sample = shap_values[1][0]      # (n_features,)
        expected_value = explainer.expected_value[1]
    else:
        # 有些版本会直接返回 (n_samples, n_features)
        shap_value_sample = shap_values[0]
        expected_value = explainer.expected_value

    plt.figure(figsize=(12, 2.8))
    shap.force_plot(
        base_value=expected_value,
        shap_values=shap_value_sample,
        features=model_input.iloc[0],
        matplotlib=True,
        show=False
    )

    st.image(fig_to_bytesio(dpi=300), use_column_width=True)

    # ===== 可选：再给一张更“论文友好”的条形图（更稳、更好读）=====
    with st.expander("Show SHAP bar plot (recommended for paper)", expanded=True):
        # 按绝对值排序贡献
        contrib = pd.Series(shap_value_sample, index=feature_names)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:15]

        plt.figure(figsize=(8, 4.8))
        colors = ["tab:red" if v > 0 else "tab:blue" for v in top.values]
        plt.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
        plt.axvline(0, linewidth=1)
        plt.xlabel("SHAP value (impact on model output for class=1)")
        plt.title("Top-15 feature contributions (local)")
        st.image(fig_to_bytesio(dpi=250), use_column_width=False)
