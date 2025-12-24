import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# 页面标题
# =========================
st.title("DPN vs Normal Prediction (SWE + Radiomics)")
st.caption("Based on duration, CSA, PCA features and radiomics features")

# =========================
# 载入模型 & 用于默认值的数据（建议用训练集/验证集都行）
# =========================
MODEL_PATH = r"C:\Users\jiang123\Desktop\LGB.pkl"
DEFAULT_DATA_PATH = r"C:\Users\jiang123\Desktop\剪切波数据_结合_val_select.xlsx"  # 你也可改 train_select

model = joblib.load(MODEL_PATH)

# 读默认数据（Excel）
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
    st.error(f"默认数据缺少这些特征列：{missing}\n请确认 Excel 表头与模型训练一致。")
    st.stop()

# 只取这些列，并按 feature_names 排序（强制对齐列顺序）
X_default = X_default[feature_names]

# =========================
# 输入表单
# =========================
with st.form("input_form"):
    st.subheader("请输入以下特征（可用默认中位数）")

    inputs = {}
    for col in feature_names:
        default_val = float(X_default[col].median())

        # 你可以按需求自定义范围/步长，这里给通用设置
        if col in ["duration"]:
            inputs[col] = st.number_input(col, value=float(default_val), min_value=0.0, max_value=1000.0, step=1.0)
        elif col in ["CSA"]:
            inputs[col] = st.number_input(col, value=float(default_val), min_value=0.0, max_value=10.0, step=0.01, format="%.4f")
        else:
            # PCA / radiomics：可能正负都有，范围给宽一点
            inputs[col] = st.number_input(col, value=float(default_val), step=0.01, format="%.6f")

    submitted = st.form_submit_button("Submit Prediction")

# =========================
# 预测 & 解释
# =========================
if submitted:
    model_input = pd.DataFrame([inputs], columns=feature_names)

    st.subheader("Model Input Features")
    st.dataframe(model_input)

    # 预测概率（class=1）
    predicted_proba = model.predict_proba(model_input)[0]
    probability = predicted_proba[1] * 100
    st.subheader("Prediction Result")
    st.markdown(f"**Estimated probability (class=1):** {probability:.1f}%")

    # ===== 分层（用默认数据分位数做阈值：这里给你一个简单三分法，可自己改）=====
    y_probs = model.predict_proba(X_default)[:, 1]
    low_th = np.percentile(y_probs, 50)
    mid_th = np.percentile(y_probs, 88.07)

    if predicted_proba[1] <= low_th:
        st.success("🟢 Low risk / predicted as Normal tendency")
    elif predicted_proba[1] <= mid_th:
        st.warning("🟡 Moderate risk")
    else:
        st.error("🔴 High risk / predicted as DPN tendency")

    # ===== SHAP Force Plot（静态图）=====
    st.subheader("SHAP Force Plot (Local Explanation)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(model_input)

    # 二分类兼容
    if isinstance(shap_values, list):
        shap_value_sample = shap_values[1][0]      # (n_features,)
        expected_value = explainer.expected_value[1]
    else:
        shap_value_sample = shap_values[0]
        expected_value = explainer.expected_value

    # 画 force plot 并保存为图片，再展示
    plt.figure()
    shap.force_plot(
        base_value=expected_value,
        shap_values=shap_value_sample,
        features=model_input.iloc[0],
        matplotlib=True,
        show=False
    )

    out_png = "shap_force_plot.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close()

    st.image(out_png)
