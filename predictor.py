import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO

# =========================
# 页面设置（✅ 已改网站名称）
# =========================
st.set_page_config(
    page_title="A Multimodal Imaging Prediction Model Integrating Shear Wave Elastography for Diabetic Peripheral Neuropathy",
    layout="wide",
)

st.title("A Multimodal Imaging Prediction Model Integrating Shear Wave Elastography for Diabetic Peripheral Neuropathy")
st.caption("Clinical/Ultrasound + Radiomics + Deep learning features (PCA)")

# =========================
# ✅ 基于脚本目录定位文件（兼容 Streamlit Cloud / Linux）
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "LGB.pkl"
DEFAULT_DATA_PATH = BASE_DIR / "剪切波数据_结合_val_select.xlsx"

if not MODEL_PATH.exists():
    st.error(f"❌ 找不到模型文件：{MODEL_PATH.name}（请确认它与 predictor.py 在同一目录）")
    st.stop()

if not DEFAULT_DATA_PATH.exists():
    st.error(f"❌ 找不到默认数据文件：{DEFAULT_DATA_PATH.name}（请确认它与 predictor.py 在同一目录）")
    st.stop()

# =========================
# 载入模型 & 默认数据
# =========================
model = joblib.load(MODEL_PATH)

try:
    df_default = pd.read_excel(DEFAULT_DATA_PATH)
except Exception as e:
    st.error(f"❌ 读取默认数据失败：{DEFAULT_DATA_PATH.name}\n\n错误信息：{e}")
    st.stop()

label_col = None
for c in ["target", "label"]:
    if c in df_default.columns:
        label_col = c
        break

X_default = df_default.drop(columns=[label_col]) if label_col else df_default.copy()

# =========================
# ✅ 三大类特征分组
# =========================
group_clinical = ["duration", "CSA"]

group_dl = [
    "PCA_6", "PCA_61", "PCA_59", "PCA_27", "PCA_7",
    "PCA_12", "PCA_11", "PCA_14", "PCA_38", "PCA_2"
]

group_radiomics = [
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

# ✅ 模型输入特征顺序（⚠️ 不改，保证与训练一致）
feature_names = group_clinical + group_dl + group_radiomics

missing = [c for c in feature_names if c not in X_default.columns]
if missing:
    st.error(f"❌ 默认数据缺少这些特征列：{missing}\n请确认 Excel 表头与模型训练一致。")
    st.stop()

X_default = X_default[feature_names]

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
# ✅ 输入表单：交换 Radiomics 和 PCA 的显示位置
# =========================
with st.form("input_form"):
    st.subheader("Please input the following features")
    inputs = {}

    # ===== ① 临床/超声：2列 =====
    st.markdown("### ① Clinical & Ultrasound")
    c1, c2 = st.columns(2)

    for col, box in zip(group_clinical, [c1, c2]):
        default_val = float(X_default[col].median())
        if col == "duration":
            inputs[col] = box.number_input(
                "duration (month)",
                value=float(default_val),
                min_value=0.0,
                max_value=1000.0,
                step=1.0
            )
        else:  # CSA
            inputs[col] = box.number_input(
                "CSA",
                value=float(default_val),
                min_value=0.0,
                max_value=10.0,
                step=0.01,
                format="%.4f"
            )

    # ===== ② Radiomics Features：4列（✅ 提前）=====
    st.markdown("### ② Radiomics Features")
    cols_rad = st.columns(4)
    for i, col in enumerate(group_radiomics):
        box = cols_rad[i % 4]
        default_val = float(X_default[col].median())
        inputs[col] = box.number_input(
            col,
            value=float(default_val),
            step=0.01,
            format="%.6f"
        )

    # ===== ③ Deep Learning Features (PCA)：4列（✅ 放到后面）=====
    st.markdown("### ③ Deep Learning Features (PCA)")
    cols_dl = st.columns(4)
    for i, col in enumerate(group_dl):
        box = cols_dl[i % 4]
        default_val = float(X_default[col].median())
        inputs[col] = box.number_input(
            col,
            value=float(default_val),
            step=0.01,
            format="%.6f"
        )

    submitted = st.form_submit_button("Submit Prediction")

# =========================
# 预测 & 解释
# =========================
if submitted:
    # ⚠️ 仍然按 feature_names（Clinical + PCA + Radiomics）喂给模型
    model_input = pd.DataFrame([inputs], columns=feature_names)

    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.subheader("Model Input Features")
        st.dataframe(model_input, use_container_width=True)

    with right:
        st.subheader("Prediction Result")

        predicted_proba = model.predict_proba(model_input)[0]
        prob1 = float(predicted_proba[1])

        st.markdown(f"**Estimated probability (class=1 / DPN):** {prob1*100:.1f}%")

        y_probs = model.predict_proba(X_default)[:, 1]
        low_th = np.percentile(y_probs, 50)
        mid_th = np.percentile(y_probs, 88.07)

        st.caption(f"Thresholds based on default set: 50%={low_th:.3f}, 88.07%={mid_th:.3f}")

        if prob1 <= low_th:
            st.success("🟢 Normal tendency (Low risk)")
        elif prob1 <= mid_th:
            st.warning("🟡 Intermediate")
        else:
            st.error("🔴 DPN tendency (High risk)")

    st.divider()

    # =========================
    # SHAP force plot（静态图）
    # =========================
    st.subheader("SHAP Force Plot")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(model_input)

    if isinstance(shap_values, list):
        shap_value_sample = shap_values[1][0]
        expected_value = explainer.expected_value[1]
    else:
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

    # =========================
    # Top SHAP bar
    # =========================
    with st.expander("Show SHAP bar plot (recommended for paper)", expanded=True):
        contrib = pd.Series(shap_value_sample, index=feature_names)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:15]

        plt.figure(figsize=(9, 5))
        colors = ["tab:red" if v > 0 else "tab:blue" for v in top.values]
        plt.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
        plt.axvline(0, linewidth=1)
        plt.xlabel("SHAP value (impact on model output for class=1 / DPN)")
        plt.title("Top-15 feature contributions (local)")
        st.image(fig_to_bytesio(dpi=250), use_container_width=False)

    # =========================
    # ✅ 分组汇总贡献（顺序也按页面显示：Clinical -> Radiomics -> PCA）
    # =========================
    with st.expander("Show grouped contribution summary (Clinical vs Radiomics vs Deep Learning)", expanded=False):
        shap_abs = np.abs(shap_value_sample)
        s = pd.Series(shap_abs, index=feature_names)

        # 注意：feature_names 仍是 Clinical + PCA + Radiomics
        grp = pd.DataFrame({
            "Group": (["Clinical&US"] * len(group_clinical)) +
                     (["DeepLearning(PCA)"] * len(group_dl)) +
                     (["Radiomics"] * len(group_radiomics)),
            "Feature": feature_names,
            "AbsSHAP": s.values
        })

        # ✅ 为了显示顺序与页面一致，这里重排
        order = ["Clinical&US", "Radiomics", "DeepLearning(PCA)"]
        grp_sum = grp.groupby("Group")["AbsSHAP"].sum().reindex(order)

        st.write("Sum of |SHAP| by group (local):")
        st.dataframe(grp_sum.reset_index().rename(columns={"AbsSHAP": "Sum(|SHAP|)"}), use_container_width=True)




