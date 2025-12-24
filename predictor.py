with st.form("input_form"):
    st.subheader("请输入以下特征（可用默认中位数）")

    # ===== 分组 =====
    group_clinical = ["duration", "CSA"]

    group_dl = [
        "PCA_6","PCA_61","PCA_59","PCA_27","PCA_7",
        "PCA_12","PCA_11","PCA_14","PCA_38","PCA_2"
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

    inputs = {}

    # ===== ① 临床/超声（2列）=====
    st.markdown("### ① Clinical & Ultrasound")
    c1, c2 = st.columns(2)

    for col, box in zip(group_clinical, [c1, c2]):
        default_val = float(X_default[col].median())
        if col == "duration":
            inputs[col] = box.number_input(col, value=default_val, min_value=0.0, max_value=1000.0, step=1.0)
        else:  # CSA
            inputs[col] = box.number_input(col, value=default_val, min_value=0.0, max_value=10.0,
                                           step=0.01, format="%.4f")

    # ===== ② 深度学习特征（PCA）（4列更紧凑）=====
    st.markdown("### ② Deep Learning Features (PCA)")
    cols_dl = st.columns(4)
    for i, col in enumerate(group_dl):
        box = cols_dl[i % 4]
        default_val = float(X_default[col].median())
        inputs[col] = box.number_input(col, value=default_val, step=0.01, format="%.6f")

    # ===== ③ 影像组学特征（4列）=====
    st.markdown("### ③ Radiomics Features")
    cols_rad = st.columns(4)
    for i, col in enumerate(group_radiomics):
        box = cols_rad[i % 4]
        default_val = float(X_default[col].median())
        inputs[col] = box.number_input(col, value=default_val, step=0.01, format="%.6f")

    submitted = st.form_submit_button("Submit Prediction")

