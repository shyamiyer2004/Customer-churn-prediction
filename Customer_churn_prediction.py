# Customer_churn_prediction_streamlit.py
import os
import warnings
warnings.filterwarnings("ignore", "ignore")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import streamlit as st

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
# Optional SMOTE import (safe for Streamlit Cloud)
SMOTE = None
try:
    from imblearn.over_sampling import SMOTE
except Exception as e:
    SMOTE = None
    st.warning("imbalanced-learn (SMOTE) not available in this environment; continuing without SMOTE.")

# Try to import TargetEncoder; fallback to OrdinalEncoder if not available
try:
    from category_encoders import TargetEncoder
    CE_AVAILABLE = True
except Exception:
    CE_AVAILABLE = False

# ----------------------------
# Paths (use files from same folder as script)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_XGB_PATH = os.path.join(BASE_DIR, "best_xgboost_model.pkl")
MODEL_RF_PATH  = os.path.join(BASE_DIR, "best_random_forest_model.pkl")
ENCODER_PATH   = os.path.join(BASE_DIR, "target_encoder.pkl")
SCALER_PATH    = os.path.join(BASE_DIR, "scaler.pkl")
SAMPLE_PATH    = os.path.join(BASE_DIR, "sample_customers.xlsx")
DATA_PATH      = os.path.join(BASE_DIR, "Telco_customer_churn.xlsx")


# ----------------------------
# Helpers: encoder / scaler (robust)
# ----------------------------
def fit_encoder_and_scaler(X_df, y_series=None):
    """
    Fit encoder (TargetEncoder or OrdinalEncoder) on categorical columns
    and StandardScaler on entire (possibly encoded) dataframe. Attach
    feature list to scaler for later alignment.
    """
    X = X_df.copy()
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    encoder_local = None
    if CE_AVAILABLE and len(cat_cols) > 0:
        encoder_local = TargetEncoder(cols=cat_cols)
        encoder_local.fit(X, y_series if y_series is not None else pd.Series(np.zeros(len(X))))
        try:
            encoder_local.feature_cols = list(cat_cols)
        except Exception:
            pass
    elif len(cat_cols) > 0:
        encoder_local = OrdinalEncoder()
        X[cat_cols] = X[cat_cols].fillna("NA")
        encoder_local.fit(X[cat_cols])
        try:
            encoder_local.feature_cols = list(encoder_local.feature_names_in_)
        except Exception:
            encoder_local.feature_cols = list(cat_cols)

    # transform categorical columns for scaler fitting (best-effort)
    X_enc = X.copy()
    if encoder_local is not None and len(cat_cols) > 0:
        try:
            transformed = _safe_encoder_transform_for_fit(X_enc, encoder_local)
            if isinstance(transformed, np.ndarray):
                X_enc[cat_cols] = transformed
            else:
                X_enc[cat_cols] = pd.DataFrame(transformed, columns=cat_cols, index=X_enc.index)
        except Exception as e:
            print("Warning: encoder.transform during fit failed:", e)

    scaler_local = StandardScaler()
    scaler_local.fit(X_enc)
    try:
        scaler_local.feature_names = list(X_enc.columns)
    except Exception:
        scaler_local.feature_names = list(X_enc.columns)

    try:
        joblib.dump(encoder_local, ENCODER_PATH)
    except Exception as e:
        print("Warning: couldn't save encoder:", e)
    try:
        joblib.dump(scaler_local, SCALER_PATH)
    except Exception as e:
        print("Warning: couldn't save scaler:", e)

    print("Fitted encoder (cols):", getattr(encoder_local, "feature_cols", None))
    print("Fitted scaler on columns:", scaler_local.feature_names)
    return encoder_local, scaler_local

def _safe_encoder_transform_for_fit(Xframe, encoder):
    """
    Used during fit: ensure encoder.transform receives expected columns.
    """
    enc_cols = getattr(encoder, "cols", None) or getattr(encoder, "feature_cols", None)
    if enc_cols is not None:
        df_sub = pd.DataFrame(index=Xframe.index)
        for c in enc_cols:
            if c in Xframe.columns:
                df_sub[c] = Xframe[c].fillna("NA")
            else:
                df_sub[c] = "NA"
        return encoder.transform(df_sub)
    if hasattr(encoder, "feature_names_in_"):
        enc_cols = list(encoder.feature_names_in_)
        df_sub = pd.DataFrame(index=Xframe.index)
        for c in enc_cols:
            if c in Xframe.columns:
                df_sub[c] = Xframe[c].fillna("NA")
            else:
                df_sub[c] = "NA"
        return encoder.transform(df_sub)
    cat_cols = Xframe.select_dtypes(include='object').columns.tolist()
    if len(cat_cols) > 0:
        df_sub = Xframe[cat_cols].fillna("NA")
        return encoder.transform(df_sub)
    return Xframe

def _encode_df_with_encoder(df_in, encoder):
    """
    Return a copy of df_in where categorical columns defined by encoder
    are transformed to numeric (if encoder is provided). Ensures encoder
    receives expected fitted columns.
    """
    Xc = df_in.copy()
    if encoder is None:
        return Xc

    enc_cols = getattr(encoder, "cols", None) or getattr(encoder, "feature_cols", None)
    if enc_cols is not None:
        df_sub = pd.DataFrame(index=Xc.index)
        for c in enc_cols:
            if c in Xc.columns:
                df_sub[c] = Xc[c].fillna("NA")
            else:
                df_sub[c] = "NA"
        transformed = encoder.transform(df_sub)
        if isinstance(transformed, np.ndarray):
            for i, c in enumerate(enc_cols):
                Xc[c] = transformed[:, i]
        else:
            Xc.loc[:, enc_cols] = pd.DataFrame(transformed, columns=enc_cols, index=Xc.index)
        return Xc

    if hasattr(encoder, "feature_names_in_"):
        enc_cols = list(encoder.feature_names_in_)
        df_sub = pd.DataFrame(index=Xc.index)
        for c in enc_cols:
            if c in Xc.columns:
                df_sub[c] = Xc[c].fillna("NA")
            else:
                df_sub[c] = "NA"
        transformed = encoder.transform(df_sub)
        if isinstance(transformed, np.ndarray):
            for i, c in enumerate(enc_cols):
                Xc[c] = transformed[:, i]
        else:
            Xc.loc[:, enc_cols] = pd.DataFrame(transformed, columns=enc_cols, index=Xc.index)
        return Xc

    cat_cols = Xc.select_dtypes(include='object').columns.tolist()
    if len(cat_cols) > 0:
        sub = Xc[cat_cols].fillna("NA")
        transformed = encoder.transform(sub)
        if isinstance(transformed, np.ndarray):
            Xc[cat_cols] = transformed
        else:
            Xc[cat_cols] = pd.DataFrame(transformed, columns=cat_cols, index=Xc.index)
    return Xc

def apply_encoder_and_scaler(X_df, encoder, scaler, training_df=None, training_y=None):
    """
    Robustly transform X_df to the same feature-space/shape the scaler expects.
    """
    X = X_df.copy()
    try:
        X_enc = _encode_df_with_encoder(X, encoder) if encoder is not None else X.copy()
    except Exception as e:
        print("Transform error during encoder.transform:", e)
        X_enc = X.copy()
        cat_cols = X_enc.select_dtypes(include='object').columns.tolist()
        for c in cat_cols:
            try:
                X_enc[c] = X_enc[c].astype('category').cat.codes
            except Exception:
                X_enc[c] = X_enc[c].astype(str).fillna("NA").astype('category').cat.codes

    if 'Total Charges' in X_enc.columns:
        X_enc['Total Charges'] = pd.to_numeric(X_enc['Total Charges'], errors='coerce')
        if 'Tenure Months' in X_enc.columns and 'Monthly Charges' in X_enc.columns:
            X_enc['Total Charges'] = X_enc['Total Charges'].fillna(X_enc['Tenure Months'] * X_enc['Monthly Charges'])

    expected = getattr(scaler, "feature_names", None)
    if expected is None:
        expected = list(X_enc.columns)

    X_for_scaler = pd.DataFrame(index=X_enc.index)
    extra_cols = [c for c in X_enc.columns if c not in expected]
    for col in expected:
        if col in X_enc.columns:
            X_for_scaler[col] = X_enc[col]
        else:
            fill_val = 0.0
            if training_df is not None and col in training_df.columns:
                try:
                    if training_df[col].dtype == 'object' and encoder is not None:
                        tr = _encode_df_with_encoder(pd.DataFrame({col: training_df[col]}), encoder)
                        fill_val = float(tr[col].median(skipna=True))
                    else:
                        fill_val = float(pd.to_numeric(training_df[col], errors='coerce').median(skipna=True))
                        if np.isnan(fill_val):
                            fill_val = 0.0
                except Exception:
                    fill_val = 0.0
            X_for_scaler[col] = fill_val

    if extra_cols:
        for c in extra_cols:
            print("Dropping extra column not in scaler expectation:", c)

    X_for_scaler = X_for_scaler[expected]

    try:
        scaled_arr = scaler.transform(X_for_scaler)
        X_scaled = pd.DataFrame(scaled_arr, columns=expected, index=X_for_scaler.index)
        return X_scaled
    except ValueError as e:
        print("Transform error:", e)
        if training_df is not None:
            try:
                print("Attempting to re-fit encoder+scaler using provided training data...")
                new_encoder, new_scaler = fit_encoder_and_scaler(training_df[expected], training_y)
                try:
                    joblib.dump(new_encoder, ENCODER_PATH)
                    joblib.dump(new_scaler, SCALER_PATH)
                except Exception:
                    pass
                return apply_encoder_and_scaler(X_df, new_encoder, new_scaler, training_df=None, training_y=None)
            except Exception as ex:
                print("Refit attempt failed:", ex)
        print("Final fallback: ordinal codes and zeros for missing columns.")
        X_fb = X_df.copy()
        cat_cols = X_fb.select_dtypes(include='object').columns.tolist()
        for c in cat_cols:
            try:
                X_fb[c] = X_fb[c].astype('category').cat.codes
            except Exception:
                X_fb[c] = X_fb[c].astype(str).fillna("NA").astype('category').cat.codes
        if 'Total Charges' in X_fb.columns:
            X_fb['Total Charges'] = pd.to_numeric(X_fb['Total Charges'], errors='coerce')
            if 'Tenure Months' in X_fb.columns and 'Monthly Charges' in X_fb.columns:
                X_fb['Total Charges'] = X_fb['Total Charges'].fillna(X_fb['Tenure Months'] * X_fb['Monthly Charges'])
        X_fb2 = pd.DataFrame(index=X_fb.index)
        for col in expected:
            if col in X_fb.columns:
                X_fb2[col] = X_fb[col]
            else:
                X_fb2[col] = 0.0
        X_fb2 = X_fb2[expected]
        scaled_arr = scaler.transform(X_fb2)
        return pd.DataFrame(scaled_arr, columns=expected, index=X_fb2.index)

# ----------------------------
# Small helpers for analytics charts
# ----------------------------
def tenure_buckets(tenure_months):
    bins = [0, 3, 6, 12, 24, 48, np.inf]
    labels = ['0-3','4-6','7-12','13-24','25-48','49+']
    return pd.cut(tenure_months, bins=bins, labels=labels, include_lowest=True)

# ----------------------------
# Explanation helper
# ----------------------------
def compute_explanation_for_row(x_row, feature_means, feature_stds, feature_importances, feature_names, top_k=3):
    import numpy as _np
    x = _np.asarray(x_row, dtype=float).flatten()
    fm = _np.asarray(feature_means, dtype=float).flatten()
    fs = _np.asarray(feature_stds, dtype=float).flatten()
    fi = _np.asarray(feature_importances, dtype=float).flatten()
    n = min(len(x), len(fm), len(fs), len(fi), len(feature_names))
    if n == 0:
        return []
    x = x[:n]; fm = fm[:n]; fs = fs[:n]; fi = fi[:n]; names = list(feature_names)[:n]
    std_safe = _np.where(fs == 0, 1e-6, fs)
    z = (x - fm) / std_safe
    contribs = z * fi
    idx = _np.argsort(contribs)[::-1][:top_k]
    result = [(names[i], float(contribs[i])) for i in idx]
    return result

# ----------------------------
# Load data & artifacts
# ----------------------------
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("📊 Telco Customer Churn — Analytics & Predictions")

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at: {DATA_PATH}")
    st.stop()

raw_df = pd.read_excel(DATA_PATH)
df = raw_df.copy()

# Basic cleaning (matching earlier pipeline)
for c in ['CustomerID', 'Count', 'Country', 'State', 'Lat Long', 'Churn Label','Churn Score']:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)
if 'Total Charges' in df.columns:
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    if 'Tenure Months' in df.columns and 'Monthly Charges' in df.columns:
        df['Total Charges'] = df['Total Charges'].fillna(df['Tenure Months'] * df['Monthly Charges'])

columns_replace = ['Multiple Lines','Online Security','Online Backup','Device Protection','Tech Support','Streaming TV','Streaming Movies']
for c in columns_replace:
    if c in df.columns:
        df.loc[~df[c].isin(['No','Yes']), c] = 'No'

# Model features exclude Churn Value, City, Latitude, Longitude, and keep Churn Reason only for analytics
model_feature_cols = [c for c in df.columns if c not in ['Churn Value','City','Latitude','Longitude','Churn Reason']]

# load or fit encoder & scaler
encoder = None; scaler = None
if os.path.exists(ENCODER_PATH) and os.path.exists(SCALER_PATH):
    try:
        encoder = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
        if not hasattr(scaler, "feature_names"):
            scaler.feature_names = model_feature_cols
            print("Attached feature_names to loaded scaler from current model_feature_cols")
    except Exception as e:
        print("Failed to load encoder/scaler - refitting. Error:", e)
        encoder, scaler = fit_encoder_and_scaler(df[model_feature_cols], df.get('Churn Value', None))
else:
    encoder, scaler = fit_encoder_and_scaler(df[model_feature_cols], df.get('Churn Value', None))

# load model
model = None
model_type = None
if os.path.exists(MODEL_XGB_PATH):
    try:
        model = joblib.load(MODEL_XGB_PATH)
        model_type = "xgboost"
    except Exception as e:
        print("Failed to load XGB model:", e)
if model is None and os.path.exists(MODEL_RF_PATH):
    try:
        model = joblib.load(MODEL_RF_PATH)
        model_type = "random_forest"
    except Exception as e:
        print("Failed to load RF model:", e)

# Precompute scaled X for analytics/explanations
X_for_scaling = df[model_feature_cols].copy()
X_scaled_full = apply_encoder_and_scaler(X_for_scaling, encoder, scaler, training_df=X_for_scaling, training_y=df.get('Churn Value', None))
feature_means = X_scaled_full.mean(axis=0).values
feature_stds  = X_scaled_full.std(axis=0).values

if model is None:
    st.warning("No trained model file found at configured paths. Prediction will be disabled until a model file is present.")
else:
    st.success(f"Loaded model type: {model_type}")

# ----------------------------
# UI: Tabs
# ----------------------------
tab1, tab2 = st.tabs(["📈 Analytics", "🤖 Prediction"])

# ----- Analytics tab -----
with tab1:
    st.header("Trendy Analytics & Insights")
    st.markdown("Use filters to explore segments. Hover any chart for details. Expand the info boxes to read why a chart matters.")

    # Filters
    f1, f2, f3 = st.columns([1,1,1])
    with f1:
        contract_filter = st.selectbox("Contract", ["All"] + sorted(df['Contract'].dropna().unique().tolist()))
    with f2:
        internet_filter = st.selectbox("Internet Service", ["All"] + sorted(df['Internet Service'].dropna().unique().tolist()))
    with f3:
        tenure_filter = st.selectbox("Tenure bucket", ["All",'0-3','4-6','7-12','13-24','25-48','49+'])

    df_f = df.copy()
    if contract_filter != "All":
        df_f = df_f[df_f['Contract'] == contract_filter]
    if internet_filter != "All":
        df_f = df_f[df_f['Internet Service'] == internet_filter]
    if 'Tenure Months' in df_f.columns:
        df_f['Tenure Bucket'] = tenure_buckets(df_f['Tenure Months'])
        if tenure_filter != "All":
            df_f = df_f[df_f['Tenure Bucket'] == tenure_filter]

    total_customers = len(df_f)
    churned_count = int(df_f['Churn Value'].sum()) if 'Churn Value' in df_f.columns else 0
    churn_rate = churned_count / total_customers * 100 if total_customers>0 else 0.0

    k1, k2, k3 = st.columns(3)
    k1.metric("Total customers", f"{total_customers:,}")
    k2.metric("Churned customers", f"{churned_count:,}")
    k3.metric("Churn rate", f"{churn_rate:.2f}%")

    # 2x2 grid of charts (Plotly)
    cA, cB = st.columns(2)
    cC, cD = st.columns(2)

    # Chart A: Churn Reasons pie + bar
    with cA:
        st.subheader("Churn Reasons (share & top causes)")
        if 'Churn Reason' in df_f.columns:
            reasons = df_f[df_f['Churn Value'] == 1]['Churn Reason'].fillna("Other")
            reason_counts = reasons.value_counts().reset_index()
            reason_counts.columns = ['Reason','Count']
            fig = px.pie(reason_counts, names='Reason', values='Count', hole=0.35, template='plotly_white',
                         title="Churn Reasons (churned customers)")
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

            topn = reason_counts.head(8)
            fig2 = px.bar(topn.sort_values('Count'), x='Count', y='Reason', orientation='h',
                          title='Top Churn Reasons', template='plotly_white')
            st.plotly_chart(fig2, use_container_width=True)
            with st.expander("Why this chart matters"):
                st.write("Shows top reasons customers left — use it to prioritize retention fixes.")
        else:
            st.info("No 'Churn Reason' column available in dataset.")

    # Chart B: Churn Reasons by Contract (stacked)
    with cB:
        st.subheader("Churn Reasons by Contract")
        if 'Churn Reason' in df_f.columns:
            df_stack = df_f[df_f['Churn Value'] == 1].copy()
            df_stack['Churn Reason'] = df_stack['Churn Reason'].fillna("Other")
            top_reasons = df_stack['Churn Reason'].value_counts().nlargest(8).index.tolist()
            df_stack['Reason Grouped'] = df_stack['Churn Reason'].apply(lambda x: x if x in top_reasons else 'Other')
            stacked = df_stack.groupby(['Contract','Reason Grouped']).size().reset_index(name='Count')
            fig = px.bar(stacked, x='Contract', y='Count', color='Reason Grouped', barmode='stack',
                         title='Churn Reasons by Contract', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Why this chart matters"):
                st.write("Identifies which contract types lose customers for which reasons.")
        else:
            st.info("No 'Churn Reason' available.")

    # Chart C: Churn timeline by Tenure
    with cC:
        st.subheader("Churn timeline by Tenure months")
        if 'Tenure Months' in df_f.columns:
            timeline = df_f.groupby('Tenure Months')['Churn Value'].agg(['sum','count']).reset_index()
            timeline['Churn Rate'] = timeline['sum'] / timeline['count']
            timeline['Churn Rate Smooth'] = timeline['Churn Rate'].rolling(3, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=timeline['Tenure Months'], y=timeline['sum'], name='Churn Count', marker_color='indianred', opacity=0.7))
            fig.add_trace(go.Scatter(x=timeline['Tenure Months'], y=timeline['Churn Rate Smooth']*timeline['count'].max(),
                                     name='Churn rate (smoothed, scaled)', yaxis='y2', line=dict(color='darkblue')))
            fig.update_layout(title="Churn count and smoothed churn rate across tenure months",
                              xaxis_title="Tenure Months",
                              yaxis=dict(title="Churn Count"),
                              yaxis2=dict(title="Churn Rate (scaled)", overlaying='y', side='right'),
                              template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Why this chart matters"):
                st.write("Shows spikes in churn across tenure — helps spot vulnerable tenure points.")
        else:
            st.info("No 'Tenure Months' column found.")

    # Chart D: Churn rate heatmap (Tenure bucket × Monthly Charges quartile)
    with cD:
        st.subheader("Churn rate heatmap by Tenure bucket × Monthly Charges quartile")
        if 'Tenure Months' in df_f.columns and 'Monthly Charges' in df_f.columns:
            df_f['Tenure Bucket'] = tenure_buckets(df_f['Tenure Months'])
            df_f['Monthly Q'] = pd.qcut(df_f['Monthly Charges'].rank(method='first'), 4, labels=['Q1','Q2','Q3','Q4'])
            pivot = df_f.groupby(['Tenure Bucket','Monthly Q'])['Churn Value'].mean().reset_index()
            pivot_table = pivot.pivot(index='Tenure Bucket', columns='Monthly Q', values='Churn Value').fillna(0)
            fig = px.imshow(pivot_table.values, x=pivot_table.columns.tolist(), y=pivot_table.index.tolist(),
                            labels=dict(x="Monthly Quartile", y="Tenure Bucket", color="Churn Rate"),
                            color_continuous_scale='RdYlGn_r', title="Churn rate heatmap", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Why this chart matters"):
                st.write("Find segments (price × tenure) with highest churn — useful for pricing or loyalty actions.")
        else:
            st.info("Need Tenure Months and Monthly Charges for this chart.")

# ----- Prediction tab -----
with tab2:
    st.header("Predict churn — Single & Bulk")
    st.write("Single: input values for one customer. Bulk: upload a file or use sample; results shown as table + analytics.")

    # Single prediction (kept concise)
    st.subheader("Single customer prediction")
    with st.form("single_form"):
        prefer = ['Gender','Senior Citizen','Partner','Dependents','Tenure Months','Contract','Monthly Charges','Total Charges','Internet Service','Payment Method']
        used = [p for p in prefer if p in model_feature_cols]
        remaining = [c for c in model_feature_cols if c not in used]
        colsL, colsR = st.columns(2)
        input_vals = {}
        for i, feat in enumerate(used):
            if feat in df.columns and df[feat].dtype == 'object':
                input_vals[feat] = colsL.selectbox(feat, df[feat].dropna().unique().tolist()) if (i%2==0) else colsR.selectbox(feat, df[feat].dropna().unique().tolist())
            else:
                default = float(df[feat].median()) if feat in df.columns else 0.0
                input_vals[feat] = colsL.number_input(feat, value=default) if (i%2==0) else colsR.number_input(feat, value=default)
        with st.expander("Edit additional features (optional)"):
            for feat in remaining[:8]:
                if feat in df.columns and df[feat].dtype == 'object':
                    input_vals[feat] = st.selectbox(feat, df[feat].dropna().unique().tolist(), key="add_"+feat)
                else:
                    default = float(df[feat].median()) if feat in df.columns else 0.0
                    input_vals[feat] = st.number_input(feat, value=default, key="addn_"+feat)
        submit_single = st.form_submit_button("Predict single")

    if submit_single:
        if model is None:
            st.error("No trained model file found. Place it in the configured model path.")
        else:
            input_df = pd.DataFrame([input_vals], columns=model_feature_cols)
            X_in_scaled = apply_encoder_and_scaler(input_df, encoder, scaler, training_df=X_for_scaling, training_y=df.get('Churn Value', None))
            prob = model.predict_proba(X_in_scaled.values)[0,1] if hasattr(model, "predict_proba") else float(model.predict(X_in_scaled.values)[0])
            st.metric("Predicted churn probability", f"{prob:.2%}")
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                x_scaled = X_in_scaled.values.flatten()
                std_safe = np.where(feature_stds == 0, 1e-6, feature_stds)
                z = (x_scaled - feature_means) / std_safe
                contribs = z * fi
                idx = np.argsort(contribs)[::-1][:4]
                st.subheader("Top contributing features (approx.)")
                for ii in idx:
                    st.write(f"{X_in_scaled.columns[ii]}: {contribs[ii]:.3f}")

    # ----------------------------
    # Bulk prediction UI (persistent with session_state)
    # ----------------------------
    st.markdown("---")
    st.subheader("Bulk prediction — upload or sample")

    # init session state keys (one-time)
    if 'bulk_df' not in st.session_state:
        st.session_state['bulk_df'] = None
    if 'bulk_source' not in st.session_state:
        st.session_state['bulk_source'] = None  # 'uploaded' or 'sample' or None

    # file uploader (keeps its own state via key)
    uploaded_file = st.file_uploader("Upload CSV/XLSX with feature columns", type=['csv','xlsx'], key='bulk_file')

    # Sample button: load sample into session_state so it persists across reruns
    if st.button("Use sample_customers.xlsx (if exists)"):
        if os.path.exists(SAMPLE_PATH):
            try:
                df_sample = pd.read_excel(SAMPLE_PATH)
                st.session_state['bulk_df'] = df_sample
                st.session_state['bulk_source'] = 'sample'
                st.success(f"Loaded sample ({len(df_sample)} rows) into session.")
            except Exception as e:
                st.error(f"Failed to load sample file: {e}")
        else:
            st.error("Sample not found. Place the sample_customers.xlsx at the configured SAMPLE_PATH or upload a file.")

    # If a user uploaded a file through the uploader, read it and store in session_state
    if uploaded_file is not None:
        try:
            if getattr(uploaded_file, "name", "").lower().endswith('.csv'):
                df_up = pd.read_csv(uploaded_file)
            else:
                df_up = pd.read_excel(uploaded_file)
            st.session_state['bulk_df'] = df_up
            st.session_state['bulk_source'] = 'uploaded'
            st.success(f"Uploaded and loaded {len(df_up)} rows into session.")
        except Exception as e:
            st.error("Failed to read uploaded file: " + str(e))

    # Provide a small UI to clear loaded data (useful for debugging)
    clear_col1, clear_col2 = st.columns([1,4])
    with clear_col1:
        if st.button("Clear loaded batch"):
            st.session_state['bulk_df'] = None
            st.session_state['bulk_source'] = None
            st.experimental_rerun()
    with clear_col2:
        if st.session_state['bulk_df'] is not None:
            st.info(f"Batch loaded from: {st.session_state['bulk_source']} (rows: {len(st.session_state['bulk_df'])})")

    # Set local variable bulk_df from session_state (persisted across reruns)
    bulk_df = st.session_state.get('bulk_df', None)

    # If there is a loaded DF, preview & allow running predictions
    if bulk_df is not None:
        missing = [c for c in model_feature_cols if c not in bulk_df.columns]
        if missing:
            st.warning(f"Loaded data is missing required columns: {missing}")
            st.write("Preview of loaded data (first 10 rows):")
            st.dataframe(bulk_df.head(10))
        else:
            st.write("Preview of loaded data (first 10 rows):")
            st.dataframe(bulk_df.head(10))

            if st.button("Run bulk prediction on loaded data"):
                if model is None:
                    st.error("No model available for prediction.")
                else:
                    # Preprocess
                    X_bulk = bulk_df[model_feature_cols].copy()
                    X_bulk_scaled = apply_encoder_and_scaler(X_bulk, encoder, scaler, training_df=X_for_scaling, training_y=df.get('Churn Value', None))

                    # Predict probs and binary class
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_bulk_scaled.values)[:,1]
                    else:
                        probs = model.predict(X_bulk_scaled.values).astype(float)

                    out = bulk_df.copy()
                    out['Predicted_Churn_Prob'] = probs
                    out['Predicted_Churn'] = (out['Predicted_Churn_Prob'] >= 0.5).astype(int)

                    # Top_Reasons heuristic
                    if hasattr(model, "feature_importances_"):
                        fi = model.feature_importances_
                        reasons = []
                        for i in range(X_bulk_scaled.shape[0]):
                            expl = compute_explanation_for_row(X_bulk_scaled.values[i], feature_means, feature_stds, fi, X_bulk_scaled.columns.tolist(), top_k=3)
                            reasons.append(", ".join([f"{n}({s:.2f})" for n,s in expl]))
                        out['Top_Reasons'] = reasons
                    else:
                        if 'Churn Reason' in out.columns:
                            out['Top_Reasons'] = out['Churn Reason'].fillna("N/A")
                        else:
                            out['Top_Reasons'] = "N/A"

                    st.success("Bulk predictions complete — preview & analytics below.")
                    st.dataframe(out.head(100))
                    csv = out.to_csv(index=False).encode('utf-8')
                    st.download_button("Download results (CSV)", csv, file_name="bulk_predictions.csv", mime="text/csv")

                    # Analytics for the predicted batch
                    st.markdown("## 🔎 Analytics for predicted batch")
                    total_rows = len(out)
                    pred_churn_count = int(out['Predicted_Churn'].sum())
                    pred_churn_rate = pred_churn_count / total_rows * 100 if total_rows>0 else 0.0
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Rows", f"{total_rows:,}")
                    m2.metric("Predicted churners", f"{pred_churn_count:,}")
                    m3.metric("Predicted churn rate", f"{pred_churn_rate:.2f}%")

                    a1, a2 = st.columns(2)
                    a3, a4 = st.columns(2)

                    # Pie (robust label/count creation)
                    with a1:
                        label_series = out['Predicted_Churn'].map({0:'No Churn',1:'Predicted Churn'})
                        counts = label_series.value_counts().reset_index()
                        counts.columns = ['label','count']
                        fig_p = px.pie(counts, names='label', values='count', hole=0.35, title="Predicted churn share", template='plotly_white')
                        fig_p.update_traces(textinfo='percent+label')
                        st.plotly_chart(fig_p, use_container_width=True)
                        st.expander("Why this chart?").write("Shows predicted churn share in the batch; quick snapshot of risk.")

                    # Top reasons
                    with a2:
                        st.subheader("Top predicted reasons (aggregated)")
                        reason_series = out['Top_Reasons'].fillna("N/A").apply(lambda x: x.split(",")[0].strip() if isinstance(x, str) else "N/A")
                        reason_counts = reason_series.value_counts().reset_index()
                        reason_counts.columns = ['Reason','Count']
                        fig_b = px.bar(reason_counts.head(15), x='Count', y='Reason', orientation='h', title="Top predicted reasons", template='plotly_white')
                        fig_b.update_layout(yaxis=dict(autorange='reversed'))
                        st.plotly_chart(fig_b, use_container_width=True)
                        st.expander("Why this chart?").write("Aggregate heuristic reasons to prioritize fixes.")

                    # Tenure distribution
                    with a3:
                        st.subheader("Tenure distribution (predicted churn vs not)")
                        if 'Tenure Months' in out.columns:
                            fig_h = px.histogram(out, x='Tenure Months', color='Predicted_Churn', barmode='overlay', nbins=30, title="Tenure distribution", template='plotly_white')
                            st.plotly_chart(fig_h, use_container_width=True)
                            st.expander("Why this chart?").write("See whether churners concentrate at specific tenure lengths.")
                        else:
                            st.info("No Tenure Months column in uploaded data.")

                    # Heatmap
                    with a4:
                        st.subheader("Heatmap: Tenure bucket × Monthly Charges quartile")
                        if ('Tenure Months' in out.columns) and ('Monthly Charges' in out.columns):
                            out_loc = out.copy()
                            out_loc['Tenure Bucket'] = tenure_buckets(out_loc['Tenure Months'])
                            try:
                                out_loc['Monthly Q'] = pd.qcut(out_loc['Monthly Charges'].rank(method='first'), 4, labels=['Q1','Q2','Q3','Q4'])
                                pivot = out_loc.groupby(['Tenure Bucket','Monthly Q'])['Predicted_Churn'].mean().reset_index()
                                pivot_table = pivot.pivot(index='Tenure Bucket', columns='Monthly Q', values='Predicted_Churn').fillna(0)
                                fig_hm = px.imshow(pivot_table.values, x=pivot_table.columns.tolist(), y=pivot_table.index.tolist(),
                                                   color_continuous_scale='RdYlGn_r', title="Predicted churn heatmap", aspect='auto', template='plotly_white')
                                st.plotly_chart(fig_hm, use_container_width=True)
                                st.expander("Why this chart?").write("Find segments with high predicted churn in this batch.")
                            except Exception as e:
                                st.info("Unable to compute heatmap: " + str(e))
                        else:
                            st.info("Need Tenure Months and Monthly Charges for heatmap.")

                    with st.expander("Show full results table"):
                        st.dataframe(out)
    else:
        st.info("No batch loaded yet. Upload a file or use the sample button.")

# Footer: artifact paths
st.markdown("---")
st.write("Artifacts (paths):")
for name, p in [("Dataset", DATA_PATH), ("Sample", SAMPLE_PATH), ("Encoder", ENCODER_PATH), ("Scaler", SCALER_PATH), ("XGB model", MODEL_XGB_PATH), ("RF model", MODEL_RF_PATH)]:
    st.write(f"- {name}: `{p}` (exists: {os.path.exists(p)})")
