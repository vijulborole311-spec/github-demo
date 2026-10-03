from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ------------------------------------------------------------
# Page setup and visual styling
# ------------------------------------------------------------
st.set_page_config(page_title="CerviCare | ML Demo", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#f5f8fc; color:#182b45;}
.block-container {padding-top:1.7rem; padding-bottom:2.5rem; max-width:1420px;}
[data-testid="stSidebar"] {background:#102b4e;}
[data-testid="stSidebar"] * {color:#eef5ff !important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {padding:7px 10px; border-radius:9px;}
.hero {background:linear-gradient(115deg,#12345b 0%,#176b78 100%); padding:27px 32px; border-radius:20px; color:white; margin-bottom:20px; box-shadow:0 10px 28px rgba(22,57,89,.12);}
.hero h1 {font-family:'Manrope',sans-serif; font-size:31px; color:white; margin:0 0 5px 0; font-weight:800;}
.hero p {color:#d8eaf4; margin:0; font-size:15px;}
.eyebrow {font-size:11px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:#a8d8db; margin-bottom:8px;}
.section-title {font-family:'Manrope',sans-serif; font-size:20px; font-weight:800; color:#173657; margin:12px 0 4px 0;}
.subtle {color:#64758b; font-size:13px;}
.metric-card {background:white; border:1px solid #e7edf4; border-radius:15px; padding:16px 18px; box-shadow:0 4px 14px rgba(22,50,80,.035); min-height:108px;}
.metric-label {color:#718198; font-size:12px; font-weight:600; margin-bottom:9px;}
.metric-value {font-family:'Manrope',sans-serif; color:#173657; font-size:27px; font-weight:800; line-height:1.1;}
.metric-foot {font-size:11px; color:#8492a5; margin-top:7px;}
.panel {background:white; border:1px solid #e7edf4; border-radius:17px; padding:20px 22px;}
.warning {background:#fff8e8; border:1px solid #f4dfaa; border-left:4px solid #dba437; padding:13px 16px; border-radius:10px; color:#674d18; font-size:13px;}
.stButton>button[kind="primary"] {background:#176b78; border:0; border-radius:10px; font-weight:700; padding:10px 18px;}
.stButton>button[kind="primary"]:hover {background:#125762;}
div[data-testid="stForm"] {background:white; border:1px solid #e7edf4; border-radius:16px; padding:18px;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cervical-cancer_csv (1).csv"
PREDICTION_EXCLUDED_COLUMNS = [
    "Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx",
    "Hinselmann", "Schiller", "Citology",
]

# Keep diagnosis and screening results out of predictors for the Biopsy target.
@st.cache_data(show_spinner=False)
def load_data(path_string, modified_time):
    df = pd.read_csv(path_string)
    df = df.fillna(0)
    df = df.drop(columns=["STDs: Time since first diagnosis", "STDs: Time since last diagnosis", "Unnamed: 0"], errors="ignore")
    return df

@st.cache_resource(show_spinner="Training the notebook's Logistic Regression model...")
def train_model(data):
    X = data.drop(columns=["Biopsy", *PREDICTION_EXCLUDED_COLUMNS], errors="ignore")
    y = data["Biopsy"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    y_baseline = baseline.predict(X_test)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X.columns.tolist(), y_test, y_pred, y_baseline

# Human-readable groupings. All feature names remain exactly those used by the notebook.
GROUPS = {
    "General information": ["Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies"],
    "Lifestyle": ["Smokes", "Smokes (years)", "Smokes (packs/year)"],
    "Contraceptive history": ["Hormonal Contraceptives", "Hormonal Contraceptives (years)", "IUD", "IUD (years)"],
    "STI-related variables": ["STDs", "STDs (number)", "STDs:condylomatosis", "STDs:cervical condylomatosis", "STDs:vaginal condylomatosis", "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", "STDs:pelvic inflammatory disease", "STDs:genital herpes", "STDs:molluscum contagiosum", "STDs:AIDS", "STDs:HIV", "STDs:Hepatitis B", "STDs:HPV", "STDs: Number of diagnosis"],
}
BINARY = {"Smokes", "Hormonal Contraceptives", "IUD", "STDs", "STDs:condylomatosis", "STDs:cervical condylomatosis", "STDs:vaginal condylomatosis", "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", "STDs:pelvic inflammatory disease", "STDs:genital herpes", "STDs:molluscum contagiosum", "STDs:AIDS", "STDs:HIV", "STDs:Hepatitis B", "STDs:HPV"}
DISPLAY = {
    "Age":"Age", "Number of sexual partners":"Number of sexual partners", "First sexual intercourse":"Age at first sexual intercourse", "Num of pregnancies":"Number of pregnancies",
    "Smokes":"Smoking history", "Smokes (years)":"Years of smoking", "Smokes (packs/year)":"Smoking exposure (packs/year)",
    "Hormonal Contraceptives":"Hormonal contraceptive history", "Hormonal Contraceptives (years)":"Years using hormonal contraceptives", "IUD":"IUD history", "IUD (years)":"Years with IUD",
    "STDs":"STI history recorded", "STDs (number)":"Number of STIs recorded", "STDs: Number of diagnosis":"Number of STI diagnoses",
}

def pretty(name):
    return DISPLAY.get(name, name.replace("STDs:", "STI: "))

def metric(label, value, foot=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>', unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("<div style='font-family:Manrope;font-size:23px;font-weight:800;margin:6px 0 0'>🩺 CerviCare</div><div style='font-size:11px;color:#b7cadc;margin:3px 0 22px'>CERVICAL HEALTH · ML PROJECT</div>", unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["Overview", "Prediction demo", "Model performance", "Dataset explorer"], label_visibility="visible")
    st.markdown("<hr style='border-color:#34516f;margin:20px 0'>", unsafe_allow_html=True)
    st.caption("Academic machine-learning prototype")
    st.caption("Model: Logistic Regression")

st.markdown("<div class='hero'><div class='eyebrow'>Machine learning project</div><h1>Cervical Health Analysis</h1><p>Explore the dataset, test the notebook-trained workflow, and review model performance.</p></div>", unsafe_allow_html=True)
st.markdown("<div class='warning'><b>Educational use only.</b> This prototype is not a medical device, screening tool, diagnosis, or clinical risk assessment. Its outputs must not guide healthcare decisions. Speak with a qualified healthcare professional about screening or medical concerns.</div>", unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error(f"Dataset not found at: {DATA_PATH}\n\nKeep your existing folder structure and ensure the CSV is inside the data folder with the name `cervical-cancer_csv (1).csv`.")
    st.stop()

try:
    data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    model, features, y_test, y_pred, y_baseline = train_model(data)
except Exception as exc:
    st.error(f"Could not load the dataset or train the model: {exc}")
    st.stop()

if page == "Overview":
    st.markdown("<div class='section-title'>Project at a glance</div><div class='subtle'>A quick summary of the data and the model setup.</div>", unsafe_allow_html=True)
    total, feature_count = len(data), len(features)
    positive = int(data["Biopsy"].sum())
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric("Dataset records", f"{total:,}", "Rows available after notebook preprocessing")
    with c2: metric("Input features", str(feature_count), "Predictors used by the notebook")
    with c3: metric("Target-positive rows", f"{positive:,}", "Biopsy target = 1 in this dataset")
    with c4: metric("Test split", "20%", "Stratified split · random state 42")
    st.write("")
    left,right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Workflow")
        st.markdown("""
        1. Load the CSV from the existing `data` folder.
        2. Fill missing values with `0`, as in the notebook.
        3. Drop the STI diagnosis-time columns, any `Unnamed: 0` index column, and prior diagnosis/screening results.
        4. Split risk-factor predictors and the `Biopsy` target using a stratified 80/20 split.
        5. Train a pipeline with median imputation, standard scaling, and balanced Logistic Regression.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Target distribution")
        counts = data["Biopsy"].value_counts().reindex([0,1], fill_value=0)
        st.bar_chart(pd.DataFrame({"Records":counts.values}, index=["Biopsy = 0", "Biopsy = 1"]), color="#27818a")
        st.caption("Counts reflect the dataset labels, not population prevalence.")
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Prediction demo":
    st.markdown("<div class='section-title'>Try the model</div><div class='subtle'>Enter values using the dataset's numeric coding. No information is saved by this app.</div>", unsafe_allow_html=True)
    st.info("This form is a software demonstration. It is not intended for entering real patient details or interpreting a person's health.")
    with st.form("demo_form"):
        entered = {}
        for group, names in GROUPS.items():
            names = [n for n in names if n in features]
            if not names: continue
            with st.expander(group, expanded=(group == "General information")):
                cols = st.columns(2)
                for i, name in enumerate(names):
                    s = pd.to_numeric(data[name], errors="coerce").dropna()
                    lo = float(s.min()) if len(s) else 0.0
                    hi = float(s.max()) if len(s) else 1.0
                    default = float(s.median()) if len(s) else 0.0
                    with cols[i % 2]:
                        if name in BINARY:
                            val = st.selectbox(pretty(name), [0,1], index=int(round(default)) if default in (0,1) else 0, format_func=lambda x: "No / 0" if x == 0 else "Yes / 1", help="Numeric encoding used in the dataset.")
                        else:
                            step = 1.0 if float(lo).is_integer() and float(hi).is_integer() else 0.5
                            val = st.number_input(pretty(name), min_value=float(lo), max_value=float(hi), value=float(np.clip(default,lo,hi)), step=step, help=f"Observed dataset range: {lo:g} to {hi:g}. This is a dataset range, not a recommended clinical range.")
                        entered[name] = float(val)
        submitted = st.form_submit_button("Run demo prediction", type="primary", use_container_width=True)
    if submitted:
        row = pd.DataFrame([{f: entered.get(f, float(pd.to_numeric(data[f], errors="coerce").median() or 0)) for f in features}], columns=features)
        pred = int(model.predict(row)[0])
        probs = model.predict_proba(row)[0]
        st.markdown("### Model output")
        a,b = st.columns(2)
        with a: metric("Predicted dataset class", str(pred), "Class label only · not a diagnosis")
        with b: metric("Model score for class 1", f"{probs[1]:.1%}", "Model output; not an individual's clinical probability")
        st.progress(float(probs[1]))
        st.caption("The model score reflects this fitted model and its dataset coding. It is not calibrated or clinically validated.")

elif page == "Model performance":
    st.markdown("<div class='section-title'>Model performance</div><div class='subtle'>Metrics on the held-out test portion generated with the notebook's split settings.</div>", unsafe_allow_html=True)
    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test,y_pred,zero_division=0)
    rec = recall_score(y_test,y_pred,zero_division=0)
    f1 = f1_score(y_test,y_pred,zero_division=0)
    baseline_acc = accuracy_score(y_test,y_baseline)
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: metric("Accuracy",f"{acc:.1%}","Overall correct classifications")
    with c2: metric("Precision · class 1",f"{prec:.1%}","Positive predictions that were correct")
    with c3: metric("Recall · class 1",f"{rec:.1%}","Actual class 1 cases detected")
    with c4: metric("F1 · class 1",f"{f1:.1%}","Harmonic mean of precision and recall")
    with c5: metric("Majority baseline",f"{baseline_acc:.1%}","Accuracy from always predicting the training majority class")
    left,right = st.columns([.8,1.2], gap="large")
    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test,y_pred,labels=[0,1])
        st.dataframe(pd.DataFrame(cm,index=["Actual 0","Actual 1"],columns=["Predicted 0","Predicted 1"]),use_container_width=True)
        st.caption("Counts for the held-out test split.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Classification report")
        report = classification_report(y_test,y_pred,output_dict=True,zero_division=0)
        rep = pd.DataFrame(report).T
        st.dataframe(rep.round(3),use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Metrics are dataset-specific and do not establish clinical safety, validity, or real-world performance.")

elif page == "Dataset explorer":
    st.markdown("<div class='section-title'>Explore the dataset</div><div class='subtle'>A lightweight view of the CSV after the notebook's preprocessing steps.</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Preview")
        st.dataframe(data.head(12),use_container_width=True, height=360)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Feature summary")
        summary = data.describe().T[["mean","std","min","max"]].round(2)
        st.dataframe(summary,use_container_width=True,height=360)
        st.markdown("</div>", unsafe_allow_html=True)
    st.download_button("Download processed dataset", data.to_csv(index=False).encode("utf-8"), file_name="cervical_dataset_processed.csv", mime="text/csv")

st.markdown("<div style='text-align:center;color:#8a98aa;font-size:11px;margin-top:35px'>CerviCare · Academic ML interface · Not for clinical use</div>", unsafe_allow_html=True)
