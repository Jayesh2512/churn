"""
Customer Churn Prediction - Streamlit app
Place this file next to WA_Fn-UseC_-Telco-Customer-Churn.csv and run:
    pip install streamlit pandas scikit-learn matplotlib seaborn
    streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_FILE = Path(__file__).parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns="customerID")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def build_pipeline(model, cat_cols):
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


@st.cache_resource
def train_all(df: pd.DataFrame):
    X, y = df.drop(columns="Churn"), (df["Churn"] == "Yes").astype(int)
    cat_cols = [c for c in X.columns if c not in NUMERIC]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    }
    fitted, rows = {}, []
    for name, m in models.items():
        pipe = build_pipeline(m, cat_cols).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        fitted[name] = (pipe, pred)
        rows.append({"Model": name, "Accuracy": accuracy_score(y_te, pred)})
    results = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    return fitted, results, y_te, cat_cols


if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE.name}. Put it in the same folder as this script.")
    st.stop()

df = load_data()
fitted, results, y_test, cat_cols = train_all(df)

st.title("📉 Customer Churn Prediction")
st.caption("Telco Customer Churn dataset · predict which customers are likely to leave")

tab_eda, tab_models, tab_predict = st.tabs(["📊 EDA", "🤖 Model Performance", "🔮 Predict Churn"])

# ---------------------------------------------------------------- EDA
with tab_eda:
    churn_rate = (df["Churn"] == "Yes").mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{churn_rate:.1%}")
    c3.metric("Avg tenure (mo)", f"{df['tenure'].mean():.1f}")
    c4.metric("Avg monthly charge", f"${df['MonthlyCharges'].mean():.2f}")

    with st.expander("Preview data"):
        st.dataframe(df.head(50), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df["Churn"].value_counts()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
               colors=["#66b3ff", "#ff9999"], explode=(0, 0.1))
        ax.set_title("Customer Churn Distribution")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(df, x="tenure", hue="Churn", bins=30, kde=True, ax=ax)
        ax.set_title("Tenure Distribution by Churn")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.kdeplot(df, x="MonthlyCharges", hue="Churn", fill=True, ax=ax)
        ax.set_title("Monthly Charges by Churn")
        st.pyplot(fig)
    with col4:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(df, x="Contract", hue="Churn", ax=ax)
        ax.set_title("Churn by Contract Type")
        plt.setp(ax.get_xticklabels(), rotation=15)
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(df, x="PaymentMethod", hue="Churn", ax=ax)
    ax.set_title("Churn by Payment Method")
    st.pyplot(fig)

# ---------------------------------------------------------------- Models
with tab_models:
    st.subheader("Model accuracy comparison")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(results.style.format({"Accuracy": "{:.2%}"}), use_container_width=True,
                     hide_index=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(data=results, x="Accuracy", y="Model", palette="mako", ax=ax)
        ax.set_xlim(0.6, 1.0)
        st.pyplot(fig)

    choice = st.selectbox("Inspect model", results["Model"], index=0)
    pipe, pred = fitted[choice]
    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with col2:
        report = classification_report(y_test, pred, target_names=["No", "Yes"], output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    if choice in ("Random Forest", "Decision Tree"):
        names = pipe.named_steps["pre"].get_feature_names_out()
        imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
        imp.index = [n.split("__", 1)[1] for n in imp.index]
        fig, ax = plt.subplots(figsize=(7, 4))
        imp.nlargest(10).sort_values().plot(kind="barh", color="teal", ax=ax)
        ax.set_title("Top 10 Important Features")
        st.pyplot(fig)

# ---------------------------------------------------------------- Predict
with tab_predict:
    st.subheader("Enter customer details")
    model_name = st.selectbox("Model", results["Model"], index=0, key="pred_model")
    pipe = fitted[model_name][0]

    def opts(col):
        return sorted(df[col].unique().tolist())

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", opts("gender"))
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", opts("Partner"))
            dependents = st.selectbox("Dependents", opts("Dependents"))
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone service", opts("PhoneService"))
            multi = st.selectbox("Multiple lines", opts("MultipleLines"))
        with c2:
            internet = st.selectbox("Internet service", opts("InternetService"))
            security = st.selectbox("Online security", opts("OnlineSecurity"))
            backup = st.selectbox("Online backup", opts("OnlineBackup"))
            protection = st.selectbox("Device protection", opts("DeviceProtection"))
            support = st.selectbox("Tech support", opts("TechSupport"))
            tv = st.selectbox("Streaming TV", opts("StreamingTV"))
            movies = st.selectbox("Streaming movies", opts("StreamingMovies"))
        with c3:
            contract = st.selectbox("Contract", opts("Contract"))
            paperless = st.selectbox("Paperless billing", opts("PaperlessBilling"))
            payment = st.selectbox("Payment method", opts("PaymentMethod"))
            monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=0.5)
            total = st.number_input("Total charges ($)", 0.0, 10000.0,
                                    float(round(monthly * max(tenure, 1), 2)), step=10.0)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multi, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support,
            "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }])[df.drop(columns="Churn").columns]
        prob = pipe.predict_proba(row)[0, 1]
        st.progress(float(prob), text=f"Churn probability: {prob:.1%}")
        if prob >= 0.5:
            st.error(f"⚠️ High churn risk ({prob:.1%}). Consider a retention offer, "
                     "e.g. a longer-term contract discount.")
        else:
            st.success(f"✅ Low churn risk ({prob:.1%}).")
