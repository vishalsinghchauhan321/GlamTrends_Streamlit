import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import warnings
import re
from collections import Counter
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error, r2_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GlamTrends Analysis",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  :root {
    --primary: #2563eb;
    --secondary: #1e40af;
    --light-bg: #ffffff;
    --text-dark: #1a202c;
    --text-light: #4a5568;
  }
  
  * { margin: 0; padding: 0; box-sizing: border-box; }
  
  /* ─── Sidebar ─── */
  [data-testid="stSidebar"] { 
    background: #f8fafc !important;
    padding-top: 20px !important;
    border-right: 2px solid #e2e8f0 !important;
  }
  [data-testid="stSidebar"] [class*="stMarkdown"] {
    color: #1a202c !important;
  }
  [data-testid="stSidebar"] h1, 
  [data-testid="stSidebar"] h2, 
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] button,
  [data-testid="stSidebar"] div {
    color: #1a202c !important;
  }
  [data-testid="stSidebar"] .stFileUploader {
    color: #1a202c !important;
  }
  
  /* ─── Sidebar Footer ─── */
  .sidebar-footer {
    color: #1a202c !important;
    background: #e8f1ff !important;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    font-size: 0.8rem;
    line-height: 1.6;
    word-wrap: break-word;
    margin-top: 30px;
    border: 1px solid #bfdbfe;
  }
  .sidebar-footer strong {
    color: #1e40af !important;
    font-weight: 600;
  }
  .sidebar-footer small {
    color: #1a202c !important;
  }
  
  /* ─── Main Content ─── */
  body, .appview-container { 
    background-color: #f8f9fa !important;
  }
  [data-testid="stAppViewContainer"] {
    background-color: #f8f9fa !important;
  }
  
  /* ─── Metric Cards ─── */
  .metric-card {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(30, 64, 175, 0.08) 100%);
    border: 2px solid rgba(37, 99, 235, 0.2);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.1);
    transition: all 0.3s ease;
  }
  .metric-card:hover {
    border-color: #2563eb;
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.2);
    transform: translateY(-4px);
  }
  .metric-card h2 { 
    color: #2563eb !important; 
    margin: 0 0 8px 0; 
    font-size: 2.5rem; 
    font-weight: 700;
    letter-spacing: -1px;
  }
  .metric-card p { 
    color: #4a5568 !important; 
    margin: 0; 
    font-size: 0.95rem;
    font-weight: 500;
  }
  
  /* ─── Headings ─── */
  h1 { 
    color: #2563eb !important; 
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 24px;
    letter-spacing: -1px;
  }
  h2 { 
    color: #2563eb !important; 
    font-size: 1.75rem;
    font-weight: 600;
    margin-top: 24px;
    margin-bottom: 16px;
  }
  h3 { 
    color: #1e40af !important; 
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 12px;
  }
  
  /* ─── Subheaders ─── */
  [data-testid="stMarkdownContainer"] h2 {
    color: #2563eb !important;
  }
  [data-testid="stMarkdownContainer"] h3 {
    color: #1e40af !important;
  }
  
  /* ─── Tabs ─── */
  .stTabs [data-baseweb="tab"] { 
    color: #2563eb !important; 
    font-weight: 600;
    font-size: 1rem;
  }
  .stTabs [aria-selected="true"] { 
    color: #2563eb !important;
    border-bottom: 3px solid #2563eb !important;
  }
  .stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid rgba(37, 99, 235, 0.2) !important;
  }
  
  /* ─── Info/Alert Boxes ─── */
  .stAlert {
    border-radius: 12px !important;
    border-left: 4px solid #2563eb !important;
  }
  
  /* ─── Buttons ─── */
  .stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
  }
  
  /* ─── Input Fields ─── */
  input, select, textarea {
    color: #1a202c !important;
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
  }
  input::placeholder, textarea::placeholder {
    color: #94a3b8 !important;
  }
  
  /* ─── Sidebar Input Fields ─── */
  [data-testid="stSidebar"] input,
  [data-testid="stSidebar"] select,
  [data-testid="stSidebar"] textarea {
    color: #1a202c !important;
    background-color: #ffffff !important;
    border: 2px solid #cbd5e1 !important;
  }
  [data-testid="stSidebar"] input::placeholder,
  [data-testid="stSidebar"] textarea::placeholder {
    color: #94a3b8 !important;
  }
  [data-testid="stSidebar"] input:focus,
  [data-testid="stSidebar"] select:focus,
  [data-testid="stSidebar"] textarea:focus {
    border: 2px solid #2563eb !important;
    background-color: #ffffff !important;
    color: #1a202c !important;
  }
  
  /* ─── Data Frame ─── */
  [data-testid="dataFrameContainer"] {
    border-radius: 12px !important;
  }
  th {
    background-color: rgba(37, 99, 235, 0.1) !important;
    color: #2563eb !important;
    font-weight: 600;
  }
  
  /* ─── Divider ─── */
  hr { 
    margin: 24px 0 !important;
    border: none;
    border-top: 2px solid rgba(102, 126, 234, 0.2);
  }
  
  /* ─── Text Styling ─── */
  body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
  p, span, label, div {
    color: #4a5568 !important;
  }
  
  /* ─── Selectbox & Slider Labels ─── */
  [data-testid="stSelectbox"] label,
  [data-testid="stSlider"] label {
    color: #667eea !important;
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)

PURPLE = "#667eea"
ACCENT = "#764ba2"
PALETTE = "RdPu"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👗 GlamTrends")
    st.markdown("*Women's E-Commerce Analytics*")
    st.markdown("---")
    st.markdown("### 📊 Data Upload")
    uploaded = st.file_uploader("Select CSV Dataset", type=["csv"])
    st.markdown("---")
    st.markdown("""
    <hr style='margin: 30px 0; border: none; border-top: 1px solid rgba(255,255,255,0.3);'>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #ffffff; font-size: 0.8rem; line-height: 1.6;'>
        <strong>Brainybeam Info-Tech</strong><br>
        Internship Project 2024<br>
        <small style='opacity: 0.9;'>Advanced Analytics Dashboard</small>
    </div>
    """, unsafe_allow_html=True)

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, index_col=0)
    # Clean
    df["Review Text"] = df["Review Text"].fillna("")
    df["Title"]       = df["Title"].fillna("")
    for col in ["Division Name", "Department Name", "Class Name"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    df = df.drop_duplicates()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    # Feature engineering
    df["Sentiment"]     = df["Rating"].apply(lambda r: "Positive" if r>=4 else ("Neutral" if r==3 else "Negative"))
    df["Review_Length"] = df["Review_Text"].apply(lambda x: len(str(x).split()))
    df["High_Engagement"] = (df["Positive_Feedback_Count"] > df["Positive_Feedback_Count"].median()).astype(int)
    return df

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("# 👗 GlamTrends — Women's Clothing Analytics")
st.markdown("<p style='font-size: 1.1rem; color: #666; margin-bottom: 24px;'>Comprehensive E-Commerce Review Analysis & Predictive Insights</p>", unsafe_allow_html=True)

if uploaded is None:
    st.info("� **Upload Dataset** — Select the Women's E-Commerce Reviews CSV file from the sidebar to begin analysis.")
    st.markdown("""
    ### 📥 Get Started
    1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
    2. Upload it using the file uploader in the left sidebar
    3. Explore insights across multiple analytics tabs
    """)
    st.stop()

df = load_data(uploaded)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 Overview", "📈 EDA", "🤖 ML Models", "💬 NLP", "🏆 Summary"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("📊 Dataset Overview")
    st.markdown("Key metrics and statistics from the uploaded dataset")
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Total Reviews", f"{len(df):,}"),
        ("Avg Rating",    f"{df['Rating'].mean():.2f} / 5"),
        ("Recommend Rate",f"{df['Recommended_IND'].mean()*100:.1f}%"),
        ("Departments",   str(df['Department_Name'].nunique())),
    ]
    for col, (label, val) in zip([c1,c2,c3,c4], metrics):
        col.markdown(f'<div class="metric-card"><h2>{val}</h2><p>{label}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📋 Raw Data Preview")
    n = st.slider("Rows to preview", 5, 50, 10)
    st.dataframe(df.head(n), use_container_width=True)

    st.subheader("📝 Column Information")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type":  [str(df[c].dtype) for c in df.columns],
        "Non-Null Count": [df[c].notna().sum() for c in df.columns],
        "Null Count": [df[c].isna().sum() for c in df.columns],
    })
    st.dataframe(info_df, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("📈 Exploratory Data Analysis")
    st.markdown("Interactive visualizations to uncover patterns and trends")
    
    eda_section = st.selectbox("Choose Analysis", [
        "Rating Distribution",
        "Age Distribution",
        "Department & Category Analysis",
        "Recommendation Analysis",
        "Correlation Heatmap",
        "Sentiment Distribution",
    ], label_visibility="collapsed")

    fig, axes = None, None

    if eda_section == "Rating Distribution":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("GlamTrends — Rating Analysis", fontsize=16, fontweight="bold", color=PURPLE)
        sns.countplot(x="Rating", data=df, palette=PALETTE, ax=axes[0])
        axes[0].set_title("Rating Distribution")
        axes[0].set_xlabel("Rating (1–5)"); axes[0].set_ylabel("Count")
        for p in axes[0].patches:
            axes[0].annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2, p.get_height()),
                             ha="center", va="bottom", fontsize=9)
        avg_rating_dept = df.groupby("Department_Name")["Rating"].mean().sort_values()
        sns.barplot(y=avg_rating_dept.index, x=avg_rating_dept.values, palette=PALETTE, ax=axes[1])
        axes[1].set_title("Avg Rating by Department"); axes[1].set_xlabel("Average Rating")
        plt.tight_layout()

    elif eda_section == "Age Distribution":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("GlamTrends — Customer Age Analysis", fontsize=16, fontweight="bold", color=PURPLE)
        axes[0].hist(df["Age"], bins=30, color="#C084FC", edgecolor="white")
        axes[0].axvline(df["Age"].mean(), color="red", linestyle="--", label=f"Mean: {df['Age'].mean():.1f}")
        axes[0].legend(); axes[0].set_title("Age Distribution"); axes[0].set_xlabel("Age")
        bins = [0,25,35,45,55,65,100]
        labels_b = ["<25","25-35","35-45","45-55","55-65","65+"]
        df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels_b)
        age_grp = df.groupby("Age_Group")["Rating"].mean()
        sns.barplot(x=age_grp.index.astype(str), y=age_grp.values, palette=PALETTE, ax=axes[1])
        axes[1].set_title("Avg Rating by Age Group"); axes[1].set_ylabel("Avg Rating")
        plt.tight_layout()

    elif eda_section == "Department & Category Analysis":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("GlamTrends — Department & Category", fontsize=16, fontweight="bold", color=PURPLE)
        dept = df["Department_Name"].value_counts()
        sns.barplot(y=dept.index, x=dept.values, palette=PALETTE, ax=axes[0])
        axes[0].set_title("Reviews by Department"); axes[0].set_xlabel("Count")
        cls = df["Class_Name"].value_counts().head(10)
        sns.barplot(y=cls.index, x=cls.values, palette="PuRd", ax=axes[1])
        axes[1].set_title("Top 10 Categories"); axes[1].set_xlabel("Count")
        plt.tight_layout()

    elif eda_section == "Recommendation Analysis":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("GlamTrends — Recommendation Analysis", fontsize=16, fontweight="bold", color=PURPLE)
        rc = df["Recommended_IND"].value_counts()
        axes[0].pie(rc, labels=["Recommended","Not Recommended"],
                    autopct="%1.1f%%", colors=["#8B2FC9","#E9D5FF"], startangle=90)
        axes[0].set_title("Overall Recommendation Rate")
        avg_rec = df.groupby("Department_Name")["Recommended_IND"].mean()*100
        sns.barplot(y=avg_rec.index, x=avg_rec.values, palette=PALETTE, ax=axes[1])
        axes[1].set_title("Recommend % by Department"); axes[1].set_xlabel("%")
        plt.tight_layout()

    elif eda_section == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 6))
        num = df.select_dtypes(include=[np.number])
        sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap=PALETTE,
                    linewidths=0.5, linecolor="white", ax=ax)
        ax.set_title("GlamTrends — Correlation Heatmap", fontsize=14, fontweight="bold", color=PURPLE)
        plt.tight_layout()

    elif eda_section == "Sentiment Distribution":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("GlamTrends — Sentiment Analysis", fontsize=16, fontweight="bold", color=PURPLE)
        sc = df["Sentiment"].value_counts()
        axes[0].pie(sc, labels=sc.index, autopct="%1.1f%%",
                    colors=["#8B2FC9","#C084FC","#F3E8FF"], startangle=90)
        axes[0].set_title("Sentiment Distribution")
        sns.countplot(x="Sentiment", data=df, order=["Positive","Neutral","Negative"],
                      palette=PALETTE, ax=axes[1])
        axes[1].set_title("Sentiment Count"); axes[1].set_xlabel("Sentiment")
        plt.tight_layout()

    if fig:
        st.pyplot(fig)
        plt.close(fig)

# ════════════════════════════════════════════════════════════════
# TAB 3 — ML MODELS
# ════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🤖 Machine Learning Models")
    st.markdown("Classification and regression models to predict customer recommendations")
    st.info("Training models on your dataset. This may take a moment... ⏳")

    @st.cache_resource
    def train_models(df_hash):
        le = LabelEncoder()
        _df = df.copy()
        _df["Dept_Encoded"]  = le.fit_transform(_df["Department_Name"])
        _df["Class_Encoded"] = le.fit_transform(_df["Class_Name"])
        _df["Div_Encoded"]   = le.fit_transform(_df["Division_Name"])
        features = ["Age","Rating","Positive_Feedback_Count","Review_Length",
                    "High_Engagement","Dept_Encoded","Class_Encoded","Div_Encoded"]
        X = _df[features].fillna(0)
        y = _df["Recommended_IND"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM":                 SVC(kernel="rbf", random_state=42),
        }
        results = {}
        for name, m in models.items():
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            results[name] = {"model": m, "acc": accuracy_score(y_test, y_pred),
                             "report": classification_report(y_test, y_pred, output_dict=True),
                             "cm": confusion_matrix(y_test, y_pred), "y_pred": y_pred}
        # Linear Regression
        X_reg = _df[["Age","Rating","Review_Length","Recommended_IND"]].fillna(0)
        y_reg = _df["Positive_Feedback_Count"].fillna(0)
        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        lr_reg = LinearRegression()
        lr_reg.fit(Xr_tr, yr_tr)
        yr_pr = lr_reg.predict(Xr_te)
        rf_feat = models["Random Forest"].feature_importances_
        return results, features, rf_feat, Xr_te, yr_te, yr_pr

    results, features, rf_feat, Xr_te, yr_te, yr_pr = train_models(len(df))

    # Model accuracy comparison
    st.markdown("### 📊 Model Performance Comparison")
    acc_data = {k: v["acc"]*100 for k, v in results.items()}
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(acc_data.keys(), acc_data.values(), color=["#667eea","#764ba2","#A78BFA"],
                  edgecolor="white", linewidth=2, width=0.6)
    for bar, val in zip(bars, acc_data.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{val:.1f}%", ha="center", fontsize=13, fontweight="bold", color="#667eea")
    ax.set_title("Classification Accuracy Across Models", fontsize=14, fontweight="bold", color=PURPLE)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    # Best model confusion matrix
    best_name = max(results, key=lambda k: results[k]["acc"])
    st.markdown(f"### 🎯 Confusion Matrix — {best_name} (Best Model ✨)")
    st.markdown(f"Accuracy: **{results[best_name]['acc']*100:.2f}%**")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(results[best_name]["cm"], annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=["Not Recommended","Recommended"],
                yticklabels=["Not Recommended","Recommended"],
                linewidths=2, linecolor="white", ax=ax, cbar_kws={"label": "Count"})
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold", color=PURPLE)
    ax.set_ylabel("Actual Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    col1, col2 = st.columns(2)
    # Feature importance
    with col1:
        st.markdown("### Feature Importance (Random Forest)")
        feat_df = pd.DataFrame({"Feature": features, "Importance": rf_feat}).sort_values("Importance")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(feat_df["Feature"], feat_df["Importance"], color="#C084FC", edgecolor="white")
        ax.set_title("Feature Importance", fontsize=13, fontweight="bold", color=PURPLE)
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    # Linear regression
    with col2:
        st.markdown("### Linear Regression — Feedback Prediction")
        rmse = np.sqrt(mean_squared_error(yr_te, yr_pr))
        r2   = r2_score(yr_te, yr_pr)
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("R² Score", f"{r2:.4f}")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(yr_te[:300], yr_pr[:300], alpha=0.5, color=PURPLE, edgecolors="white", s=40)
        ax.plot([yr_te.min(), yr_te.max()], [yr_te.min(), yr_te.max()], "r--", lw=2)
        ax.set_title("Actual vs Predicted Feedback", fontsize=12, fontweight="bold", color=PURPLE)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

# ════════════════════════════════════════════════════════════════
# TAB 4 — NLP
# ════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("💬 Natural Language Processing")
    st.markdown("Text analysis and sentiment-based word frequency insights")

    stop_words = {"the","a","an","is","it","in","on","at","to","for","of","and","or",
                  "but","i","my","this","that","was","with","have","had","not","be",
                  "are","so","very","just","its","as","do","did","they","we","you","me"}

    def top_words(series, n=15):
        words = []
        for text in series:
            text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
            words.extend([w for w in text.split() if w not in stop_words and len(w) > 2])
        return Counter(words).most_common(n)

    sentiment_filter = st.selectbox("Filter by Sentiment", ["All","Positive","Neutral","Negative"], key="sentiment_filter_words")
    n_words = st.slider("Number of Top Words", 10, 30, 15, key="top_words_slider")

    # Filter data based on sentiment selection
    if sentiment_filter == "All":
        subset = df
    else:
        subset = df[df["Sentiment"] == sentiment_filter]

    tw = top_words(subset["Review_Text"], n=n_words)
    words, counts = zip(*tw) if tw else ([], [])

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#667eea" if i % 2 == 0 else "#764ba2" for i in range(len(words))]
    ax.barh(list(words), list(counts), color=colors, edgecolor="white", linewidth=1.5)
    ax.set_title(f"Most Frequent Words in {sentiment_filter} Reviews",
                 fontsize=14, fontweight="bold", color=PURPLE)
    ax.set_xlabel("Frequency", fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    st.markdown("### 📏 Review Length Distribution")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.histplot(data=df, x="Review_Length", hue="Sentiment",
                 palette={"Positive":"#667eea","Neutral":"#764ba2","Negative":"#A78BFA"},
                 bins=40, ax=ax, kde=True)
    ax.set_title("Review Length Distribution by Sentiment", fontsize=13, fontweight="bold", color=PURPLE)
    ax.set_xlabel("Word Count", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

# ════════════════════════════════════════════════════════════════
# TAB 5 — SUMMARY
# ════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("🏆 Executive Summary & Insights")
    st.markdown("Key findings and actionable insights from the analysis")

    best_name_summary = max(results, key=lambda k: results[k]["acc"])

    summary_data = {
        "Total Reviews":   f"{len(df):,}",
        "Avg Rating":      f"{df['Rating'].mean():.2f} / 5.0",
        "Recommend Rate":  f"{df['Recommended_IND'].mean()*100:.1f}%",
        "Top Department":  df["Department_Name"].value_counts().index[0],
        "Top Category":    df["Class_Name"].value_counts().index[0],
        "Best ML Model":   f"{best_name_summary} ({results[best_name_summary]['acc']*100:.2f}%)",
        "Avg Review Length": f"{df['Review_Length'].mean():.0f} words",
        "Most Common Rating": f"{int(df['Rating'].mode()[0])} stars",
    }

    col1, col2 = st.columns(2)
    items = list(summary_data.items())
    for i, (k, v) in enumerate(items):
        col = col1 if i % 2 == 0 else col2
        col.markdown(f'<div class="metric-card" style="margin-bottom:12px"><h2>{v}</h2><p>{k}</p></div>',
                     unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Key Insights & Findings")
    positive_pct = (df["Sentiment"]=="Positive").mean()*100
    recommend_pct = df['Recommended_IND'].mean()*100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Positive Reviews", f"{positive_pct:.1f}%", delta="High Satisfaction")
    with col2:
        st.metric("Recommendation Rate", f"{recommend_pct:.1f}%", delta="Strong Trust")
    with col3:
        st.metric("Best ML Accuracy", f"{results[best_name_summary]['acc']*100:.2f}%", delta="Excellent")
    
    st.markdown(f"""
    #### 📌 Strategic Findings
    
    - **Customer Satisfaction**: {positive_pct:.1f}% of reviews are positive, indicating high overall satisfaction with products
    - **Product Recommendations**: {recommend_pct:.1f}% of customers would recommend products, showing strong brand loyalty
    - **Top Department**: The **{df['Department_Name'].value_counts().index[0]}** department drives the most reviews ({df['Department_Name'].value_counts().values[0]:,} reviews)
    - **Predictive Power**: **{best_name_summary}** model achieves {results[best_name_summary]['acc']*100:.2f}% accuracy in predicting recommendations
    - **Customer Demographics**: Average customer age is **{df['Age'].mean():.1f} years** with an average review length of **{df['Review_Length'].mean():.0f} words**
    - **Sentiment Balance**: Predominantly positive sentiment corpus with constructive feedback for continuous improvement
    
    #### 🎯 Recommendations
    
    - Focus marketing efforts on high-performing departments
    - Leverage positive reviews for social proof and credibility
    - Use ML predictions to proactively identify at-risk customers
    - Engage with reviewers to build community and loyalty
    """)

    st.markdown("---")
    st.caption("🚀 GlamTrends Analytics Dashboard | Brainybeam Info-Tech | Advanced E-Commerce Insights")
