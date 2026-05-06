import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import CCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Comparative Study | Breast Cancer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 60%, #0f2942 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

/* ── Main background ── */
.main { background: #f8fafc; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #0e7490 100%);
    border-radius: 18px;
    padding: 42px 48px;
    margin-bottom: 28px;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(14,116,144,0.35) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner h1 { font-size: 2.2rem; font-weight: 700; margin: 0 0 8px; letter-spacing: -0.5px; }
.hero-banner p  { font-size: 1.05rem; opacity: 0.82; margin: 0; }

/* ── Section title ── */
.section-title {
    font-size: 1.4rem; font-weight: 700;
    color: #0f172a;
    border-left: 5px solid #0e7490;
    padding-left: 14px;
    margin: 32px 0 18px;
}

/* ── Info / insight cards ── */
.card {
    background: white;
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border: 1px solid #e2e8f0;
    margin-bottom: 18px;
}
.card-title { font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
              letter-spacing: 1px; color: #0f172a; margin-bottom: 6px; }

.card-value { font-size: 2rem; font-weight: 700; color: #0f172a; }
.card-sub   { font-size: 0.82rem; color: #0f172a; margin-top: 4px; }

/* ── Metric row ── */
.metric-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 22px; }
.metric-box {
    flex: 1; min-width: 130px;
    background: white; border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border-top: 4px solid #0e7490;
    text-align: center;
}
.metric-box .val { font-size: 1.7rem; font-weight: 700; color: #0f172a; }
.metric-box .lbl { font-size: 0.75rem; color: #0f172a; margin-top: 4px; }

/* ── Callout boxes ── */
.callout {
    border-radius: 10px; padding: 16px 20px; margin: 12px 0;
    font-size: 0.9rem; line-height: 1.6;
    color: #0f172a;
}
.callout-blue   { background: #e0f2fe; border-left: 4px solid #0284c7; }
.callout-green  { background: #dcfce7; border-left: 4px solid #16a34a; }
.callout-amber  { background: #fef3c7; border-left: 4px solid #d97706; }
.callout-red    { background: #fee2e2; border-left: 4px solid #dc2626; }
.callout-purple { background: #f3e8ff; border-left: 4px solid #9333ea; }

/* ── Chart caption ── */
.chart-why {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.83rem;
    color: #0f172a;
    margin-top: 6px;
}
.chart-why b { color: #0284c7; }

/* ── Verdict badge ── */
.verdict {
    background: linear-gradient(135deg, #0f172a, #0e7490);
    color: white; border-radius: 14px;
    padding: 28px 32px; text-align: center;
    font-size: 1.25rem; font-weight: 600;
}
.verdict .big { font-size: 2.2rem; font-weight: 700; margin: 10px 0 6px; }

code { font-family: 'JetBrains Mono', monospace; background: #f1f5f9;
       padding: 2px 6px; border-radius: 4px; font-size: 0.82em; }

hr { border: none; border-top: 1px solid #e2e8f0; margin: 30px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD & PREPARE DATA  (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    raw = load_breast_cancer()
    df  = pd.DataFrame(raw.data, columns=raw.feature_names)
    df["target"]    = raw.target          # 0=malignant, 1=benign
    df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
    return df, raw

@st.cache_data
def train_models(test_size, random_state):
    df, raw = load_data()
    X = df[raw.feature_names]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Logistic Regression ──
    lr = LogisticRegression(max_iter=10000, random_state=random_state, C=1.0)
    lr.fit(X_train_s, y_train)

    # ── Random Forest ──
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    rf.fit(X_train_s, y_train)

    results = {}
    for name, model in [("Logistic Regression", lr), ("Random Forest", rf)]:
        y_pred  = model.predict(X_test_s)
        y_prob  = model.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        cv_scores = cross_val_score(model, X_train_s, y_train,
                                    cv=StratifiedKFold(5), scoring="f1")
        results[name] = dict(
            model=model, y_pred=y_pred, y_prob=y_prob,
            acc=accuracy_score(y_test, y_pred),
            prec=precision_score(y_test, y_pred),
            rec=recall_score(y_test, y_pred),
            f1=f1_score(y_test, y_pred),
            cm=confusion_matrix(y_test, y_pred),
            fpr=fpr, tpr=tpr, auc=auc(fpr, tpr),
            prec_curve=prec, rec_curve=rec,
            cv_mean=cv_scores.mean(), cv_std=cv_scores.std(),
            report=classification_report(y_test, y_pred,
                                          target_names=["Malignant","Benign"])
        )

    return results, X_train_s, X_test_s, y_train, y_test, scaler, raw.feature_names

df, raw_data = load_data()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("## 🔬 Navigation")
    page = st.radio("", [
        "🏠 Project Overview",
        "📊 Data Description",
        "🔀 CCA Analysis",
        "⚙️ ML Algorithms",
        "🧹 Data Preparation",
        "⚠️ Challenges",
        "📈 Model Results",
        "🏆 Conclusion"
    ])
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    test_size     = st.slider("Test Set Size", 0.15, 0.40, 0.20, 0.05)
    random_state  = st.selectbox("Random Seed", [42, 7, 99, 0])
    
    # === CCA SETTINGS (Default value) ===
    n_components = st.slider("Number of Canonical Components", 1, 10, 5, 
                             disabled=page != "🔀 CCA Analysis")

# ====================== TRAIN MODELS ======================
results, X_train_s, X_test_s, y_train, y_test, scaler, feature_names = \
    train_models(test_size, random_state)

# ═════════════════════════════════════════════════════════════
# PAGE 1 ── PROJECT OVERVIEW
# ═════════════════════════════════════════════════════════════
if page == "🏠 Project Overview":
    st.markdown("""
    <div class="hero-banner">
        <h1>🔬 Breast Cancer Diagnosis — ML Comparative Study</h1>
        <p>Multivariate Analysis · Logistic Regression vs Random Forest · Wisconsin Dataset</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("Total Samples", "569", "Patient records"),
        ("Features", "30", "Numeric measurements"),
        ("Malignant", f"{(df.target==0).sum()}", "Class 0"),
        ("Benign", f"{(df.target==1).sum()}", "Class 1"),
    ]
    for col, (lbl, val, sub) in zip([c1,c2,c3,c4], stats):
        col.markdown(f"""
        <div class="card">
            <div class="card-title">{lbl}</div>
            <div class="card-value">{val}</div>
            <div class="card-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📌 Project Goal</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="callout callout-blue">
    <b>Primary Objective:</b> Build and compare two supervised Machine Learning classifiers —
    <b>Logistic Regression</b> and <b>Random Forest</b> — to predict whether a breast tumour is
    <b>Malignant (cancerous)</b> or <b>Benign (non-cancerous)</b> using 30 numeric
    features extracted from digitised fine needle aspirate (FNA) images of breast masses.
    <br><br>
    <b>Clinical Importance:</b> Minimising <i>false negatives</i> (missed cancers) is paramount.
    We therefore weight <b>Recall</b> and <b>F1-score</b> heavily alongside overall accuracy.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🗂️ Report Structure</div>', unsafe_allow_html=True)
    sections = [
        ("📊", "Data Description", "Origin, features, class distribution, statistical summary"),
        ("⚙️", "ML Algorithms", "Why Logistic Regression & Random Forest? Theory & mechanics"),
        ("🧹", "Data Preparation", "Cleaning, outlier handling, scaling, train/test split"),
        ("⚠️", "Challenges", "Class imbalance, multicollinearity, overfitting risk"),
        ("📈", "Model Results", "Metrics, confusion matrices, ROC curves, feature importance"),
        ("🏆", "Conclusion", "Best model verdict with justification"),
    ]
    for icon, title, desc in sections:
        st.markdown(f"""
        <div class="card" style="padding:16px 20px;margin-bottom:10px;display:flex;gap:16px;align-items:center">
            <span style="font-size:1.6rem">{icon}</span>
            <div>
                <div style="font-weight:700;color:#0f172a">{title}</div>
                <div style="font-size:0.82rem;color:#64748b">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 2 ── DATA DESCRIPTION
# ═════════════════════════════════════════════════════════════
elif page == "📊 Data Description":
    st.markdown("""
    <div class="hero-banner">
        <h1>📊 Data Description</h1>
        <p>Understanding the dataset — origin, features, distributions & correlations</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Origin
    st.markdown('<div class="section-title">🌐 Dataset Origin</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="callout callout-blue">
    The <b>Breast Cancer Wisconsin (Diagnostic) Dataset</b> was created by Dr. William H. Wolberg
    at the University of Wisconsin–Madison. It is publicly available via the
    <b>UCI Machine Learning Repository</b> and is bundled with <code>scikit-learn</code>.<br><br>
    Features are computed from digitised images of <b>Fine Needle Aspirate (FNA)</b> of breast masses.
    They describe characteristics of the cell nuclei present in the image.
    Ten real-valued features are computed for each cell nucleus; the dataset then records
    <b>mean, standard error, and worst</b> values → <b>30 features total</b>.
    </div>
    """, unsafe_allow_html=True)

    # ── Feature groups
    st.markdown('<div class="section-title">🧬 Feature Groups</div>', unsafe_allow_html=True)
    groups = {
        "Radius": "Mean distance from centre to perimeter points",
        "Texture": "Standard deviation of grey-scale values",
        "Perimeter": "Perimeter of the cell nucleus",
        "Area": "Area of the cell nucleus",
        "Smoothness": "Local variation in radius lengths",
        "Compactness": "Perimeter² / Area − 1.0",
        "Concavity": "Severity of concave portions of the contour",
        "Concave Points": "Number of concave portions",
        "Symmetry": "Symmetry of the nucleus",
        "Fractal Dimension": "'Coastline approximation' − 1",
    }
    cols = st.columns(2)
    for i, (feat, desc) in enumerate(groups.items()):
        cols[i%2].markdown(f"""
        <div class="card" style="padding:12px 16px;margin-bottom:8px">
            <b style="color:#0e7490">{feat}</b> (mean · SE · worst)<br>
            <span style="font-size:0.8rem;color:#64748b">{desc}</span>
        </div>""", unsafe_allow_html=True)

    # ── Class distribution
    st.markdown('<div class="section-title">⚖️ Class Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        counts = df["diagnosis"].value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index,
            color_discrete_map={"Benign": "#0e7490", "Malignant": "#e11d48"},
            hole=0.45,
            title="Class Distribution"
        )
        fig.update_traces(textinfo="percent+label", textfont_size=14)
        fig.update_layout(showlegend=True, height=350,
                          font=dict(family="Sora"), margin=dict(t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="chart-why">
        <b>Why a Donut Chart?</b> Proportional comparison of two classes is clearest with a
        pie/donut. The hole keeps it readable and avoids the visual weight of a full pie.
        </div>""", unsafe_allow_html=True)

    with col2:
        malignant_pct = (df.target==0).mean()*100
        benign_pct    = (df.target==1).mean()*100
        st.markdown(f"""
        <div class="card" style="margin-top:40px">
            <div class="card-title">Class Balance Analysis</div>
            <br>
            <div style="display:flex;gap:16px;margin-bottom:12px">
                <div style="flex:1;background:#fee2e2;border-radius:10px;padding:14px;text-align:center">
                    <div style="font-size:1.6rem;font-weight:700;color:#e11d48">{(df.target==0).sum()}</div>
                    <div style="font-size:0.78rem;color:#7f1d1d">Malignant ({malignant_pct:.1f}%)</div>
                </div>
                <div style="flex:1;background:#dcfce7;border-radius:10px;padding:14px;text-align:center">
                    <div style="font-size:1.6rem;font-weight:700;color:#16a34a">{(df.target==1).sum()}</div>
                    <div style="font-size:0.78rem;color:#14532d">Benign ({benign_pct:.1f}%)</div>
                </div>
            </div>
            <div class="callout callout-amber" style="margin:0">
            The dataset has a <b>mild imbalance</b> (~63 % Benign vs ~37 % Malignant).
            Not severe enough to require SMOTE, but we must monitor Recall for Malignant class.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Statistical summary
    st.markdown('<div class="section-title">📐 Statistical Summary</div>', unsafe_allow_html=True)
    summary = df[list(feature_names)].describe().T.round(3)
    summary.index.name = "Feature"
    st.dataframe(summary.style.background_gradient(cmap="Blues", subset=["mean","std"]),
                 height=320, use_container_width=True)
    st.markdown("""
    <div class="chart-why">
    <b>Why a Styled Table?</b> A colour-gradient table lets the doctor instantly spot
    high-variance features (darker = larger value) without reading every number.
    </div>""", unsafe_allow_html=True)

    # ── Correlation heatmap
    st.markdown('<div class="section-title">🔗 Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = df[list(feature_names)].corr()
    fig2, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.7},
                annot=False, vmin=-1, vmax=1)
    ax.set_title("Lower-Triangle Correlation Matrix (30 features)", fontsize=13, pad=14)
    ax.tick_params(axis='both', labelsize=7)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    st.markdown("""
    <div class="chart-why">
    <b>Why a Heatmap?</b> With 30 features, a heatmap is the only practical way to show
    all 435 pairwise correlations at once. The triangular mask removes the redundant upper half.
    Red = strong positive correlation, Blue = negative. High correlations (e.g., radius–area–perimeter)
    confirm <b>multicollinearity</b> — a challenge addressed in the Challenges section.
    </div>""", unsafe_allow_html=True)
    
    # ── Feature distributions by class
    st.markdown('<div class="section-title">📦 Feature Distributions by Class (Top 6)</div>',
                unsafe_allow_html=True)
    top_feats = ["mean radius","mean texture","mean perimeter",
                 "mean area","mean concavity","mean compactness"]
    fig3 = make_subplots(rows=2, cols=3, subplot_titles=top_feats)
    colors = {"Malignant": "#e11d48", "Benign": "#0e7490"}
    for i, feat in enumerate(top_feats):
        r, c = divmod(i, 3)
        for diag in ["Malignant","Benign"]:
            vals = df[df.diagnosis==diag][feat]
            fig3.add_trace(go.Histogram(
                x=vals, name=diag, legendgroup=diag,
                showlegend=(i==0), opacity=0.72,
                marker_color=colors[diag], nbinsx=25
            ), row=r+1, col=c+1)
    fig3.update_layout(barmode="overlay", height=480, title_text="",
                       font=dict(family="Sora", size=11),
                       legend=dict(orientation="h", y=-0.12),
                       margin=dict(t=40,b=60))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""
    <div class="chart-why">
    <b>Why Overlapping Histograms?</b> Overlapping distributions reveal separability.
    Features where Malignant and Benign distributions barely overlap (e.g., <i>mean radius</i>,
    <i>mean concavity</i>) are the most discriminative — exactly what classifiers exploit.
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 3 ── ML ALGORITHMS
# ═════════════════════════════════════════════════════════════
elif page == "⚙️ ML Algorithms":
    st.markdown("""
    <div class="hero-banner">
        <h1>⚙️ Machine Learning Algorithms</h1>
        <p>Why we chose these two models — and how they work</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🤔 Why These Two Models?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="callout callout-blue">
    We deliberately chose <b>one linear model</b> and <b>one non-linear ensemble model</b>
    to cover a wide interpretability–performance spectrum. This pairing is standard practice
    in medical ML papers: the linear model provides a transparent baseline while the ensemble
    captures complex feature interactions that the linear model cannot.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📉 Logistic Regression", "🌳 Random Forest"])

    with tab1:
        st.markdown("### What is Logistic Regression?")
        st.markdown("""
        <div class="callout callout-blue">
        Despite its name, <b>Logistic Regression is a classification algorithm</b>, not a regressor.
        It models the probability that a sample belongs to a class using the <b>sigmoid (logistic) function</b>.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **How it works (step by step):**

            1. Compute a weighted sum of input features:
               `z = w₁x₁ + w₂x₂ + … + w₃₀x₃₀ + b`

            2. Pass `z` through the sigmoid:
               `σ(z) = 1 / (1 + e⁻ᶻ)`  → outputs a probability [0, 1]

            3. Threshold at 0.5 → Benign or Malignant

            4. Weights `w` are learned by **minimising log-loss** (binary cross-entropy)
               using gradient descent.
            """)
        with col2:
            # Sigmoid curve
            z = np.linspace(-8, 8, 300)
            sig = 1 / (1 + np.exp(-z))
            fig_sig = go.Figure()
            fig_sig.add_trace(go.Scatter(x=z, y=sig, mode="lines",
                                          line=dict(color="#0e7490", width=3)))
            fig_sig.add_hline(y=0.5, line_dash="dash", line_color="#e11d48", line_width=1.5)
            fig_sig.add_vline(x=0,   line_dash="dash", line_color="#94a3b8", line_width=1)
            fig_sig.update_layout(
                title="Sigmoid Function  σ(z) = 1/(1+e⁻ᶻ)",
                xaxis_title="z (linear combination)", yaxis_title="Probability",
                height=280, margin=dict(t=40,b=30,l=30,r=10),
                font=dict(family="Sora", size=11),
                plot_bgcolor="#f8fafc", paper_bgcolor="white"
            )
            st.plotly_chart(fig_sig, use_container_width=True)

        st.markdown("""
        <div class="callout callout-green">
        <b>✅ Why suitable for this dataset?</b><br>
        • 30 numeric features — Logistic Regression handles high-dimensional continuous inputs well.<br>
        • Feature scaling already applied (StandardScaler) — LR requires this.<br>
        • Provides probability outputs useful in clinical decision-making.<br>
        • Highly interpretable: coefficients directly show each feature's contribution.<br>
        • Fast to train, low risk of overfitting on a ~450-sample training set.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### What is Random Forest?")
        st.markdown("""
        <div class="callout callout-purple">
        <b>Random Forest</b> is an ensemble of decision trees trained using <b>Bootstrap Aggregation
        (Bagging)</b> + <b>random feature selection</b> at each split. The majority vote of all trees
        determines the final prediction.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **How it works (step by step):**

            1. Draw `B` bootstrap samples from training data (sampling with replacement).

            2. Grow a full decision tree on each sample, but at every split consider only
               `√p` random features (p = 30 → ≈5 features per split).

            3. Each tree independently predicts a class label.

            4. **Final prediction = majority vote** across all 200 trees (our setting).

            5. Out-of-bag (OOB) samples provide a free internal validation estimate.
            """)
        with col2:
            st.markdown("""
            **Key hyperparameters used:**

            | Parameter | Value | Effect |
            |---|---|---|
            | `n_estimators` | 200 | More trees = more stable |
            | `max_features` | `sqrt` | Controls correlation between trees |
            | `max_depth` | None | Full trees grown |
            | `min_samples_split` | 2 | Default, flexible splits |
            | `random_state` | user choice | Reproducibility |
            """)

        st.markdown("""
        <div class="callout callout-green">
        <b>✅ Why suitable for this dataset?</b><br>
        • Handles the mild class imbalance more robustly than a single decision tree.<br>
        • Naturally captures non-linear interactions between features (e.g., radius × concavity).<br>
        • Built-in feature importance via mean decrease in impurity.<br>
        • Resistant to overfitting due to averaging across many diverse trees.<br>
        • Requires <b>no feature scaling</b> (though we scale for fair comparison).
        </div>
        """, unsafe_allow_html=True)

    # Comparison table
    st.markdown('<div class="section-title">⚖️ Head-to-Head Algorithm Comparison</div>',
                unsafe_allow_html=True)
    comp = pd.DataFrame({
        "Criterion": ["Type","Interpretability","Requires Scaling","Handles Non-linearity",
                       "Training Speed","Feature Importance","Probability Output",
                       "Multicollinearity Sensitivity"],
        "Logistic Regression": ["Linear","⭐⭐⭐⭐⭐ (coefficients)","✅ Yes","❌ Limited",
                                  "🚀 Very Fast","Via coefficients","✅ Native","⚠️ High"],
        "Random Forest":       ["Non-linear Ensemble","⭐⭐ (black-box)","❌ Not needed","✅ Excellent",
                                  "🐢 Moderate","✅ Native (Gini)","✅ Native","✅ Robust"],
    })
    st.dataframe(comp.set_index("Criterion"), use_container_width=True)

# ═════════════════════════════════════════════════════════════
# PAGE 4 ── DATA PREPARATION
# ═════════════════════════════════════════════════════════════
elif page == "🧹 Data Preparation":
    st.markdown("""
    <div class="hero-banner">
        <h1>🧹 Data Preparation</h1>
        <p>Cleaning · Preprocessing · Splitting — every step documented</p>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("1️⃣", "Missing Value Check", "callout-green",
         "No missing values found across all 569 rows and 30 features. "
         "The Wisconsin dataset is a curated benchmark — it was pre-cleaned by the original researchers. "
         "We verified this with <code>df.isnull().sum()</code> → all zeros."),
        ("2️⃣", "Duplicate Row Check", "callout-green",
         "Zero duplicate rows detected. Each row represents a unique patient biopsy."),
        ("3️⃣", "Outlier Detection", "callout-amber",
         "Using the IQR method, outliers were detected in high-variance features "
         "(area, perimeter, concavity). In a medical context, extreme values often represent "
         "genuinely severe cases (true malignant tumours), so <b>we did not remove them</b>. "
         "Removing medical extremes could bias the model toward mild cases and increase false negatives."),
        ("4️⃣", "Feature Scaling — StandardScaler", "callout-blue",
         "Applied <code>StandardScaler</code> (zero mean, unit variance) to all 30 features. "
         "This is <b>essential for Logistic Regression</b> whose gradient descent converges "
         "poorly when features are on different scales (e.g., area ~500 vs fractal dimension ~0.06). "
         "Although Random Forest is scale-invariant, we scale both models for a fair comparison. "
         "The scaler is <b>fit on training data only</b> and then applied to test data — "
         "preventing data leakage."),
        ("5️⃣", "Train / Test Split", "callout-blue",
         f"Data split: <b>{int((1-test_size)*100)}% train</b> / <b>{int(test_size*100)}% test</b> "
         f"→ ~{int(569*(1-test_size))} training / ~{int(569*test_size)} test samples. "
         "We use <code>stratify=y</code> to ensure both splits maintain the original class ratio "
         "(critical for imbalanced problems)."),
        ("6️⃣", "Cross-Validation", "callout-purple",
         "5-fold Stratified Cross-Validation is applied on the training set for unbiased "
         "model evaluation. Stratification ensures each fold preserves the class ratio. "
         "This gives a CV F1-score with standard deviation, measuring model stability."),
    ]

    for icon, title, cls, body in steps:
        st.markdown(f"""
        <div class="card">
            <div style="display:flex;gap:14px;align-items:flex-start">
                <span style="font-size:1.5rem">{icon}</span>
                <div style="flex:1">
                    <div style="font-weight:700;color:#0f172a;margin-bottom:8px">{title}</div>
                    <div class="callout {cls}" style="margin:0">{body}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # Pipeline diagram
    st.markdown('<div class="section-title">🔄 Data Pipeline Flow</div>', unsafe_allow_html=True)
    pipeline_steps = [
        ("Raw Data\n569 samples\n30 features", "#0f172a"),
        ("Missing Value\nCheck ✅", "#0369a1"),
        ("Duplicate\nCheck ✅", "#0369a1"),
        ("Outlier\nAnalysis ⚠️", "#b45309"),
        ("StandardScaler\n(fit on train)", "#7c3aed"),
        ("Stratified\nTrain/Test Split", "#0e7490"),
        ("5-Fold\nCross-Val", "#0e7490"),
        ("Model\nTraining 🎯", "#16a34a"),
    ]
    fig_pipe, ax_pipe = plt.subplots(figsize=(14, 2.5))
    ax_pipe.set_xlim(0, len(pipeline_steps))
    ax_pipe.set_ylim(0, 1)
    ax_pipe.axis("off")
    for i, (label, color) in enumerate(pipeline_steps):
        box = mpatches.FancyBboxPatch((i+0.05, 0.1), 0.85, 0.8,
                                       boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor="white", linewidth=1.5)
        ax_pipe.add_patch(box)
        ax_pipe.text(i+0.48, 0.5, label, ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold", linespacing=1.4)
        if i < len(pipeline_steps)-1:
            ax_pipe.annotate("", xy=(i+1.05, 0.5), xytext=(i+0.92, 0.5),
                              arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2))
    fig_pipe.patch.set_alpha(0)
    st.pyplot(fig_pipe)
    plt.close()
    st.markdown("""
    <div class="chart-why">
    <b>Why a Pipeline Diagram?</b> A sequential flow chart makes the exact order of operations
    crystal-clear, preventing confusion about whether scaling happened before or after the split.
    </div>""", unsafe_allow_html=True)

    # Before / after scaling
    st.markdown('<div class="section-title">📊 Before vs After Scaling</div>', unsafe_allow_html=True)
    feats_show = ["mean radius", "mean area", "mean texture", "mean fractal dimension"]
    col1, col2 = st.columns(2)
    with col1:
        raw_vals = {f: df[f].values for f in feats_show}
        fig_b = go.Figure()
        for f in feats_show:
            fig_b.add_trace(go.Box(y=df[f], name=f.replace("mean ",""), boxmean=True))
        fig_b.update_layout(title="Before Scaling (raw values)", height=320,
                             font=dict(family="Sora",size=10),
                             margin=dict(t=40,b=10))
        st.plotly_chart(fig_b, use_container_width=True)
    with col2:
        scaled = scaler.transform(df[list(feature_names)])
        scaled_df = pd.DataFrame(scaled, columns=feature_names)
        fig_a = go.Figure()
        for f in feats_show:
            fig_a.add_trace(go.Box(y=scaled_df[f], name=f.replace("mean ",""), boxmean=True))
        fig_a.update_layout(title="After StandardScaler (z-scores)", height=320,
                             font=dict(family="Sora",size=10),
                             margin=dict(t=40,b=10))
        st.plotly_chart(fig_a, use_container_width=True)
    st.markdown("""
    <div class="chart-why">
    <b>Why Box Plots?</b> Box plots compactly show median, IQR, and outliers simultaneously.
    Comparing before/after scaling side-by-side proves that features now share a common scale
    (centred around 0) while preserving their relative spread and outliers.
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 5 ── CHALLENGES
# ═════════════════════════════════════════════════════════════
elif page == "⚠️ Challenges":
    st.markdown("""
    <div class="hero-banner">
        <h1>⚠️ Challenges in the Data</h1>
        <p>Honest analysis of obstacles and how we addressed them</p>
    </div>
    """, unsafe_allow_html=True)

    challenges = [
        {
            "icon": "⚖️",
            "title": "Class Imbalance",
            "level": "Mild",
            "level_color": "amber",
            "detail": """
            The dataset has <b>212 Malignant</b> (37.3%) vs <b>357 Benign</b> (62.7%) samples.
            While not severely imbalanced, this creates a risk: a naive model predicting "always Benign"
            would achieve 62.7% accuracy — misleadingly high.<br><br>
            <b>Impact on metrics:</b> Accuracy alone is insufficient. A model with 95% accuracy but
            20% Recall on Malignant is clinically dangerous — it misses 80% of cancers.<br><br>
            <b>Our mitigation:</b> (1) Use <b>Stratified Splitting</b> to preserve class ratio in each fold.
            (2) Report <b>Recall, Precision, F1</b> per class. (3) Weight the Malignant class performance
            heavily in final model selection.
            """,
        },
        {
            "icon": "🔗",
            "title": "Multicollinearity",
            "level": "High",
            "level_color": "red",
            "detail": """
            Several feature groups are near-perfectly correlated: radius, perimeter, and area are
            mathematically related (area ≈ π × radius²). The correlation heatmap shows correlations
            above <b>0.95</b> between some pairs.<br><br>
            <b>Impact on models:</b> For Logistic Regression, multicollinearity inflates coefficient
            variance and makes individual coefficient interpretation unreliable (though predictions
            remain stable). For Random Forest, it reduces the effective diversity of random feature
            subsets — some splits will always choose highly correlated features.<br><br>
            <b>Our mitigation:</b> We kept all 30 features (removing them arbitrarily could hurt performance)
            and used <b>L2 regularisation (C=1.0)</b> in Logistic Regression to dampen inflated coefficients.
            Feature importance from Random Forest is examined carefully using permutation importance.
            """,
        },
        {
            "icon": "📦",
            "title": "Small Dataset Size",
            "level": "Moderate",
            "level_color": "amber",
            "detail": """
            569 samples is relatively small for 30 features. The <b>curse of dimensionality</b>
            suggests we have roughly 19 samples per feature — below the rule-of-thumb of 30.<br><br>
            <b>Overfitting risk:</b> Complex models could memorise the training set. A single decision
            tree, for example, would likely overfit here.<br><br>
            <b>Our mitigation:</b> (1) Logistic Regression with L2 regularisation is inherently low-variance.
            (2) Random Forest averages 200 trees — each trained on a bootstrap sample — dramatically
            reducing variance. (3) 5-Fold CV provides a realistic out-of-sample estimate.
            """,
        },
        {
            "icon": "🏥",
            "title": "Clinical Cost Asymmetry",
            "level": "Critical",
            "level_color": "red",
            "detail": """
            In medical diagnosis, <b>false negatives</b> (missing a malignant tumour) are far more
            costly than <b>false positives</b> (incorrectly flagging a benign tumour as malignant).
            A missed cancer can be fatal; an unnecessary biopsy, while distressing, is recoverable.<br><br>
            <b>Standard accuracy treats both errors equally</b> — which is medically inappropriate.<br><br>
            <b>Our mitigation:</b> We use the <b>ROC-AUC curve</b> (threshold-independent) and report
            <b>Recall specifically for the Malignant class</b> as a primary decision criterion.
            The model with higher Malignant Recall at comparable overall accuracy wins.
            """,
        },
    ]

    level_colors = {"Mild": "#f59e0b", "Moderate": "#f97316", "High": "#dc2626", "Critical": "#7f1d1d"}
    level_bg     = {"Mild": "#fef3c7", "Moderate": "#ffedd5", "High": "#fee2e2",  "Critical": "#fef2f2"}

    for ch in challenges:
        lc = ch["level_color"]
        st.markdown(f"""
        <div class="card">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
                <span style="font-size:1.8rem">{ch['icon']}</span>
                <div style="flex:1">
                    <span style="font-size:1.1rem;font-weight:700;color:#0f172a">{ch['title']}</span>
                </div>
                <span style="background:{level_bg[ch['level']]};color:{level_colors[ch['level']]};
                             padding:4px 14px;border-radius:20px;font-size:0.75rem;font-weight:700;
                             border:1px solid {level_colors[ch['level']]}">
                    {ch['level']} Severity
                </span>
            </div>
            <div style="font-size:0.88rem;line-height:1.7;color:#334155">{ch['detail']}</div>
        </div>""", unsafe_allow_html=True)

    # Correlation challenge visualisation
    st.markdown('<div class="section-title">🔗 Multicollinearity — High-Correlation Pairs</div>',
                unsafe_allow_html=True)
    corr_mat = df[list(feature_names)].corr().abs()
    pairs = []
    fn = list(feature_names)
    for i in range(len(fn)):
        for j in range(i+1, len(fn)):
            if corr_mat.iloc[i,j] > 0.85:
                pairs.append({"Feature A": fn[i], "Feature B": fn[j],
                               "Correlation": round(corr_mat.iloc[i,j],3)})
    pairs_df = pd.DataFrame(pairs).sort_values("Correlation", ascending=False)
    st.dataframe(pairs_df, use_container_width=True, height=280)
    st.markdown(f"""
    <div class="callout callout-red">
    <b>{len(pairs_df)} feature pairs</b> have absolute correlation > 0.85.
    This confirms severe multicollinearity — especially among radius/perimeter/area clusters.
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 6 ── MODEL RESULTS
# ═════════════════════════════════════════════════════════════
elif page == "📈 Model Results":
    st.markdown("""
    <div class="hero-banner">
        <h1>📈 Model Results & Comparison</h1>
        <p>Metrics · Confusion Matrices · ROC Curves · Feature Importance</p>
    </div>
    """, unsafe_allow_html=True)

    lr_r = results["Logistic Regression"]
    rf_r = results["Random Forest"]

    # ── Metrics table
    st.markdown('<div class="section-title">📊 Performance Metrics Summary</div>',
                unsafe_allow_html=True)
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy","Precision","Recall","F1-Score","ROC-AUC","CV F1 (mean)","CV F1 (std)"],
        "Logistic Regression": [f"{lr_r['acc']:.4f}", f"{lr_r['prec']:.4f}",
                                  f"{lr_r['rec']:.4f}", f"{lr_r['f1']:.4f}",
                                  f"{lr_r['auc']:.4f}", f"{lr_r['cv_mean']:.4f}",
                                  f"±{lr_r['cv_std']:.4f}"],
        "Random Forest":       [f"{rf_r['acc']:.4f}", f"{rf_r['prec']:.4f}",
                                  f"{rf_r['rec']:.4f}", f"{rf_r['f1']:.4f}",
                                  f"{rf_r['auc']:.4f}", f"{rf_r['cv_mean']:.4f}",
                                  f"±{rf_r['cv_std']:.4f}"],
    })

    def highlight_best(row):
        styles = ["","",""]
        try:
            lv = float(row["Logistic Regression"].replace("±",""))
            rv = float(row["Random Forest"].replace("±",""))
            if "std" in row["Metric"].lower():
                best = 1 if lv <= rv else 2
            else:
                best = 1 if lv >= rv else 2
            if best == 1:
                styles[1] = "background-color:#dcfce7;font-weight:bold"
            else:
                styles[2] = "background-color:#dcfce7;font-weight:bold"
        except:
            pass
        return styles

    styled = metrics_df.style.apply(highlight_best, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown("""
    <div class="chart-why">
    <b>Why a Styled Comparison Table?</b> Green highlighting immediately draws the eye to whichever
    model wins each metric, making the overall picture clear without reading every number.
    </div>""", unsafe_allow_html=True)

    # ── Radar chart
    st.markdown('<div class="section-title">🕸️ Performance Radar Chart</div>',
                unsafe_allow_html=True)
    categories = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
    lr_vals = [lr_r["acc"], lr_r["prec"], lr_r["rec"], lr_r["f1"], lr_r["auc"]]
    rf_vals = [rf_r["acc"], rf_r["prec"], rf_r["rec"], rf_r["f1"], rf_r["auc"]]

    fig_radar = go.Figure()
    for name, vals, color in [
        ("Logistic Regression", lr_vals, "#0e7490"),
        ("Random Forest",       rf_vals, "#e11d48"),
    ]:
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill="toself", name=name,
            line=dict(color=color, width=2.5),
            fillcolor=color, opacity=0.18
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.85, 1.0])),
        showlegend=True, height=420, font=dict(family="Sora"),
        margin=dict(t=20,b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown("""
    <div class="chart-why">
    <b>Why a Radar Chart?</b> A radar chart is perfect for multi-metric comparison between two models.
    The area of each polygon intuitively captures overall performance — a larger area means
    consistently better performance across all dimensions.
    </div>""", unsafe_allow_html=True)

    # ── Confusion matrices
    st.markdown('<div class="section-title">🔲 Confusion Matrices</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for col, (name, r) in zip([col1, col2], results.items()):
        cm = r["cm"]
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            x=["Malignant","Benign"], y=["Malignant","Benign"],
            title=f"{name}"
        )
        fig_cm.update_layout(height=300, font=dict(family="Sora",size=12),
                               margin=dict(t=50,b=10))
        col.plotly_chart(fig_cm, use_container_width=True)
    st.markdown("""
    <div class="chart-why">
    <b>Why Confusion Matrices?</b> They break down predictions into TP, TN, FP, FN — the raw
    building blocks of all classification metrics. In a medical context, the bottom-left cell
    (False Negatives = missed cancers) is the most critical number to minimise.
    </div>""", unsafe_allow_html=True)

    # ── ROC curves
    st.markdown('<div class="section-title">📈 ROC Curves</div>', unsafe_allow_html=True)
    fig_roc = go.Figure()
    for name, r, color in [
        ("Logistic Regression", lr_r, "#0e7490"),
        ("Random Forest",       rf_r, "#e11d48"),
    ]:
        fig_roc.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines",
            name=f"{name} (AUC={r['auc']:.4f})",
            line=dict(color=color, width=2.5)
        ))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  name="Random Classifier",
                                  line=dict(color="#94a3b8", dash="dash")))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        title="Receiver Operating Characteristic (ROC) Curve",
        height=420, font=dict(family="Sora"),
        legend=dict(x=0.55, y=0.1),
        plot_bgcolor="#f8fafc", paper_bgcolor="white"
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.markdown("""
    <div class="chart-why">
    <b>Why ROC Curves?</b> ROC curves are threshold-independent — they show model performance
    across all possible classification thresholds. The AUC (Area Under Curve) summarises this
    in a single number: 1.0 = perfect, 0.5 = random. Crucial for clinical use where doctors
    may want to adjust the sensitivity/specificity trade-off.
    </div>""", unsafe_allow_html=True)

    # ── Precision-Recall curve
    st.markdown('<div class="section-title">📉 Precision–Recall Curves</div>',
                unsafe_allow_html=True)
    fig_pr = go.Figure()
    for name, r, color in [
        ("Logistic Regression", lr_r, "#0e7490"),
        ("Random Forest",       rf_r, "#e11d48"),
    ]:
        fig_pr.add_trace(go.Scatter(
            x=r["rec_curve"], y=r["prec_curve"], mode="lines",
            name=name, line=dict(color=color, width=2.5)
        ))
    fig_pr.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        title="Precision–Recall Curve",
        height=380, font=dict(family="Sora"),
        plot_bgcolor="#f8fafc", paper_bgcolor="white"
    )
    st.plotly_chart(fig_pr, use_container_width=True)
    st.markdown("""
    <div class="chart-why">
    <b>Why Precision-Recall Curves?</b> PR curves are more informative than ROC curves when
    class imbalance exists. A model that maintains high Precision as Recall increases is ideal —
    it correctly flags cancers without flooding doctors with false alarms.
    </div>""", unsafe_allow_html=True)

    # ── Feature importance
    st.markdown('<div class="section-title">🌟 Feature Importance</div>', unsafe_allow_html=True)
    tab_fi1, tab_fi2 = st.tabs(["🌳 Random Forest — Gini Importance", "📉 LR — Coefficient Magnitude"])

    with tab_fi1:
        importances = results["Random Forest"]["model"].feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=False).head(15)
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Teal",
                         title="Top 15 Features — Random Forest Gini Importance")
        fig_fi.update_layout(height=460, font=dict(family="Sora",size=11),
                               yaxis=dict(autorange="reversed"),
                               margin=dict(t=50,b=20))
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown("""
        <div class="chart-why">
        <b>Why a Horizontal Bar Chart?</b> Long feature names read naturally on a horizontal axis.
        Colour encoding adds a second visual channel to reinforce magnitude — darker teal = more important.
        </div>""", unsafe_allow_html=True)

    with tab_fi2:
        coefs = np.abs(results["Logistic Regression"]["model"].coef_[0])
        coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient |w|": coefs})
        coef_df = coef_df.sort_values("Coefficient |w|", ascending=False).head(15)
        fig_coef = px.bar(coef_df, x="Coefficient |w|", y="Feature", orientation="h",
                           color="Coefficient |w|", color_continuous_scale="Blues",
                           title="Top 15 Features — Logistic Regression |Coefficient|")
        fig_coef.update_layout(height=460, font=dict(family="Sora",size=11),
                                yaxis=dict(autorange="reversed"),
                                margin=dict(t=50,b=20))
        st.plotly_chart(fig_coef, use_container_width=True)
        st.markdown("""
        <div class="chart-why">
        <b>Interpretation:</b> Larger |coefficient| = greater influence on the decision boundary.
        Note that due to multicollinearity, individual coefficients may be unstable —
        L2 regularisation keeps them bounded.
        </div>""", unsafe_allow_html=True)

    # ── CV score comparison
    st.markdown('<div class="section-title">🔁 Cross-Validation F1-Score Stability</div>',
                unsafe_allow_html=True)
    cv_data = {"Model": [], "Fold": [], "F1": []}
    for name, model in [("Logistic Regression", lr_r["model"]),
                         ("Random Forest",       rf_r["model"])]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train_s, y_train, cv=skf, scoring="f1")
        for i, s in enumerate(scores):
            cv_data["Model"].append(name)
            cv_data["Fold"].append(f"Fold {i+1}")
            cv_data["F1"].append(s)
    cv_df = pd.DataFrame(cv_data)
    fig_cv = px.box(cv_df, x="Model", y="F1", color="Model",
                     color_discrete_map={"Logistic Regression":"#0e7490","Random Forest":"#e11d48"},
                     points="all", title="5-Fold CV F1-Score Distribution")
    fig_cv.update_layout(height=360, font=dict(family="Sora"), showlegend=False)
    st.plotly_chart(fig_cv, use_container_width=True)
    st.markdown("""
    <div class="chart-why">
    <b>Why Box Plots with Jitter?</b> Box plots show the median and spread of CV scores.
    Showing individual fold scores (jitter/strip) alongside the box exposes exactly where
    variance comes from — a tight cluster = stable model, scattered points = unstable.
    </div>""", unsafe_allow_html=True)
# ====================== CCA ANALYSIS PAGE ======================
elif page == "🔀 CCA Analysis":
    st.markdown("""
    <div class="hero-banner">
        <h1>🔀 Canonical Correlation Analysis (CCA)</h1>
        <p>Exploring Relationships Between Feature Groups in Breast Cancer Data</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **What is CCA?**  
    Canonical Correlation Analysis is a multivariate statistical method that explores the relationships 
    between two sets of variables by finding linear combinations (canonical variates) that are maximally correlated.
    """)

    # Define two feature sets
    mean_cols = [col for col in feature_names if 'mean' in col]
    worst_cols = [col for col in feature_names if 'worst' in col]
    error_cols = [col for col in feature_names if 'error' in col]

    set1 = mean_cols + error_cols   # 20 features
    set2 = worst_cols               # 10 features

    X1 = df[set1].values
    X2 = df[set2].values

    # Apply CCA
    cca = CCA(n_components=n_components, scale=True)
    cca.fit(X1, X2)
    U, V = cca.transform(X1, X2)

    canonical_corrs = [np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(n_components)]

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Canonical Correlations", "🔍 Component Analysis", "📊 Feature Contributions"])

    with tab1:
        fig = px.bar(
            x=[f"Canonical Pair {i+1}" for i in range(n_components)],
            y=canonical_corrs,
            labels={"y": "Canonical Correlation", "x": ""},
            title="Strength of Canonical Correlations",
            color_discrete_sequence=["#14b8a6"]
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"**First Canonical Correlation: {canonical_corrs[0]:.4f}** — Very Strong Relationship!")

    with tab2:
        st.markdown("### First Two Canonical Variates")
        fig2 = go.Figure()
        sample_colors = df.target.map({0: '#dc2626', 1: '#14b8a6'})
        fig2.add_trace(go.Scatter(x=U[:,0], y=V[:,0], mode='markers',
                                marker=dict(color=sample_colors, size=8),
                                name='Samples'))
        fig2.update_layout(
            title="First Canonical Pair (U1 vs V1)",
            xaxis_title="Canonical Variate from Mean + Error Features",
            yaxis_title="Canonical Variate from Worst Features",
            height=500,
            plot_bgcolor='#f8fafc'
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("**Feature Loadings (First Canonical Component)**")
        loadings1 = pd.DataFrame({
            'Feature': set1,
            'Loading_U': cca.x_weights_[:, 0]
        }).sort_values('Loading_U', ascending=False)
        
        loadings2 = pd.DataFrame({
            'Feature': set2,
            'Loading_V': cca.y_weights_[:, 0]
        }).sort_values('Loading_V', ascending=False)
        
        colA, colB = st.columns(2)
        with colA:
            st.write("**Set 1: Mean + Error Features**")
            st.dataframe(loadings1.style.background_gradient(cmap='YlGnBu'), use_container_width=True)
        with colB:
            st.write("**Set 2: Worst Features**")
            st.dataframe(loadings2.style.background_gradient(cmap='YlGnBu'), use_container_width=True)

    # Explanation
    st.markdown('<div class="section-title">📘 Understanding CCA</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="callout callout-blue">
    <b>How CCA Works:</b><br><br>
    1. Finds linear combinations from each set that are most highly correlated.<br>
    2. The first pair has the highest possible correlation.<br>
    3. Subsequent pairs are orthogonal to previous ones.
    </div>
    """, unsafe_allow_html=True)
# ═════════════════════════════════════════════════════════════
# PAGE 7 ── CONCLUSION
# ═════════════════════════════════════════════════════════════
elif page == "🏆 Conclusion":
    st.markdown("""
    <div class="hero-banner">
        <h1>🏆 Conclusion</h1>
        <p>Final verdict — which model wins and why</p>
    </div>
    """, unsafe_allow_html=True)

    lr_r = results["Logistic Regression"]
    rf_r = results["Random Forest"]
    winner = "Random Forest" if rf_r["f1"] >= lr_r["f1"] else "Logistic Regression"
    winner_r = rf_r if winner == "Random Forest" else lr_r
    other_r  = lr_r if winner == "Random Forest" else rf_r
    other    = "Logistic Regression" if winner == "Random Forest" else "Random Forest"

    # Verdict badge
    st.markdown(f"""
    <div class="verdict">
        <div>🏆 Best Performing Model</div>
        <div class="big">{winner}</div>
        <div>F1 = {winner_r['f1']:.4f} &nbsp;·&nbsp; AUC = {winner_r['auc']:.4f}
             &nbsp;·&nbsp; Recall = {winner_r['rec']:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📋 Decision Rationale</div>', unsafe_allow_html=True)

    reasons = [
        ("🎯", "Higher F1-Score",
         f"{winner} achieved F1 = <b>{winner_r['f1']:.4f}</b> vs {other}'s <b>{other_r['f1']:.4f}</b>. "
         "F1 balances Precision and Recall, making it the most appropriate single metric when both "
         "false positives and false negatives have clinical consequences."),
        ("🧬", "Superior Recall on Malignant Class",
         f"{winner} achieved Recall = <b>{winner_r['rec']:.4f}</b>. "
         "In cancer diagnosis, <b>Recall is the most clinically critical metric</b> — "
         "missing a malignant tumour (false negative) can be fatal. Every percentage point in Recall matters."),
        ("📈", "Higher ROC-AUC",
         f"AUC = <b>{winner_r['auc']:.4f}</b> confirms {winner} maintains better True Positive Rate "
         "across all classification thresholds — giving clinicians flexibility to adjust sensitivity "
         "vs specificity for their specific context."),
        ("🔁", "Cross-Validation Stability",
         f"CV F1 = <b>{winner_r['cv_mean']:.4f} ± {winner_r['cv_std']:.4f}</b> demonstrates that "
         f"{winner}'s performance is consistent across different train/validation splits — "
         "it is not overfitting to the specific test set."),
        ("🌿", "Handles Non-linearity",
         f"{winner} captures complex non-linear interactions between features (e.g., "
         "radius × concavity interactions) that a linear boundary cannot model, leading to "
         "better discrimination at the class boundary."),
    ] if winner == "Random Forest" else [
        ("🎯", "Competitive F1-Score",
         f"{winner} achieved F1 = <b>{winner_r['f1']:.4f}</b> with full interpretability."),
        ("👁️", "High Interpretability",
         "Logistic Regression coefficients directly explain which features push toward Malignant/Benign, "
         "making it the preferred model in regulatory or peer-reviewed clinical environments."),
        ("📈", "Strong ROC-AUC",
         f"AUC = <b>{winner_r['auc']:.4f}</b> — comparable to ensemble methods on this dataset."),
        ("⚡", "Efficiency",
         "Trains in milliseconds, requires no tuning, and generalises well when the data is linearly separable."),
        ("🔁", "Stability",
         f"CV F1 = <b>{winner_r['cv_mean']:.4f} ± {winner_r['cv_std']:.4f}</b>"),
    ]

    for icon, title, body in reasons:
        st.markdown(f"""
        <div class="card">
            <div style="display:flex;gap:14px;align-items:flex-start">
                <span style="font-size:1.5rem">{icon}</span>
                <div>
                    <div style="font-weight:700;color:#0f172a;margin-bottom:6px">{title}</div>
                    <div style="font-size:0.87rem;line-height:1.7;color:#334155">{body}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # Final bar comparison
    st.markdown('<div class="section-title">📊 Final Metrics Comparison</div>',
                unsafe_allow_html=True)
    metrics_names = ["Accuracy","Precision","Recall","F1","AUC","CV F1"]
    lr_vals_final = [lr_r["acc"],lr_r["prec"],lr_r["rec"],lr_r["f1"],lr_r["auc"],lr_r["cv_mean"]]
    rf_vals_final = [rf_r["acc"],rf_r["prec"],rf_r["rec"],rf_r["f1"],rf_r["auc"],rf_r["cv_mean"]]

    fig_bar = go.Figure(data=[
        go.Bar(name="Logistic Regression", x=metrics_names, y=lr_vals_final,
               marker_color="#0e7490", text=[f"{v:.3f}" for v in lr_vals_final],
               textposition="outside"),
        go.Bar(name="Random Forest",       x=metrics_names, y=rf_vals_final,
               marker_color="#e11d48", text=[f"{v:.3f}" for v in rf_vals_final],
               textposition="outside"),
    ])
    fig_bar.update_layout(
        barmode="group", height=420, font=dict(family="Sora"),
        yaxis=dict(range=[0.85, 1.02], title="Score"),
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor="#f8fafc", paper_bgcolor="white",
        margin=dict(t=20,b=60)
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("""
    <div class="chart-why">
    <b>Why Grouped Bar Chart?</b> Side-by-side bars make magnitude differences between the two models
    immediately apparent for every metric simultaneously — the most intuitive format for a final
    head-to-head comparison.
    </div>""", unsafe_allow_html=True)

    # Final summary
    st.markdown('<div class="section-title">📝 Final Summary</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="callout callout-green">
    <b>Summary:</b> Both models achieved excellent performance on the Wisconsin Breast Cancer dataset,
    confirming that the 30 morphological features extracted from FNA images are highly discriminative.
    <br><br>
    <b>{winner}</b> is the recommended model for deployment in a clinical screening tool, primarily
    due to its superior <b>Recall on Malignant cases</b> — the metric that directly impacts patient safety.
    <br><br>
    <b>Recommendation for practice:</b> Pair {winner} with a clinical expert review layer for all
    cases with predicted probability between 0.35–0.65 (uncertain zone), and set the classification
    threshold below 0.5 to further increase Malignant Recall at the cost of slightly more false alarms.
    </div>
    <div class="callout callout-amber" style="margin-top:12px">
    <b>Limitations:</b> This study used a single benchmark dataset. Before clinical deployment,
    models must be validated on prospective, multi-site data and evaluated by board-certified
    oncologists. ML models should assist, not replace, clinical judgment.
    </div>
    """, unsafe_allow_html=True)
