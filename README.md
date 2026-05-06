🔬 Breast Cancer Diagnosis — ML Comparative Study
A fully interactive Streamlit web application that performs a comprehensive multivariate analysis and comparison of two supervised Machine Learning classifiers — Logistic Regression and Random Forest — on the Wisconsin Breast Cancer Diagnostic Dataset.

📌 Project Overview
The primary objective of this project is to predict whether a breast tumour is Malignant (cancerous) or Benign (non-cancerous) using 30 numeric features extracted from digitised Fine Needle Aspirate (FNA) images of breast masses.
Given the clinical context, Recall and F1-score are weighted heavily alongside overall accuracy, as minimising false negatives (missed cancers) is paramount.

🖥️ App Pages
The app is structured as a multi-page interactive report with the following sections:
PageDescription🏠 Project OverviewGoals, dataset summary, and report structure📊 Data DescriptionFeature groups, class distribution, statistical summary, correlation heatmap🔀 CCA AnalysisCanonical Correlation Analysis between feature sets⚙️ ML AlgorithmsTheory and mechanics behind both classifiers🧹 Data PreparationPreprocessing pipeline — scaling, splitting, outlier handling⚠️ ChallengesClass imbalance, multicollinearity, overfitting risk📈 Model ResultsMetrics, confusion matrices, ROC curves, feature importance🏆 ConclusionBest model verdict with full clinical justification

📊 Dataset
Breast Cancer Wisconsin (Diagnostic) Dataset

Source: UCI Machine Learning Repository (bundled with scikit-learn)
Created by: Dr. William H. Wolberg, University of Wisconsin–Madison
Samples: 569 patient records
Features: 30 numeric measurements (mean, standard error, and worst values for 10 cell nucleus characteristics)
Classes: Malignant (212) · Benign (357)

Feature Groups
Each of the following is measured as mean, SE, and worst value:
Radius · Texture · Perimeter · Area · Smoothness · Compactness · Concavity · Concave Points · Symmetry · Fractal Dimension

🤖 Models
Logistic Regression

Regularisation: L2 (C=1.0)
Max iterations: 10,000
Features standardised with StandardScaler

Random Forest

200 estimators
Parallelised training (n_jobs=-1)
Features standardised with StandardScaler

Both models are evaluated with:

Accuracy, Precision, Recall, F1-Score
ROC-AUC curve
5-Fold Stratified Cross-Validation
Confusion Matrix & Classification Report


⚙️ Sidebar Controls
Users can interactively adjust:

Test Set Size — slider from 15% to 40%
Random Seed — choose from [0, 7, 42, 99]
CCA Components — slider from 1 to 10 (active on CCA page)

All models retrain automatically on parameter change (cached for performance with @st.cache_data).

🚀 Getting Started
Prerequisites

Python 3.8+
pip

Installation
bash# 1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run MV.py
The app will open in your browser at http://localhost:8501.

📦 Dependencies
streamlit
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
Save these to a requirements.txt:
bashpip freeze > requirements.txt
Or create it manually:
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
scikit-learn>=1.3.0

📁 Project Structure
.
├── MV.py               # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

📈 Key Results
The app dynamically computes and displays a verdict based on your chosen settings. In the default configuration (80/20 split, seed=42), both models achieve >95% accuracy, with the winner determined by F1-score and Recall on the Malignant class.

⚠️ Clinical Disclaimer: This application is built for educational and research purposes. ML models should assist, not replace, clinical judgment. Any clinical deployment requires validation on prospective, multi-site data and review by board-certified oncologists.


🧪 Analysis Techniques

Canonical Correlation Analysis (CCA) — explores relationships between feature subsets (mean+error vs. worst features)
Permutation Importance — model-agnostic feature importance ranking
ROC & Precision-Recall Curves — threshold-independent performance evaluation
Stratified K-Fold Cross-Validation — stability and generalisation assessment


📜 License
This project is open-source and available under the MIT License.

🙏 Acknowledgements

UCI ML Repository for the dataset
scikit-learn for ML utilities
Streamlit for the interactive app framework
Dr. William H. Wolberg for the original dataset collection
