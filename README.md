# 🏦 Universal Bank — Customer Predictive Analytics Dashboard

A comprehensive **4-layer analytics dashboard** built with Streamlit for Universal Bank's personal loan campaign optimization and cross-selling strategy.

## 📊 Analytics Layers

| Layer | Question Answered | Key Visualizations |
|-------|------------------|--------------------|
| **Descriptive** | *What happened?* | KPI cards, distributions, product adoption rates |
| **Diagnostic** | *Why did it happen?* | Correlation heatmap, bivariate scatter, cross-tabs, association rules |
| **Predictive** | *What will happen?* | Gradient Boosting + SMOTE, ROC/PR curves, SHAP analysis, What-If predictor |
| **Prescriptive** | *What should we do?* | Customer segmentation, target profiles, campaign recommendations |

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🛠️ Tech Stack

- **Frontend:** Streamlit, Plotly
- **ML Model:** Gradient Boosting Classifier (scikit-learn)
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **Class Balancing:** SMOTE (imbalanced-learn)
- **Association Mining:** Apriori algorithm (mlxtend)

## 📁 Project Structure

```
├── app.py                  # Main dashboard (single file)
├── UniversalBank.csv       # Dataset (5,000 customers)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Dark theme configuration
└── README.md
```

## ⚡ Quick Start

### Local Setup
```bash
git clone https://github.com/your-username/universal-bank-analytics.git
cd universal-bank-analytics
pip install -r requirements.txt
streamlit run app.py
```

### Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path: `app.py`
5. Deploy!

## 📈 Dataset

**Universal Bank** dataset with 5,000 customer records and 14 variables covering demographics, financial behaviour, and product holdings. Target variable: **Personal Loan** acceptance (binary, 9.6% positive class).

## 🔍 Key Findings

- **Income** is the strongest predictor of loan acceptance (correlation: 0.50)
- **CD Account** holders are ~6× more likely to accept a personal loan
- The Gradient Boosting model achieves **>95% accuracy** with SMOTE-balanced training
- SHAP analysis confirms Income, CD Account, and CCAvg as the top 3 drivers

## 👤 Author

**Nikhil** — SP Jain School of Global Management (MBA Global)

---
*Built as part of Applied Analytics coursework*
