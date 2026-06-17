# 🏦 Bank Transaction Fraud Detection

A machine learning project for detecting fraudulent banking transactions using a Random Forest classifier trained on 1,000,000 synthetic transactions across 10 countries.

---

## 📊 Dataset

- **Source:** [Kaggle — Bank Transaction Fraud Detection](https://www.kaggle.com/datasets/marusagar/bank-transaction-fraud-detection)
- **Size:** 1,000,000 transactions × 26 features
- **Class distribution:** 94.47% legitimate / 5.53% fraud (imbalanced)
- **Coverage:** 10 countries, multiple merchant categories, payment methods, and device types

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| ML Library | scikit-learn |
| Data Processing | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Environment | Google Colab |
| Model Persistence | joblib |

---

## 🔄 Project Pipeline

```
Data Loading → EDA → Preprocessing → Train/Test Split → Model Training → Evaluation → Feature Importance → Save Model
```

**1. EDA** — Analyzed class distribution, missing values, and data types across 26 features  
**2. Preprocessing** — Dropped ID/timestamp columns, label-encoded 5 categorical features, resulting in 20 model-ready features  
**3. Train/Test Split** — 80/20 stratified split to preserve fraud ratio in both sets  
**4. Model Training** — Random Forest with `class_weight='balanced'` to handle class imbalance; trained in 2.6 minutes on 800,000 rows  
**5. Evaluation** — Assessed using Recall, Precision, F1-Score, and ROC-AUC  
**6. Feature Importance** — Identified key behavioral signals driving fraud predictions  

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 62.24% |
| ROC-AUC | 0.7179 |
| Fraud Recall | 69% |
| Fraud Precision | 10% |

### Confusion Matrix (200,000 test transactions)

| | Predicted: Not Fraud | Predicted: Fraud |
|---|---|---|
| **Actual: Not Fraud** | 116,873 ✅ | 72,076 ❌ False Alarm |
| **Actual: Fraud** | 3,451 ❌ Missed | 7,600 ✅ Caught |

> **Note on metrics:** In fraud detection, **Recall is prioritized over Accuracy**. Missing a fraud transaction is far more costly than a false alarm. The model successfully catches 69% of all fraud cases in the test set.

---

## 🔍 Key Findings — Feature Importance

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 🥇 1 | `failed_attempts` | 34.3% | Number of failed PIN/password attempts before transaction — strongest signal of account takeover |
| 🥈 2 | `is_international` | 15.9% | Cross-border transactions carry significantly higher fraud risk |
| 🥉 3 | `is_night_transaction` | 14.7% | Fraudulent transactions disproportionately occur at night when account owners are unaware |
| 4 | `hour_of_day` | 8.6% | Specific transaction hours correlate with fraud patterns |
| 5 | `merchant_category` | 7.6% | Certain merchant types are more frequently targeted |
| 6 | `time_since_last_txn_hrs` | 4.9% | Unusual gaps or rapid sequences between transactions signal anomalies |
| 7 | `pin_changed_recently` | 3.6% | Recent PIN changes followed immediately by transactions are a red flag |
| 8 | `transaction_amount` | 1.8% | Surprisingly low — fraud is not primarily about large amounts |

> **Counter-intuitive insight:** `transaction_amount` ranks only #8 (1.8% importance). Behavioral signals — how the transaction happens — are far more predictive than how much money is involved. This is consistent with real-world fraud detection literature.

---

## ⚙️ Model Configuration

```python
RandomForestClassifier(
    n_estimators=100,         # 100 decision trees
    max_depth=10,             # prevent overfitting
    class_weight='balanced',  # handle class imbalance
    random_state=42,
    n_jobs=-1                 # use all CPU cores
)
```

---

## 🚀 Next Steps

- [ ] Apply SMOTE for minority class oversampling
- [ ] Compare with XGBoost and LightGBM
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Optimize decision threshold to maximize fraud Recall
- [ ] Wrap model into a REST API using FastAPI
- [ ] Build a real-time monitoring dashboard with Streamlit

---

## 📁 Repository Structure

```
bank-fraud-detection/
│
├── fraud_detection_random_forest.ipynb   # Main notebook
├── README.md                             # Project documentation
└── model_features.json                   # Feature list for inference
```

---

## 👤 Author

**Fil Ardhi Kamza**  
Jakarta, Indonesia  
[github.com/techyfeels](https://github.com/techyfeels) | filardhian@gmail.com
