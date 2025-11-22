Audit a Loan Approval Model for Bias

This project demonstrates how to audit, measure, and mitigate bias in a loan approval machine learning model using Fairlearn, Scikit-learn, and Python.

It highlights how sensitive attributes—such as gender—can influence model predictions, and how fairness constraints can help reduce unwanted disparities.

📌 Project Overview

Financial institutions increasingly rely on ML models for loan approval decisions. However, these models may unintentionally exhibit bias against certain demographic groups.

This notebook performs a full workflow of bias detection and mitigation, including:

Data preparation

Model training (Logistic Regression)

Fairness metrics evaluation

Visualization of disparities

Fairness mitigation using ThresholdOptimizer

Re-evaluation after correction

The project uses a synthetic loan dataset containing demographic and financial attributes.

🧰 Tech Stack

Python 3

Pandas, NumPy

Scikit-learn

Fairlearn

Matplotlib, Seaborn

📂 Dataset Description

The synthetic dataset consists of 1000 rows and the following columns:

Feature	Description
Gender	Sensitive attribute (Male/Female → encoded as 0/1)
Age	Applicant age
Income	Annual income
Loan_Amount	Loan requested
Credit_Score	Credit score (300–850)
Approved	Loan approval label (0 = Reject, 1 = Approve)

A random distribution is used to simulate a typical loan approval dataset.

⚙️ Workflow Summary
1️⃣ Data Preprocessing

Label Encode gender

Normalize continuous variables using StandardScaler

Split into train–test sets

2️⃣ Model Training

Train a Logistic Regression classifier

Evaluate performance using accuracy & classification report

3️⃣ Fairness Evaluation

Using Fairlearn MetricFrame, the notebook measures:

Selection Rate per gender

Demographic Parity Difference (DPD)

Measures disparity in approval decisions

Example finding:

DPD ≈ 0.18, indicating moderate bias.

4️⃣ Visualization

A bar plot shows selection rate differences between:

Gender = 0 (Male)

Gender = 1 (Female)

5️⃣ Bias Mitigation

Using Fairlearn’s ThresholdOptimizer with:

Constraint: "demographic_parity"

Sensitive feature: gender

After applying mitigation:

DPD drops to ≈ 0.05, showing improved fairness.

📊 Key Results
Metric	Before Mitigation	After Mitigation
Demographic Parity Difference	~0.187	~0.053
Selection Rate Gap	Significant	Reduced

The fairness intervention successfully lowers disparity—though potential trade-offs in performance may occur.

📌 How to Run

Install dependencies:

pip install fairlearn scikit-learn pandas matplotlib seaborn


Open the notebook:

jupyter notebook Audit_a_loan_approval_model_for_bias.ipynb


Run all cells.

📘 Learning Outcomes

By completing this notebook, you will understand how to:

✔ Detect bias in machine learning models
✔ Use fairness metrics (DPD, selection rate)
✔ Visualize model disparities
✔ Apply post-processing techniques to improve fairness
✔ Balance accuracy vs fairness trade-offs

🔍 Future Improvements

Use real loan datasets (e.g., Home Mortgage Disclosure Act)

Compare multiple fairness mitigation strategies:

Reweighting

Preprocessing transformations

Adversarial debiasing

Extend fairness evaluation to:

Equal Opportunity

Predictive Parity

Evaluate intersectional bias (e.g., Gender × Age)

📄 License

This project is open-source and free to use for educational and research purposes.
