# Iris Flower Classification using Machine Learning

## 📌 Project Overview
This project focuses on classifying iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — based on their sepal and petal measurements using supervised machine learning techniques.

The goal of this project is to understand and implement the complete machine learning workflow, including data exploration, model training, evaluation, and prediction.

---

## 📊 Dataset
- **Dataset Name:** Iris Dataset
- **Source:** scikit-learn built-in dataset
- **Number of Samples:** 150
- **Features:**
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- **Target Variable:** Iris flower species

The dataset is clean and does not contain any missing values.

---

## ⚙️ Machine Learning Models Used
The following supervised learning models were implemented and compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier

---

## 🔍 Methodology
The project follows these steps:

1. Load the Iris dataset
2. Perform exploratory data analysis using pair plots
3. Split the dataset into training and testing sets
4. Train multiple classification models
5. Compare model performance using accuracy
6. Select the best-performing model
7. Evaluate the model using a confusion matrix
8. Validate performance using cross-validation
9. Predict the species for a new sample input

---

## 📈 Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Cross-Validation Accuracy

Cross-validation helps ensure that the model generalizes well to unseen data.

---

## ✅ Results
All models achieved high accuracy due to the clear separability of the Iris dataset.  
Logistic Regression was selected as the final model because of its consistent performance and simplicity.

---

## 🧠 Conclusion
This project demonstrates that simple machine learning algorithms can achieve excellent performance on well-structured datasets. It highlights the importance of exploratory data analysis, model comparison, and proper evaluation techniques in machine learning.

---

## 🛠️ Tools & Technologies
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

---

## ▶️ How to Run the Project
1. Clone or download this repository
2. Install the required Python libraries
3. Run the Python script:
   ```bash
   python iris_classification.py
