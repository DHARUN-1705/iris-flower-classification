# Iris Flower Classification using Machine Learning

## 📌 Project Overview
This project focuses on classifying iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — based on sepal and petal measurements using supervised machine learning techniques.

The objective of this project is to understand and implement the complete machine learning workflow, including data exploration, model comparison, evaluation, and prediction.

---

## 📊 Dataset
- **Dataset Name:** Iris Dataset
- **Source:** scikit-learn built-in dataset
- **Number of Samples:** 150
- **Number of Features:** 4
- **Target Classes:** Setosa, Versicolor, Virginica

The dataset is clean, balanced, and does not contain missing values.

---

## ⚙️ Machine Learning Models Used
The following classification models were implemented and compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier

Logistic Regression was selected as the final model due to its simplicity, interpretability, and consistent performance.

---

## 🔍 Methodology
1. Load the Iris dataset
2. Perform exploratory data analysis using pair plots
3. Split the dataset into training and testing sets
4. Train multiple classification models
5. Compare models using accuracy
6. Evaluate the best model using a confusion matrix
7. Validate performance using cross-validation
8. Predict the species for a new sample input

---

## 📈 Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Cross-Validation Accuracy

---

## 📊 Model Performance
- **Logistic Regression Accuracy:** 100%
- **KNN Accuracy:** 100%
- **Decision Tree Accuracy:** 100%
- **Mean Cross-Validation Accuracy:** 97.33%

High accuracy is achieved due to the clear separability of classes in the Iris dataset.

---

## 📸 Output Visualizations

### Pair Plot of Iris Features
![Pair Plot](images/pairplot.png)

### Confusion Matrix (Logistic Regression)
![Confusion Matrix](images/confusion_matrix.png)

---

## 🧠 Conclusion
This project demonstrates that simple supervised learning models can achieve excellent performance on well-structured datasets. It highlights the importance of data visualization, model comparison, and validation techniques in machine learning.

---

## 🛠️ Tools & Technologies
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

---

## ▶️ How to Run the Project
```bash
python iris_classification.py
