# =========================================
# Iris Flower Classification using ML
# =========================================

def main():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

    # -----------------------------------------
    # 1. Load Dataset
    # -----------------------------------------
    iris = load_iris()
    X = iris.data
    y = iris.target

    df = pd.DataFrame(X, columns=iris.feature_names)
    df['species'] = y

    # -----------------------------------------
    # 2. Data Visualization (Pair Plot)
    # -----------------------------------------
    custom_palette = ["#56B4E9", "#E69F00", "#009E73"]

    sns.pairplot(
        df,
        hue="species",
        palette=custom_palette,
        plot_kws={'s': 60, 'alpha': 0.9}
    )
    plt.suptitle("Iris Dataset Feature Distribution", y=1.02)
    plt.show()

    # -----------------------------------------
    # 3. Train-Test Split
    # -----------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )

    # -----------------------------------------
    # 4. Model Training and Comparison
    # -----------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    print("Model Accuracy Comparison:\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.2f}")

    # -----------------------------------------
    # 5. Best Model: Logistic Regression
    # -----------------------------------------
    best_model = LogisticRegression(max_iter=200)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix - Logistic Regression")
    plt.show()

    # -----------------------------------------
    # 6. Cross-Validation
    # -----------------------------------------
    cv_scores = cross_val_score(best_model, X, y, cv=5)
    print("\nCross-Validation Accuracy Scores:", cv_scores)
    print("Mean Cross-Validation Accuracy:", cv_scores.mean())

    # -----------------------------------------
    # 7. Prediction on New Sample
    # -----------------------------------------
    sample = [[5.1, 3.5, 1.4, 0.2]]
    prediction = best_model.predict(sample)

    print("\nPredicted Species:",
          iris.target_names[prediction][0])


# -----------------------------------------
# Run main
# -----------------------------------------
if __name__ == "__main__":
    main()

