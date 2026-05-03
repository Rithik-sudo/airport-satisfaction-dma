import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import numpy as np
import os

def load_data():
    """Loads train and test datasets from the data/ directory."""
    if not os.path.exists('data/train.csv') or not os.path.exists('data/test.csv'):
        raise FileNotFoundError("Data files not found. Ensure 'data/train.csv' and 'data/test.csv' exist.")
        
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    return train_data, test_data

def preprocess_data(train_data, test_data):
    """Combines, cleans, and encodes the train and test data."""
    # Combine train and test data for preprocessing
    data = pd.concat([train_data, test_data], ignore_index=True)

    # Preprocessing
    # Encoding categorical variables including target variable
    label_encoders = {}
    categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
    for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))  # Convert to string to ensure correct encoding
        label_encoders[column] = le

    # Handle missing numeric values
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # Split the data back into train and test sets
    train_data_clean = data.iloc[:len(train_data)]
    test_data_clean = data.iloc[len(train_data):]

    # Separate features and target
    X_train = train_data_clean.drop('satisfaction', axis=1)
    y_train = train_data_clean['satisfaction']
    X_test = test_data_clean.drop('satisfaction', axis=1)
    y_test = test_data_clean['satisfaction']
    
    return X_train, X_test, y_train, y_test, label_encoders

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Trains Decision Tree and Naive Bayes classifiers and evaluates them."""
    # Decision Tree Model
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    dt_predictions = dt_classifier.predict(X_test)

    # Naive Bayes Model
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    nb_predictions = nb_classifier.predict(X_test)

    # Evaluation
    print("--------------------------------------------------")
    print("Decision Tree Classifier Report:")
    print(classification_report(y_test, dt_predictions))
    dt_acc = accuracy_score(y_test, dt_predictions)
    print("Decision Tree Accuracy:", dt_acc)

    print("--------------------------------------------------")
    print("Naive Bayes Classifier Report:")
    print(classification_report(y_test, nb_predictions))
    nb_acc = accuracy_score(y_test, nb_predictions)
    print("Naive Bayes Accuracy:", nb_acc)
    print("--------------------------------------------------")
    
    return dt_classifier, nb_classifier, dt_predictions, nb_predictions, dt_acc, nb_acc

def plot_feature_importances(dt_classifier, feature_names):
    """Plots feature importances for the Decision Tree model."""
    feature_importances = pd.DataFrame(dt_classifier.feature_importances_,
                                       index=feature_names,
                                       columns=['importance']).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar', legend=False)
    plt.title('Feature Importance from Decision Tree')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    print("Saved feature importances plot to feature_importances.png")

def plot_confusion_matrices(y_test, dt_predictions, nb_predictions):
    """Plots confusion matrices for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(confusion_matrix(y_test, dt_predictions), annot=True, fmt="d", cmap='Blues', 
                xticklabels=['Dissatisfied', 'Satisfied'], yticklabels=['Dissatisfied', 'Satisfied'], ax=axes[0])
    axes[0].set_title('Decision Tree Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    sns.heatmap(confusion_matrix(y_test, nb_predictions), annot=True, fmt="d", cmap='Oranges', 
                xticklabels=['Dissatisfied', 'Satisfied'], yticklabels=['Dissatisfied', 'Satisfied'], ax=axes[1])
    axes[1].set_title('Naive Bayes Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    print("Saved confusion matrices plot to confusion_matrices.png")

def main():
    print("Loading data...")
    train_data, test_data = load_data()
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, label_encoders = preprocess_data(train_data, test_data)
    
    print("Training models...")
    dt_classifier, nb_classifier, dt_predictions, nb_predictions, dt_acc, nb_acc = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    print("Plotting feature importances...")
    plot_feature_importances(dt_classifier, X_train.columns)
    
    print("Plotting confusion matrices...")
    plot_confusion_matrices(y_test, dt_predictions, nb_predictions)
    
    print("Generating Decision Tree Graph (Depth 3)...")
    clf_small = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf_small.fit(X_train, y_train)
    
    dot_data = export_graphviz(clf_small, out_file=None,
                               feature_names=X_train.columns,
                               class_names=['Dissatisfied', 'Satisfied'],
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree_viz")
    print("Saved decision tree visualization to decision_tree_viz.pdf")
    print("Run completed successfully!")

if __name__ == "__main__":
    main()
