# Airport Passenger Satisfaction Analysis

This repository contains data and code to predict passenger satisfaction for an airline based on various service metrics.

## Project Structure

- `data/`: Contains the `train.csv` and `test.csv` datasets. (Ignored in git to save space, download from Kaggle)
- `src/`: Contains the main Python script `train.py` for model training and evaluation.
- `airport_satisfaction_analysis.ipynb`: Original Jupyter Notebook with the analysis.
- `decision_tree_viz.pdf`: Generated visualization of the Decision Tree model (depth 3).
- `feature_importances.png`: Visual representation of which features impact passenger satisfaction the most.
- `confusion_matrices.png`: Heatmaps showing the accuracy of both models on the test set.

## Environment Setup

This project uses `uv` for fast dependency management.

1. Ensure `uv` is installed on your system.
2. The dependencies are defined in `pyproject.toml`. To run the script, simply execute:
   ```bash
   uv run python src/train.py
   ```

## Models Used

The script trains two distinct models:
- **Decision Tree Classifier:** Achieved roughly ~94.4% accuracy. The most critical features were found to be *Online boarding*, *Inflight wifi service*, and *Type of Travel*.
- **Gaussian Naive Bayes:** Achieved ~80.4% accuracy.

By comparing both models, the Decision Tree is the recommended model for predicting passenger satisfaction with this dataset.

## Visualizations

Here are some key visualizations generated from the analysis:

### Feature Importances (Decision Tree)
![Feature Importances](feature_importances.png)

### Confusion Matrices
![Confusion Matrices](confusion_matrices.png)
