Credit Risk Assessment Project


This project focuses on evaluating and predicting credit risk using both Artificial Neural Networks (ANN) and Classical Machine Learning (ML) models. 
It leverages various machine learning techniques to classify borrowers into categories (e.g., high risk, low risk) based on features extracted from their respective datasets.

Project Structure:
Data Files:
1. Persistent Files: (credit_risk_ann_model.h5 && credit_risk_model.pkl)
      Contain processed datasets, pre-trained model weights, or serialized objects for reuse.
2. CSV Files: Include raw and preprocessed datasets used for training, validation, and testing purposes.
3. Notebooks: (CreditRiskAssesmentSystem_Classical_ML.ipynb && CreditRiskAssesmentSystem_ANN_DL.ipynb)

-> ANN.ipynb: Implements credit risk prediction using an Artificial Neural Network model, detailing data preprocessing, model architecture, training, and evaluation.
-> Classical_ML.ipynb: Explores traditional machine learning algorithms like Logistic Regression, Random Forests, or Gradient Boosting for the same task.

Key Features:
-Data Processing: 
  Handled missing values, outliers, and feature engineering for optimal input quality.
  Data normalization and encoding for compatibility with machine learning algorithms.
-Model Training:
  ANN: Designed a neural network architecture, fine-tuned hyperparameters, and used appropriate activation functions to maximize performance.
  Classical ML: Experimented with various models, optimized hyperparameters using GridSearchCV or RandomizedSearchCV.
-Performance Metrics:
  Evaluated models based on metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC to ensure reliable credit risk predictions.
-File Organization:
  Persistent and CSV files ensure seamless data reuse and experiment reproducibility.
  Separate notebooks for ANN and classical ML methods provide clear implementation and comparison.
-Technologies Used:
  Programming Language: Python
  Libraries and Tools: Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn
  Database/Storage: CSVs for raw data, pickled files for persistence
