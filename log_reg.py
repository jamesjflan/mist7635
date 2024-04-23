#%% read in libs 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

# Custom functions for data cleaning
from utils import CleanData

# Load data & pre-process using custom CleanData class
df = pd.read_excel('./data/full_data_set.xlsx')
cleaner = CleanData()
df = cleaner.convert_draft_pick(df)
df = cleaner.fill_na(df, 0)

# Define the feature columns (assuming these are correct and complete)
feature_cols = [col for col in df.columns if col not in ['NFL Draft Pick', 'Name', 'Hometown', 'State', 'High School']]

# Split the dataset into features (X) and target variable (y)
X = df[feature_cols]
y = df['NFL Draft Pick']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80-20 split

# Creating a Pipeline for scaling and logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),  # Initialize PCA but number of components will be defined in the grid search
    ('logreg', LogisticRegression(solver='liblinear'))
])

# Parameters grid including PCA components
param_grid = {
    'pca__n_components': [None, 5, 10, 15, 20],  # None means no dimensionality reduction
    'logreg__C': np.logspace(-4, 4, 10),
    'logreg__penalty': ['l1', 'l2']
}

# Setting up GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Predictions with the best found parameters
y_pred = grid_search.predict(X_test)
y_scores = grid_search.decision_function(X_test)  # Scores for ROC curve

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#%%
mcr = np.mean(y_pred != y_test)
print(f"Misclassification Rate (MCR): {mcr:.2f}")
# %%
