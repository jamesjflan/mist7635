#%% read in libs 
#general
import pandas as pd 

#sci-kit learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay

# finding multicollinearity
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#plotting
import matplotlib.pyplot as plt
import seaborn as sns

#custom functions for data cleaning
from utils import CleanData

#%% load data & pre-process using customer CleanData class
df = pd.read_excel(r'./data/full_data_set.xlsx')
cleaner = CleanData()
df = cleaner.convert_draft_pick(df)
df = cleaner.fill_na(df, 0)

#%%
# Define the feature columns (all columns you've listed except 'NFL Draft Pick')
feature_cols = ['Height (inches) (247)', 'Weight (247)', 'GP - Senior (Rush)', 'Car - Senior (Rush)', 'Yds - Senior (Rush)', 'Avg - Senior (Rush)', 'Y/G - Senior (Rush)', 'Lng - Senior (Rush)', '100+ - Senior (Rush)', 'TD - Senior (Rush)', 'GP - Junior (Rush)', 'Car - Junior (Rush)', 'Yds - Junior (Rush)', 'Avg - Junior (Rush)', 'Y/G - Junior (Rush)', 'Lng - Junior (Rush)', '100+ - Junior (Rush)', 'TD - Junior (Rush)', 'GP - Sophomore (Rush)', 'Car - Sophomore (Rush)', 'Yds - Sophomore (Rush)', 'Avg - Sophomore (Rush)', 'Y/G - Sophomore (Rush)', 'Lng - Sophomore (Rush)', '100+ - Sophomore (Rush)', 'TD - Sophomore (Rush)', 'GP - Freshman (Rush)', 'Car - Freshman (Rush)', 'Yds - Freshman (Rush)', 'Avg - Freshman (Rush)', 'Y/G - Freshman (Rush)', 'Lng - Freshman (Rush)', '100+ - Freshman (Rush)', 'TD - Freshman (Rush)', 'GP - Senior (Rec)', 'Rec - Senior', 'Yds - Senior (Rec)', 'Avg - Senior (Rec)', 'Y/G - Senior (Rec)', 'Lng - Senior (Rec)', 'TD - Senior (Rec)', 'GP - Junior (Rec)', 'Rec - Junior', 'Yds - Junior (Rec)', 'Avg - Junior (Rec)', 'Y/G - Junior (Rec)', 'Lng - Junior (Rec)', 'TD - Junior (Rec)', 'GP - Sophomore (Rec)', 'Rec - Sophomore', 'Yds - Sophomore (Rec)', 'Avg - Sophomore (Rec)', 'Y/G - Sophomore (Rec)', 'Lng - Sophomore (Rec)', 'TD - Sophomore (Rec)', 'GP - Freshman (Rec)', 'Rec - Freshman', 'Yds - Freshman (Rec)', 'Avg - Freshman (Rec)', 'Y/G - Freshman (Rec)', 'Lng - Freshman (Rec)', 'TD - Freshman (Rec)']

# Split the dataset into features (X) and target variable (y)
X = df[feature_cols]
y = df['NFL Draft Pick']


#%%
corr_cols = ['NFL Draft Pick', 'Height (inches) (247)', 'Weight (247)', 'GP - Senior (Rush)', 'Car - Senior (Rush)', 'Yds - Senior (Rush)', 'Avg - Senior (Rush)', 'Y/G - Senior (Rush)', 'Lng - Senior (Rush)', '100+ - Senior (Rush)', 'TD - Senior (Rush)', 'GP - Junior (Rush)', 'Car - Junior (Rush)', 'Yds - Junior (Rush)', 'Avg - Junior (Rush)', 'Y/G - Junior (Rush)', 'Lng - Junior (Rush)', '100+ - Junior (Rush)', 'TD - Junior (Rush)', 'GP - Sophomore (Rush)', 'Car - Sophomore (Rush)', 'Yds - Sophomore (Rush)', 'Avg - Sophomore (Rush)', 'Y/G - Sophomore (Rush)', 'Lng - Sophomore (Rush)', '100+ - Sophomore (Rush)', 'TD - Sophomore (Rush)', 'GP - Freshman (Rush)', 'Car - Freshman (Rush)', 'Yds - Freshman (Rush)', 'Avg - Freshman (Rush)', 'Y/G - Freshman (Rush)', 'Lng - Freshman (Rush)', '100+ - Freshman (Rush)', 'TD - Freshman (Rush)', 'GP - Senior (Rec)', 'Rec - Senior', 'Yds - Senior (Rec)', 'Avg - Senior (Rec)', 'Y/G - Senior (Rec)', 'Lng - Senior (Rec)', 'TD - Senior (Rec)', 'GP - Junior (Rec)', 'Rec - Junior', 'Yds - Junior (Rec)', 'Avg - Junior (Rec)', 'Y/G - Junior (Rec)', 'Lng - Junior (Rec)', 'TD - Junior (Rec)', 'GP - Sophomore (Rec)', 'Rec - Sophomore', 'Yds - Sophomore (Rec)', 'Avg - Sophomore (Rec)', 'Y/G - Sophomore (Rec)', 'Lng - Sophomore (Rec)', 'TD - Sophomore (Rec)', 'GP - Freshman (Rec)', 'Rec - Freshman', 'Yds - Freshman (Rec)', 'Avg - Freshman (Rec)', 'Y/G - Freshman (Rec)', 'Lng - Freshman (Rec)', 'TD - Freshman (Rec)']
correlation_matrix = df[corr_cols].corr()

# Extract correlations with 'NFL Draft Pick' column
draft_correlations = correlation_matrix['NFL Draft Pick'].sort_values(ascending=False)[:10]

# Create a bar plot for visualization
plt.figure(figsize=(10, 8))
sns.barplot(x=draft_correlations.index, y=draft_correlations.values)
plt.xticks(rotation=90)  # Rotate the x labels so they don't overlap
plt.title('Correlation Coefficients with NFL Draft Pick')
plt.show()

features = df.columns.difference(['NFL Draft Pick', 'Name', 'Hometown', 'State', 'High School'])

# Set up the matplotlib figure - adjust the figure size as necessary
plt.figure(figsize=(20, 20))

# Loop through each feature to create a scatter plot
for i, feature in enumerate(features, 1):
    plt.subplot(5, 5, i)  # Adjust the grid dimensions (5x5) based on the number of features
    plt.scatter(df[feature], df['NFL Draft Pick'], alpha=0.5)
    plt.title(f"{feature} vs NFL Draft Pick")
    plt.xlabel(feature)
    plt.ylabel('NFL Draft Pick')

plt.tight_layout()
plt.show()

# %%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

class CleanData:
    """Class to clean data."""
    
    @staticmethod
    def convert_draft_pick(data):
        data['NFL Draft Pick'] = data['NFL Draft Pick'].map({'Yes': 1, 'No': 0})
        return data
    
    @staticmethod
    def fill_na(data, value=0):
        data.fillna(value, inplace=True)
        return data

class DataPreprocessor:
    """Class to preprocess data using scaling and PCA."""

    def __init__(self, n_components=None):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components) if n_components else None

    def fit_transform(self, X):
        X_scaled = self.scaler.fit_transform(X)
        if self.pca:
            return self.pca.fit_transform(X_scaled)
        return X_scaled

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        if self.pca:
            return self.pca.transform(X_scaled)
        return X_scaled

class ModelTrainer:
    """Class to train logistic regression model."""
    
    def __init__(self, use_pca=False, pca_components=None):
        self.use_pca = use_pca
        self.model = LogisticRegression(max_iter=1000)
        self.preprocessor = DataPreprocessor(n_components=pca_components if use_pca else None)

    def train(self, X, y):
        X_processed = self.preprocessor.fit_transform(X)
        self.model.fit(X_processed, y)
        return self

    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed), self.model.predict_proba(X_processed)[:, 1]

class PerformanceEvaluator:
    """Class to evaluate model performance."""
    
    def __init__(self, y_true, y_pred, y_scores):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_scores = y_scores

    def confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def mcr(self):
        return np.mean(self.y_pred != self.y_true)

    def roc_auc(self):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_scores)
        return auc(fpr, tpr)

# Load data
df = pd.read_excel(r'./data/full_data_set.xlsx')
df = CleanData.convert_draft_pick(df)
df = CleanData.fill_na(df)

# Feature selection
target = 'NFL Draft Pick'
features = [col for col in df.columns if col not in [target, 'Name', 'Hometown', 'State', 'High School']]
X = df[features]
y = df[target]

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
model_no_pca = ModelTrainer(use_pca=False)
model_no_pca.train(X_train, y_train)

model_pca = ModelTrainer(use_pca=True, pca_components=5)
model_pca.train(X_train, y_train)

# Predictions
y_pred_no_pca, y_scores_no_pca = model_no_pca.predict(X_test)
y_pred_pca, y_scores_pca = model_pca.predict(X_test)

# Evaluate models
eval_no_pca = PerformanceEvaluator(y_test, y_pred_no_pca, y_scores_no_pca)
eval_pca = PerformanceEvaluator(y_test, y_pred_pca, y_scores_pca)

# Output results
print("Results without PCA:")
print("Confusion Matrix:\n", eval_no_pca.confusion_matrix())
print("MCR:", eval_no_pca.mcr())
print("ROC AUC:", eval_no_pca.roc_auc())

print("\nResults with PCA:")
print("Confusion Matrix:\n", eval_pca.confusion_matrix())
print("MCR:", eval_pca.mcr())
print("ROC AUC:", eval_pca.roc_auc())

# %%


#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
grid_search.fit(X, y)

# Best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Predictions with the best found parameters
y_pred = grid_search.predict(X)
y_scores = grid_search.decision_function(X)  # Scores for ROC curve

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Classification Report
print(classification_report(y, y_pred))

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y, y_scores)
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
mcr = np.mean(y_pred != y)
print(f"Misclassification Rate (MCR): {mcr:.2f}")
# %%
