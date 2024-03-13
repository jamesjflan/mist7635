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
cleaner = CleanData(df)
df = cleaner.convert_draft_pick()
df = cleaner.fill_na(0)


# Define the feature columns (all columns you've listed except 'NFL Draft Pick')
feature_cols = ['Height (inches) (247)', 'Weight (247)', 'GP - Senior (Rush)', 'Car - Senior (Rush)', 'Yds - Senior (Rush)', 'Avg - Senior (Rush)', 'Y/G - Senior (Rush)', 'Lng - Senior (Rush)', '100+ - Senior (Rush)', 'TD - Senior (Rush)', 'GP - Junior (Rush)', 'Car - Junior (Rush)', 'Yds - Junior (Rush)', 'Avg - Junior (Rush)', 'Y/G - Junior (Rush)', 'Lng - Junior (Rush)', '100+ - Junior (Rush)', 'TD - Junior (Rush)', 'GP - Sophomore (Rush)', 'Car - Sophomore (Rush)', 'Yds - Sophomore (Rush)', 'Avg - Sophomore (Rush)', 'Y/G - Sophomore (Rush)', 'Lng - Sophomore (Rush)', '100+ - Sophomore (Rush)', 'TD - Sophomore (Rush)', 'GP - Freshman (Rush)', 'Car - Freshman (Rush)', 'Yds - Freshman (Rush)', 'Avg - Freshman (Rush)', 'Y/G - Freshman (Rush)', 'Lng - Freshman (Rush)', '100+ - Freshman (Rush)', 'TD - Freshman (Rush)', 'GP - Senior (Rec)', 'Rec - Senior', 'Yds - Senior (Rec)', 'Avg - Senior (Rec)', 'Y/G - Senior (Rec)', 'Lng - Senior (Rec)', 'TD - Senior (Rec)', 'GP - Junior (Rec)', 'Rec - Junior', 'Yds - Junior (Rec)', 'Avg - Junior (Rec)', 'Y/G - Junior (Rec)', 'Lng - Junior (Rec)', 'TD - Junior (Rec)', 'GP - Sophomore (Rec)', 'Rec - Sophomore', 'Yds - Sophomore (Rec)', 'Avg - Sophomore (Rec)', 'Y/G - Sophomore (Rec)', 'Lng - Sophomore (Rec)', 'TD - Sophomore (Rec)', 'GP - Freshman (Rec)', 'Rec - Freshman', 'Yds - Freshman (Rec)', 'Avg - Freshman (Rec)', 'Y/G - Freshman (Rec)', 'Lng - Freshman (Rec)', 'TD - Freshman (Rec)']

# Split the dataset into features (X) and target variable (y)
X = df[feature_cols]
y = df['NFL Draft Pick']

#%% ------- check for multicollinearity + select features ------------------

#%%

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# Ensure X is your features DataFrame without the constant
X_vif = df[feature_cols]  
#%%

# Add constant for intercept (as needed for statsmodels)
X_vif = sm.add_constant(X_vif)

# Initialize DataFrame to hold VIF for each feature (excluding the constant)
vif_data = pd.DataFrame()
vif_data['Feature'] = [column for column in X.columns]  # Exclude the constant here

# Calculate VIF for each feature (excluding the constant)
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i+1) for i in range(len(X.columns))]  # i+1 to skip the constant

# vif_data.sort_values(by='VIF', ascending=False)

#%%
# Filtering out features with VIF > 10
high_vif_features = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()

# remove the constant
high_vif_features.remove('const') if 'const' in high_vif_features else None

# Dropping high VIF features from X
X_filtered = X.drop(columns=high_vif_features)
X_filtered.fillna(0, inplace=True)  # Fill NaNs with 0
print(f"There are {len(X_filtered.columns)} features with VIF < 10.")

X_significant = X_filtered
#%%  -------------- first pass to select features with low multicollinearity and ID potential features
X_train, X_test, y_train, y_test = train_test_split(X_significant, y, test_size=0.3, random_state=42)

# Standardize the features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#%%
# Add a constant to the feature set for the intercept
X_train_scaled_sm = sm.add_constant(X_train_scaled)
X_test_scaled_sm = sm.add_constant(X_test_scaled)

# Fit the logistic regression model using statsmodels
logit_model = sm.Logit(y_train, X_train_scaled_sm)
result = logit_model.fit()

#%%
p_values = result.pvalues
significant_features = p_values[p_values < 0.05].index.tolist()

# Removing 'const' from the list if it's there
if 'const' in significant_features:
    significant_features.remove('const')

# Adjusting significant_features to match original feature names
# This is necessary because the scaled and constant-added DataFrame doesn't directly correspond to original feature names
significant_features_names = [X_train.columns[i - 1] for i in range(1, len(significant_features))]  # Adjusting index for 'const'

# Rebuilding the DataFrame with only significant features
X_significant = X_filtered[significant_features_names]

print("Significant features based on p-values:", significant_features_names)

#reset X_significant back to X for the next steps
X = X_significant

#%% ---------------- Run RIDGE and LASSO models on only significant features

# Changing the X alters results below (can do full X, X with VIF > 10, X VIF> 10 AND pvalue < 0.05)
#%%
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a pipeline that includes scaling and the model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(solver='liblinear'))
])

# Define the grid of parameters to search over
param_grid = [
    {
        'model__penalty': ['l2'],
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__solver': ['lbfgs', 'sag'],  # Including 'lbfgs' as an alternative
        'model__max_iter': [200, 300, 400],  # Increased max_iter values
        'model__tol': [1e-4, 1e-3],  # Adjusting tolerance
        'model__class_weight': [None, 'balanced']
    },
    {
        'model__penalty': ['l1'],
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__solver': ['saga'],  # 'saga' supports L1 penalty
        'model__max_iter': [200, 300, 400],  # Similarly increased
        'model__tol': [1e-4, 1e-3],
        'model__class_weight': [None, 'balanced']
    }
]

# Initialize the grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# setup best model for predictions
best_model = grid_search.best_estimator_

# Make predictions 
y_pred = best_model.predict(X_test)

#%%
# Calculate overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Generate a classification report
report = classification_report(y_test, y_pred, target_names=['Undrafted', 'Drafted'])  # Adjust class names as necessary
print(report)

# Predicted probabilities for confusion matrix and ROC curve
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_matrix, title='Best Model Confusion Matrix')

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Best Model AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for the Best Model')
plt.legend(loc='lower right')
plt.show()

#%% show the best features
import numpy as np
best_logreg_model = grid_search.best_estimator_.named_steps['model']
feature_names = X.columns  # Adjust if your feature matrix X isn't a DataFrame

# Extract coefficients
coefficients = best_logreg_model.coef_[0]

# Match coefficients to feature names
features_coefficients = zip(feature_names, coefficients)

# Sort features by absolute value of coefficient
sorted_features_coefficients = sorted(features_coefficients, key=lambda x: np.abs(x[1]), reverse=True)[:10]

# Separate feature names and their corresponding coefficients
sorted_features, sorted_coefficients = zip(*sorted_features_coefficients)

# Visualize feature importance
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_features)), sorted_coefficients, align='center')
plt.yticks(range(len(sorted_features)), sorted_features)
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.xlabel('Coefficient Magnitude')
plt.ylabel('Feature')
plt.title('Feature Importance in NFL Prediction')
plt.show()

# %%
