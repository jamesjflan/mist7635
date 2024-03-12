#%% read in libs 
#general
import pandas as pd 

#sci-kit learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc

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

#%%
# Define the feature columns (all columns you've listed except 'NFL Draft Pick')
feature_cols = ['Height (inches) (247)', 'Weight (247)', 'GP - Senior (Rush)', 'Car - Senior (Rush)', 'Yds - Senior (Rush)', 'Avg - Senior (Rush)', 'Y/G - Senior (Rush)', 'Lng - Senior (Rush)', '100+ - Senior (Rush)', 'TD - Senior (Rush)', 'GP - Junior (Rush)', 'Car - Junior (Rush)', 'Yds - Junior (Rush)', 'Avg - Junior (Rush)', 'Y/G - Junior (Rush)', 'Lng - Junior (Rush)', '100+ - Junior (Rush)', 'TD - Junior (Rush)', 'GP - Sophomore (Rush)', 'Car - Sophomore (Rush)', 'Yds - Sophomore (Rush)', 'Avg - Sophomore (Rush)', 'Y/G - Sophomore (Rush)', 'Lng - Sophomore (Rush)', '100+ - Sophomore (Rush)', 'TD - Sophomore (Rush)', 'GP - Freshman (Rush)', 'Car - Freshman (Rush)', 'Yds - Freshman (Rush)', 'Avg - Freshman (Rush)', 'Y/G - Freshman (Rush)', 'Lng - Freshman (Rush)', '100+ - Freshman (Rush)', 'TD - Freshman (Rush)', 'GP - Senior (Rec)', 'Rec - Senior', 'Yds - Senior (Rec)', 'Avg - Senior (Rec)', 'Y/G - Senior (Rec)', 'Lng - Senior (Rec)', 'TD - Senior (Rec)', 'GP - Junior (Rec)', 'Rec - Junior', 'Yds - Junior (Rec)', 'Avg - Junior (Rec)', 'Y/G - Junior (Rec)', 'Lng - Junior (Rec)', 'TD - Junior (Rec)', 'GP - Sophomore (Rec)', 'Rec - Sophomore', 'Yds - Sophomore (Rec)', 'Avg - Sophomore (Rec)', 'Y/G - Sophomore (Rec)', 'Lng - Sophomore (Rec)', 'TD - Sophomore (Rec)', 'GP - Freshman (Rec)', 'Rec - Freshman', 'Yds - Freshman (Rec)', 'Avg - Freshman (Rec)', 'Y/G - Freshman (Rec)', 'Lng - Freshman (Rec)', 'TD - Freshman (Rec)']

#%% ------- check for multicollinearity + select features ------------------
# Split the dataset into features (X) and target variable (y)
X = df[feature_cols]
y = df['NFL Draft Pick']

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
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.3, random_state=42)

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


#%% ---------------- Run RIDGE and LASSO models on only significant features

# Changing the X alters results below (can do full X, X with VIF > 10, X VIF> 10 AND pvalue < 0.05)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_significant, y, test_size=0.3, random_state=42)

# Standardize the features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model with Ridge
ridge_model = LogisticRegression(penalty='l2', solver='liblinear')
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)

# Evaluate the Ridge model
print("Ridge Model Accuracy:", accuracy_score(y_test, y_pred_ridge))
print(classification_report(y_test, y_pred_ridge))

# Train a logistic regression model with LASSO 
lasso_model = LogisticRegression(penalty='l1', solver='liblinear')
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)

# Evaluate the LASSO model
print("LASSO Model Accuracy:", accuracy_score(y_test, y_pred_lasso))
print(classification_report(y_test, y_pred_lasso))

#%% Understand which features are important in Ridge v LASSO 
# Extract feature names and coefficients for both models
ridge_coefficients = ridge_model.coef_[0]
lasso_coefficients = lasso_model.coef_[0]
feature_names = X_significant.columns

# Create DataFrames for easy viewing
ridge_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': ridge_coefficients})
lasso_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': lasso_coefficients})

# Sort the coefficients by their absolute values for better interpretation
ridge_df = ridge_df.reindex(ridge_df.Coefficient.abs().sort_values(ascending=False).index)
lasso_df = lasso_df.reindex(lasso_df.Coefficient.abs().sort_values(ascending=False).index)

# Print the sorted DataFrames
print("Ridge Model Coefficients:")
print(ridge_df, "\n")
print("LASSO Model Coefficients (non-zero):")
print(lasso_df[lasso_df['Coefficient'] != 0])

#%%
# Confusion Matrices
conf_matrix_ridge = confusion_matrix(y_test, y_pred_ridge)
conf_matrix_lasso = confusion_matrix(y_test, y_pred_lasso)
def plot_confusion_matrix(cm, title='Confusion Matrix', labels=['Not Drafted', 'Drafted']):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)

# Plotting Ridge Model Confusion Matrix
plot_confusion_matrix(conf_matrix_ridge, title='Ridge Model Confusion Matrix')

# Plotting Lasso Model Confusion Matrix
plot_confusion_matrix(conf_matrix_lasso, title='Lasso Model Confusion Matrix')

plt.show()

#%%
# ROC Curve and AUC for Ridge Model
fpr_ridge, tpr_ridge, thresholds_ridge = roc_curve(y_test, ridge_model.predict_proba(X_test_scaled)[:, 1])
auc_ridge = auc(fpr_ridge, tpr_ridge)

# ROC Curve and AUC for Lasso Model
fpr_lasso, tpr_lasso, thresholds_lasso = roc_curve(y_test, lasso_model.predict_proba(X_test_scaled)[:, 1])
auc_lasso = auc(fpr_lasso, tpr_lasso)

# Plotting ROC Curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_ridge, tpr_ridge, label=f'Ridge AUC = {auc_ridge:.2f}')
plt.plot(fpr_lasso, tpr_lasso, label=f'Lasso AUC = {auc_lasso:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal for random chance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Ridge and Lasso Models')
plt.legend(loc='lower right')
plt.show()
# %%
