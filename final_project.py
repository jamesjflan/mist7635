#%% read in libs 
import pandas as pd 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

#%% load data
df = pd.read_excel(r'./data/full_data_set.xlsx')
df.head()
df["NFL Draft Pick"].value_counts()
#%%

# Prepare the data
# Convert 'NFL Draft Pick' to a binary variable where 'Yes' = 1 and 'No' = 0
df['NFL Draft Pick'] = df['NFL Draft Pick'].map({'Yes': 1, 'No': 0})
df['NFL Draft Pick'].value_counts()
#%%
# Define the feature columns (all columns you've listed except 'NFL Draft Pick')
feature_cols = ['Height (inches) (247)', 'Weight (247)', 'GP - Senior (Rush)', 'Car - Senior (Rush)', 'Yds - Senior (Rush)', 'Avg - Senior (Rush)', 'Y/G - Senior (Rush)', 'Lng - Senior (Rush)', '100+ - Senior (Rush)', 'TD - Senior (Rush)', 'GP - Junior (Rush)', 'Car - Junior (Rush)', 'Yds - Junior (Rush)', 'Avg - Junior (Rush)', 'Y/G - Junior (Rush)', 'Lng - Junior (Rush)', '100+ - Junior (Rush)', 'TD - Junior (Rush)', 'GP - Sophomore (Rush)', 'Car - Sophomore (Rush)', 'Yds - Sophomore (Rush)', 'Avg - Sophomore (Rush)', 'Y/G - Sophomore (Rush)', 'Lng - Sophomore (Rush)', '100+ - Sophomore (Rush)', 'TD - Sophomore (Rush)', 'GP - Freshman (Rush)', 'Car - Freshman (Rush)', 'Yds - Freshman (Rush)', 'Avg - Freshman (Rush)', 'Y/G - Freshman (Rush)', 'Lng - Freshman (Rush)', '100+ - Freshman (Rush)', 'TD - Freshman (Rush)', 'GP - Senior (Rec)', 'Rec - Senior', 'Yds - Senior (Rec)', 'Avg - Senior (Rec)', 'Y/G - Senior (Rec)', 'Lng - Senior (Rec)', 'TD - Senior (Rec)', 'GP - Junior (Rec)', 'Rec - Junior', 'Yds - Junior (Rec)', 'Avg - Junior (Rec)', 'Y/G - Junior (Rec)', 'Lng - Junior (Rec)', 'TD - Junior (Rec)', 'GP - Sophomore (Rec)', 'Rec - Sophomore', 'Yds - Sophomore (Rec)', 'Avg - Sophomore (Rec)', 'Y/G - Sophomore (Rec)', 'Lng - Sophomore (Rec)', 'TD - Sophomore (Rec)', 'GP - Freshman (Rec)', 'Rec - Freshman', 'Yds - Freshman (Rec)', 'Avg - Freshman (Rec)', 'Y/G - Freshman (Rec)', 'Lng - Freshman (Rec)', 'TD - Freshman (Rec)']

# Split the dataset into features (X) and target variable (y)
X = df[feature_cols]
y = df['NFL Draft Pick']

X.fillna(0, inplace=True)
#%%
y
#%%
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model with Ridge regularization (L2)
ridge_model = LogisticRegression(penalty='l2', solver='liblinear')
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)

# Evaluate the Ridge model
print("Ridge Model Accuracy:", accuracy_score(y_test, y_pred_ridge))
print(classification_report(y_test, y_pred_ridge))

# Train a logistic regression model with LASSO regularization (L1)
lasso_model = LogisticRegression(penalty='l1', solver='liblinear')
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)

# Evaluate the LASSO model
print("LASSO Model Accuracy:", accuracy_score(y_test, y_pred_lasso))
print(classification_report(y_test, y_pred_lasso))


# %%
