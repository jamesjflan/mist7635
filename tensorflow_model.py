#%%

# custom libs
from utils import CleanData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Function to create a TensorFlow model
def create_model(learning_rate, dropout_rate, num_units):
    model = Sequential([
        Dense(num_units, activation='relu', input_shape=(input_dim,)),
        Dropout(dropout_rate),
        Dense(num_units // 2, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


class ResultsAnalyzer:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)
        self.y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

    def MCR(self):
        mcr = np.mean(self.y_pred != self.y_test)
        return mcr

    def ConfusionMatrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        return cm

    def PrecisionRecallF1(self):
        report = classification_report(self.y_test, self.y_pred, target_names=['Undrafted', 'Drafted'], output_dict=True)
        print("Classification Report:")
        print(classification_report(self.y_test, self.y_pred, target_names=['Undrafted', 'Drafted']))
        return report

    def plotROC(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        return roc_auc

#%% Load and preprocess data
# Replace 'path_to_your_data.csv' and column names according to your dataset specifics
df = pd.read_excel(r'./data/cleaned_data_77.xlsx')
df = data.fillna(0)  # Simplified cleaning for example purposes
target = "NFL Draft Pick"
cols = ['NFL Draft Pick', 'Name', 'Hometown', 'State', 'High School']
cleaner = CleanData()
#y = cleaner.convert_draft_pick(df)
X = df.drop(cols, axis=1).values
y = df[target].values
#%%

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
input_dim = X_train.shape[1]

# Wrap the model with KerasClassifier
model = KerasClassifier(model=create_model, verbose=0, learning_rate=0.01, dropout_rate=0.2, num_units=128)

# Define the parameter grid
param_grid = {
    'learning_rate': [0.001, 0.01],
    'dropout_rate': [0.1, 0.2],
    'num_units': [64, 128],
    'epochs': [50, 100],
    'batch_size': [32, 64]
}

# Setup GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1', verbose=1)
grid_result = grid.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters:", grid_result.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_result.best_score_))

# Predictions with the best found parameters
y_pred = grid_result.predict(X_test)
y_scores = grid_result.predict_proba(X_test)[:, 1]  # get probabilities for the positive class
#%%
best_model = grid_result.best_estimator_


analyzer = ResultsAnalyzer(best_model, X_test, y_test)
print("Misclassification Rate (MCR):", analyzer.MCR())
print("Confusion Matrix:\n", analyzer.ConfusionMatrix())
analyzer.PrecisionRecallF1()
print("AUC:", analyzer.plotROC())

#%%
# %%
import shap 
best_estimator = grid_result.best_estimator_
keras_model = best_estimator.model_
#%%
#%%
# Now, use SHAP's DeepExplainer (or another appropriate explainer) on the Keras model
explainer = shap.KernelExplainer(keras_model, X_train[:5])  # Using a subset of X_train for efficiency
shap_values = explainer.shap_values(X_train[:5])
#%%
shap_values[:5]
#%%
# Since you have used the KerasClassifier, X_test will likely not have a 'columns' attribute.
# Ensure you have the feature names from your original DataFrame.
feature_names = df.drop(cols, axis=1).columns.tolist()
feature_names[:5]
#%%
# Plot the SHAP summary
shap.summary_plot(shap_values, X_train[:5], feature_names=feature_names)
#%%
# Visualize the first prediction's explanation
# Here we take the first sample's SHAP values and the first sample from X_train
shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_train[0], feature_names=feature_names)

# %%
background = shap.sample(X_train, 5)  # For efficiency, use a sample of the train data
explainer = shap.KernelExplainer(keras_model.predict, background)

# Calculate SHAP values - this may take some time depending on the complexity of the model and data
shap_values = explainer.shap_values(X_test[:5])

# Ensure that the feature names are correctly formatted as a list
feature_names = df.drop(cols, axis=1).columns.tolist()

# Plot the SHAP summary
# Note: If using a classification model with multiple classes, you may need to adjust this call
shap.summary_plot(shap_values, X_test[:5], feature_names=feature_names)
# %%
