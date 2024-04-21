#%%
#%%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# custom libs
from utils import CleanData

# Show relationship between features and target

class Preprocessor:
    def __init__(self, n_components=None):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
    
    # Do PCA and scale features
    def fit_transform(self, X):
        X_scaled = self.scaler.fit_transform(X)
        if self.pca.n_components is not None:
            X_pca = self.pca.fit_transform(X_scaled)
            return X_pca
        return X_scaled
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        if self.pca.n_components is not None:
            X_pca = self.pca.transform(X_scaled)
            return X_pca
        return X_scaled

class ModelTrainer:
    def __init__(self, input_shape, num_classes):
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    def train(self, X_train, y_train, X_val, y_val):
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[self.early_stopping],
            verbose=1
        )

    def plot_training_validation_loss(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

class ResultsAnalyzer:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = np.argmax(model.predict(X_test), axis=1)
        self.y_pred_proba = model.predict(X_test)  # Probabilities for all classes

    def MCR(self):
        mcr = np.mean(self.y_pred != self.y_test)
        return mcr

    def ConfusionMatrix(self):
        return confusion_matrix(self.y_test, self.y_pred)

    def plotROC(self):
        # Assuming binary classification; adapt as necessary for multi-class
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba[:, 1])
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


#%%
data = pd.read_excel(r'./data/full_data_set.xlsx')  # Load your data
df = CleanData.convert_draft_pick(data)
df = CleanData.fill_na(df, 0)
target = "NFL Draft Pick"
#%%
cols = ['NFL Draft Pick', 'Name', 'Hometown', 'State', 'High School']
#%%
X = df.drop(cols, axis=1).values
y = df[target].values

#%%
# Preprocessing
preprocessor = Preprocessor(n_components=20)  # Use PCA to reduce dims
X_processed = preprocessor.fit_transform(X)

#%%
# splitting full data 
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

# Separate training and validation sets from X_train for train cycle
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=24)

#%%
# Model Configuration
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#%%
# Model training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

#%%
# Plotting training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%
# Results Analysis
predictions = model.predict(X_test).ravel()
y_pred = (predictions > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)

#%%
# Print results
print("Confusion Matrix:\n", conf_matrix)
print("ROC AUC:", roc_auc)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
# %%
