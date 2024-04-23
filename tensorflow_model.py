#%%
#%%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# custom libs
from utils import CleanData
#%%
class Preprocessor:
    def __init__(self, n_components=None):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
    
    def fit_transform(self, X):
        X_scaled = self.scaler.fit_transform(X)
        if self.pca.n_components is not None:
            return self.pca.fit_transform(X_scaled)
        return X_scaled
    
def create_model(input_shape, num_classes=1, layers=[128, 64], dropout_rate=0.5, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(dropout_rate))
    for layer_size in layers[1:]:
        model.add(Dense(layer_size, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy', metrics=['accuracy'])
    return model

class ModelTrainer:
    def __init__(self, input_shape, num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def train(self, X, y):
        # Define a model creation function that accepts parameters correctly
        model_func = lambda: create_model(input_shape=self.input_shape, num_classes=self.num_classes)
        model_wrapper = KerasClassifier(model=model_func, epochs=100, batch_size=32, verbose=1)
        
        param_grid = {
            'model__dropout_rate': [0.5, 0.6],
            'model__learning_rate': [0.001, 0.01],
            'batch_size': [32, 64],
            'epochs': [50, 100]
        }
        
        grid = GridSearchCV(estimator=model_wrapper, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_result = grid.fit(X, y)
        print(f"Best parameters: {grid_result.best_params_}")
        print(f"Best cross-validation score: {grid_result.best_score_}")
    
    def plot_training_validation_loss(self, history):
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        return roc_auc

#%%
data = pd.read_excel(r'./data/full_data_set.xlsx')
df = data.fillna(0)  # Simplified cleaning for example purposes
target = "NFL Draft Pick"
cols = ['NFL Draft Pick', 'Name', 'Hometown', 'State', 'High School']
X = df.drop(cols, axis=1).values
y = df[target].values
#%%
# Preprocessing
preprocessor = Preprocessor(n_components=62)
X_processed = preprocessor.fit_transform(X)
#%%
# Model Configuration and Training
trainer = ModelTrainer(input_shape=X_processed.shape[1], num_classes=1)  
trainer.train(X_processed, y)
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
