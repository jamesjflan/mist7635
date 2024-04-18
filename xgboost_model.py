#%%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback  
import matplotlib.pyplot as plt

# custom libs 
from utils import CleanData


class Preprocessor:
    def __init__(self, n_components=None):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
    
    # do PCA
    def fit_transform(self, X):
        X_scaled = self.scaler.fit_transform(X)
        if self.pca.n_components is not None:
            X_pca = self.pca.fit_transform(X_scaled)
            return X_pca
        return X_scaled
    
    # scale features 
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        if self.pca.n_components is not None:
            X_pca = self.pca.transform(X_scaled)
            return X_pca
        return X_scaled

class EvalCallback(TrainingCallback):
    """Custom callback to record evaluation results per iteration."""
    def __init__(self, evals_result):
        super().__init__()
        self.evals_result = evals_result

    def after_iteration(self, model, epoch, evals_log):
        """Run after each iteration. Return True if training should stop."""
        for data, metric in evals_log.items():
            for metric_name, log in metric.items():
                metric_full_name = f"{data}-{metric_name}"
                if metric_full_name not in self.evals_result:
                    self.evals_result[metric_full_name] = []
                self.evals_result[metric_full_name].extend(log)
        return False

class ModelTrainer:
    def __init__(self, params_grid, cv=5, scoring='accuracy'):
        self.model = XGBClassifier(use_label_encoder=False)
        self.params_grid = params_grid
        self.cv = cv
        self.scoring = scoring
        self.grid_search = None
        self.evals_result = {}

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train, y_train, 
            eval_set=[(X_train, y_train), (X_val, y_val)], 
            eval_metric="logloss", 
            verbose=True, 
            early_stopping_rounds=10,
            callbacks=[EvalCallback(self.evals_result)]
        )

    def plot_training_validation_loss(self):
        for metric, result in self.evals_result.items():
            epochs = range(1, len(result) + 1)
            plt.plot(epochs, result, label=metric)
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def perform_grid_search(self, X_train, y_train):
        self.grid_search = GridSearchCV(self.model, self.params_grid, cv=self.cv, scoring=self.scoring)
        self.grid_search.fit(X_train, y_train)

    def best_params(self):
        return self.grid_search.best_params_
    
    def best_model(self):
        return self.grid_search.best_estimator_
    
    def validation_summary(self):
        if self.grid_search is not None:
            cv_results = self.grid_search.cv_results_
            for mean, std, params in zip(cv_results['mean_test_score'], cv_results['std_test_score'], cv_results['params']):
                print(f"Mean: {mean:.3f}, Std: {std:.3f} with: {params}")
        else:
            print("No grid search has been performed yet.")

class ResultsAnalyzer:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    
    
    def MCR(self):
        mcr = np.mean(self.y_pred != self.y_test)
        return mcr
    
    def ConfusionMatrix(self):
        return confusion_matrix(self.y_test, self.y_pred)
    
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
    
#%%
# Example of how to use these classes
data = pd.read_excel(r'./data/full_data_set.xlsx')  # Load your data
df = CleanData.convert_draft_pick(data)
df = CleanData.fill_na(df, 0)
target = "NFL Draft Pick"
len(df.columns)
#%%
df.columns
cols = ['NFL Draft Pick', 'Name', 'Hometown', 'State', 'High School']
#%%
X = data.drop(cols, axis=1).values
y = data[target].values

#%%
# Preprocessing
preprocessor = Preprocessor(n_components=20)  # Use PCA to reduce to 10 dimensions
X_processed = preprocessor.fit_transform(X)

# Splitting the data
# splitting full data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Separate training and validation sets from X_train for monitoring
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=24)

# Model training
params_grid = {
    'max_depth': [3, 4, 5,7],
    'n_estimators': [100, 150, 500],
    'learning_rate': [0.01, 0.1, 0.2]
}
trainer = ModelTrainer(params_grid)
trainer.train(X_train, y_train, X_val, y_val)
trainer.validation_summary()
trainer.plot_training_validation_loss()

#%%
# Analyzing results
analyzer = ResultsAnalyzer(trainer.best_model(), X_test, y_test)
print("MCR:", analyzer.MCR())
print("Confusion Matrix:\n", analyzer.ConfusionMatrix())
print("ROC AUC:", analyzer.plotROC())

# %%
