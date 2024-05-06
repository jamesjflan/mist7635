#%%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback  
import matplotlib.pyplot as plt

# custom libs 
from utils import CleanData

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
    def __init__(self, params_grid, cv=5, scoring='f1'):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
        ])
        self.params_grid = params_grid
        self.cv = cv
        self.scoring = scoring
        self.grid_search = None

    def train(self, X_train, y_train):
        self.grid_search = GridSearchCV(self.pipeline, self.params_grid, cv=self.cv, scoring=self.scoring, verbose=1)
        self.grid_search.fit(X_train, y_train)
        print("Best Parameters Found: ", self.grid_search.best_params_)

    def plot_training_validation_loss(self):
        if self.grid_search and 'validation_0' in self.grid_search.best_estimator_.named_steps['xgb'].evals_result():
            evals_result = self.grid_search.best_estimator_.named_steps['xgb'].evals_result()
            epochs = range(1, len(evals_result['validation_0']['logloss']) + 1)
            plt.plot(epochs, evals_result['validation_0']['logloss'], label='Train')
            plt.plot(epochs, evals_result['validation_1']['logloss'], label='Test')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
        else:
            print("No evaluation results to plot.")
    
    def best_params(self):
        if self.grid_search:
            return self.grid_search.best_params_
        return None

    def best_model(self):
        if self.grid_search:
            return self.grid_search.best_estimator_
        return None

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
        self.y_pred = self.model.predict(X_test)
        self.y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    
    
    def MCR(self):
        mcr = np.mean(self.y_pred != self.y_test)
        return mcr
    
    def ConfusionMatrix(self):
        return confusion_matrix(self.y_test, self.y_pred)

    def PrecisionRecallF1(self):
        report = classification_report(self.y_test, self.y_pred, target_names=['Class 0', 'Class 1'], output_dict=True)
        print("Classification Report:")
        print(classification_report(self.y_test, self.y_pred, target_names=['Class 0', 'Class 1']))
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

#%%
df = pd.read_excel(r'./data/cleaned_data_77.xlsx')  
cleaner = CleanData()
# df = cleaner.convert_draft_pick(data)
df = cleaner.fill_na(df, 0)
target = "NFL Draft Pick"
len(df.columns)
cols = ['NFL Draft Pick', 'Name', 'Hometown', 'State', 'High School']
df['NFL Draft Pick'].value_counts() 

#%%
# capture col names for Shapley values
football_players = df[["Name", "NFL Draft Pick"]]
feature_names = df.drop(cols, axis=1).columns.tolist()
#%%
X = df.drop(cols, axis=1).values
y = df[target].values

# splitting full data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# Model training
params_grid = {
    'pca__n_components': [None, 5, 10, 20],  # Include None to test without PCA
    'xgb__max_depth': [3, 4, 5],
    'xgb__n_estimators': [50, 100, 200],
    'xgb__learning_rate': [0.01, 0.1, 0.2]
}

trainer = ModelTrainer(params_grid)
trainer.train(X_train, y_train)

#%%
trainer.best_params()
best_model = trainer.best_model()
#%%
# Analyzing results
analyzer = ResultsAnalyzer(best_model, X_test, y_test)
print("MCR:", analyzer.MCR())
print("Confusion Matrix:\n", analyzer.ConfusionMatrix())
analyzer.PrecisionRecallF1()
print("ROC AUC:", analyzer.plotROC())


#%%
import shap 
tree_model = best_model.named_steps['xgb']

# Create a SHAP TreeExplainer for the XGBoost model
explainer = shap.TreeExplainer(tree_model)
shap_values = explainer.shap_values(X_test[2:3])
shap.summary_plot(shap_values, X_test[2:3], feature_names = feature_names, plot_type="bar")

#%%
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[2], X_test[0], feature_names=feature_names)

# %%
football_players[2:3]
# %%
