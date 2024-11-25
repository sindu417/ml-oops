import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
# Load the dataset
data = load_iris()
X = data.data
y = data.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Set experiment name and parameters
experiment_name = "Iris_RandomForest_Experiment"
mlflow.set_experiment(experiment_name)
# Define hyperparameters for tracking
n_estimators_options = [50, 100, 150]
max_depth_options = [3, 5, 7]
random_state = 42

# Run experiments with different hyperparameters
for n_estimators in n_estimators_options:
    for max_depth in max_depth_options:
        with mlflow.start_run(run_name=f"RF_Model_n{n_estimators}_d{max_depth}") as run:
            # Log hyperparameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("random_state", random_state)
            # Initialize and train the model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
            model.fit(X_train, y_train)
            # Make predictions and calculate accuracy
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average="macro")
            recall = recall_score(y_test, predictions, average="macro")
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            # Log feature importances as an artifact
            np.savetxt("feature_importances.csv", model.feature_importances_, delimiter=",")
            mlflow.log_artifact("feature_importances.csv")
            # Log and register the model in the MLflow Model Registry
            mlflow.sklearn.log_model(model, artifact_path="model")
            model_name = "Iris_RandomForest_Classifier"
            model_version = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
            print(f"Model registered as {model_name}, version {model_version.version}")
            print(f"Run ID: {run.info.run_id} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")