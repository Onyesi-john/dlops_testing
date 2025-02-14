import mlflow
import subprocess

# Check last model loss
experiment = mlflow.get_experiment_by_name("dlops_experiment")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
latest_run = runs.iloc[-1]
loss = latest_run["metrics.loss"]

if loss > 0.5:  # If loss is too high, retrain
    print("Performance dropped! Retraining model...")
    subprocess.run(["python", "model/train.py"])
