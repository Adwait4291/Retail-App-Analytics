# delete_model.py

import mlflow
from mlflow.exceptions import MlflowException

def delete_registered_model(model_name):
    client = mlflow.tracking.MlflowClient()

    try:
        # Delete all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")
        for mv in versions:
            print(f"Deleting version: {mv.version}")
            client.delete_model_version(name=mv.name, version=mv.version)

        # Delete the registered model
        client.delete_registered_model(name=model_name)
        print(f"\n✅ Successfully deleted model: {model_name}")

    except MlflowException as e:
        print(f"\n❌ Error deleting model: {e}")

if __name__ == "__main__":
    # Replace this with your model name in the MLflow registry
    model_name = "Retail24RandomForestClassifier"
    delete_registered_model(model_name)
