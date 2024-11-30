import torch
import numpy as np
import pandas as pd
from climax.arch import ClimaX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_accuracy_metrics(predictions, targets):
    """
    Compute accuracy metrics between predictions and ground truth.

    Args:
        predictions (np.ndarray): Predicted values.
        targets (np.ndarray): Ground truth values.

    Returns:
        dict: A dictionary containing accuracy metrics.
    """
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }

def load_model(checkpoint_path, config):
    from src.climax.arch import ClimaX  # Ensure the correct import
    config['default_vars'] = ['geopotential_500', 'temperature_850',
                              '2m_temperature']  # Update with the appropriate variables
    model = ClimaX(**config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Resize parameters as necessary for compatibility
    if "var_embed" in state_dict:
        state_dict["var_embed"] = state_dict["var_embed"].resize_as_(model.var_embed)
    if "head.4.weight" in state_dict:
        state_dict["head.4.weight"] = state_dict["head.4.weight"].resize_as_(model.head[4].weight)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def ensemble_predict(models, x, y, lead_times, variables, out_variables, metric=None, lat=None, method="average"):
    predictions = []

    for model in models:
        _, preds = model(x, y, lead_times, variables, out_variables, metric, lat)
        predictions.append(preds)

    if method == "average":
        ensemble_preds = torch.mean(torch.stack(predictions), dim=0)
    elif method == "median":
        ensemble_preds = torch.median(torch.stack(predictions), dim=0).values
    else:
        raise ValueError("Unknown ensemble method. Use 'average' or 'median'.")

    return ensemble_preds


if __name__ == "__main__":
    # Paths to the model checkpoints
    model_paths = ["/mnt/c/Users/Owner/climax/logs/checkpoints/epoch_000-v8.ckpt",
                   "/mnt/c/Users/Owner/climax/logs/checkpoints/epoch_000-v7.ckpt",
                   "/mnt/c/Users/Owner/climax/logs/checkpoints/epoch_003.ckpt"]

    # Example model configuration
    config = {
        "img_size": [32, 64],
        "patch_size": 2,
        "embed_dim": 1024,
        "depth": 8,
        "decoder_depth": 2,
        "num_heads": 16,
        "mlp_ratio": 4,
        "drop_path": 0.1,
        "drop_rate": 0.1
    }

    # Load models
    models = [load_model(path, config) for path in model_paths]

    # Example input data
    x = torch.randn(1, 3, 32, 64)  # Adjust shape as per the model requirements
    y = torch.randn(1, 3, 32, 64)  # Dummy target data
    lead_times = torch.FloatTensor([1, 2, 3])
    variables = ["geopotential_500", "temperature_850", "2m_temperature"]

    out_variables = ["geopotential_500", "temperature_850", "2m_temperature"]
    lat = torch.FloatTensor([10, 20, 30])

    # Perform ensemble prediction
    predictions = ensemble_predict(models, x, y, lead_times, variables, out_variables, metric=None, lat=lat,
                                   method="average")

    # Convert tensor to NumPy array for better readability
    predictions_np = predictions.detach().cpu().numpy()

    # Save predictions to CSV files
    for i, pred in enumerate(predictions_np):
        df = pd.DataFrame(pred[0], columns=[f"Column_{j}" for j in range(pred.shape[2])])
        df.to_csv(f"prediction_output_{i}.csv", index=False)

    print("Predictions have been saved to CSV files.")
