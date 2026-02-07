import glob
import os


def get_latest_checkpoint(
    base_model_id: str, checkpoint_dir: str = "checkpoints"
) -> str:
    """
    Returns the path to the latest checkpoint in the checkpoint_dir.
    If no checkpoints are found, returns the base_model_id.
    """
    if not os.path.exists(checkpoint_dir):
        print(
            f"Checkpoint directory '{checkpoint_dir}' does not exist. Using base model: {base_model_id}"
        )
        return base_model_id

    # Check for subdirectories (assumes checkpoints are saved as dirs)
    checkpoints = [
        d for d in glob.glob(os.path.join(checkpoint_dir, "*")) if os.path.isdir(d)
    ]

    if not checkpoints:
        print(
            f"No checkpoints found in '{checkpoint_dir}'. Using base model: {base_model_id}"
        )
        return base_model_id

    # Sort by modification time (latest first)
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint
