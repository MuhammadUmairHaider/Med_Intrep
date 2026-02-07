# Med-Gemma Cancer Type Detection

This project uses `google/medgemma-1.5-4b-it` to classify cancer types from medical reports (`tcga_reports_valid.csv`).

## Setup

1.  **Install Dependencies**:
    The project uses `uv` for dependency management.
    ```bash
    uv venv .venv
    source .venv/bin/activate
    uv pip install torch transformers accelerate bitsandbytes pandas scikit-learn huggingface_hub peft trl
    ```

2.  **Authenticate with Hugging Face**:
    You need a token with access to Med-Gemma.
    ```bash
    huggingface-cli login --token <YOUR_TOKEN>
    ```

## Usage

### 1. Train the Model
To fine-tune the model using LoRA:
```bash
python src/train.py
```
This will save the trained adapter to `./final_medgemma_model`.

### 2. Evaluate
To run evaluation on the test set:
```bash
python src/evaluate.py
```

### 3. Test Loading
To quickly verify the model loads correctly:
```bash
python src/test_loading.py
```

## implementation Details
-   **Model**: `google/medgemma-1.5-4b-it` (4 billion parameters).
-   **Quantization**: 4-bit loading with `bitsandbytes` to reduce VRAM usage.
-   **Fine-tuning**: LoRA (Low-Rank Adaptation) via `peft` library.
-   **Dataset**: TCGA reports with 20 cancer classes.
