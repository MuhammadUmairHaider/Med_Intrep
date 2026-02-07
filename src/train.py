import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import SFTConfig, SFTTrainer

from dataset import load_data

# Model from Hugging Face hub
model_name = "google/medgemma-1.5-4b-it"


# Function to format the dataset for SFTTrainer
def format_instruction(sample):
    return f"""### Instruction:
Analyze the medical report and extract the following:
- Cancer Type
- ICD-O-3 Site
- ICD-O-3 Histology

### Input:
{sample['text']}

### Response:
Cancer Type: {sample['cancer_type']}
ICD-O-3 Site: {sample['icd_o_3_site']}
ICD-O-3 Histology: {sample['icd_o_3_histology']}"""


def main():
    print("Loading data...")
    # Load raw data first to split
    train_df, val_df, test_df = load_data("tcga_reports_valid.csv")

    # Convert to HuggingFace Datasets for TRL/Trainer
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    # test_dataset = Dataset.from_pandas(test_df) # Unused in training loop

    # Mixed Precision Config
    # bnb_config removed for full precision/bf16 on 3090

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    # Training Arguments using SFTConfig
    training_args = SFTConfig(
        output_dir="./checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,  # Enable BF16 for 3090
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb",
        max_length=512,  # Correct argument name
        dataset_text_field="text",
        packing=False,
    )

    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,  # Correct argument name
        args=training_args,
        formatting_func=format_instruction,
    )

    trainer.train()

    print("Saving model...")
    trainer.model.save_pretrained("./final_medgemma_model")

    # Evaluation (Basic generation test)
    print("Running evaluation on a few test samples...")
    pipe = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, max_length=200
    )

    for i in range(5):
        sample = test_df.iloc[i]
        prompt = (
            f"### Instruction:\nAnalyze the following medical report "
            f"and classify the cancer type.\n\n### Input:\n{sample['text']}"
            f"\n\n### Response:\n"
        )
        result = pipe(f"<s>{prompt}")
        print(f"True Label: {sample['cancer_type']}")
        print(
            f"Predicted: {result[0]['generated_text'].replace(f'<s>{prompt}', '').strip()[:100]}..."
        )
        print("-" * 50)


if __name__ == "__main__":
    main()
