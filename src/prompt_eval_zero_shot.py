import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import load_data

model_id = "google/medgemma-1.5-4b-it"


def run_prompt_eval():
    print(f"Loading base model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Load test data
    _, _, test_df = load_data("tcga_reports_valid.csv")

    # Select a larger sample for better statistics
    samples = test_df.sample(20, random_state=42)

    prompts_templates = {
        "cancer_type": {
            "prompt": (
                "### Instruction:\nClassify the cancer type (e.g., BRCA, GBM, OV, KIRC) from the report. "
                "Output ONLY the code.\n\n### Input:\n{text}\n\n### Response:\n"
            ),
            "target_col": "cancer_type",
        },
        "icd_site": {
            "prompt": (
                "### Instruction:\nIdentify the primary tumor site ICD-O-3 code (e.g., C50.9, C61.9) from the report. "
                "Output ONLY the code.\n\n### Input:\n{text}\n\n### Response:\n"
            ),
            "target_col": "icd_o_3_site",
        },
        "icd_histology": {
            "prompt": (
                "### Instruction:\nIdentify the diagnosis ICD-O-3 histology code (e.g., 8500/3, 8140/3) "
                "from the report. Output ONLY the code.\n\n### Input:\n{text}\n\n### Response:\n"
            ),
            "target_col": "icd_o_3_histology",
        },
    }

    print("\nStarting Detailed Prompt Evaluation...\n")

    results = {k: {"correct": 0, "total": 0} for k in prompts_templates.keys()}

    for idx, row in samples.iterrows():
        print(f"\nProcessing ID: {row['patient_id']}")

        for task_name, config in prompts_templates.items():
            template = config["prompt"]
            target_val = str(row[config["target_col"]]).strip()

            prompt = template.format(text=row["text"])
            input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **input_ids,
                    max_new_tokens=20,  # Short generation for codes
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Naive parsing: Clean up prompt and take the first token/line
            response = (
                output_text.replace(prompt.replace("### Response:\n", "").strip(), "")
                .split("### Response:")[-1]
                .strip()
            )
            # Further cleanup to get just the code if possible
            response_clean = response.split("\n")[0].strip().replace("**", "")

            is_correct = target_val in response_clean
            results[task_name]["total"] += 1
            if is_correct:
                results[task_name]["correct"] += 1

            print(f"  Task: {task_name}")
            print(f"    Target: {target_val}")
            print(f"    Pred:   {response_clean}")
            print(f"    Match:  {is_correct}")

    print("\nSummary Results:")
    for task, metrics in results.items():
        acc = metrics["correct"] / metrics["total"] * 100 if metrics["total"] > 0 else 0
        print(f"{task}: {acc:.2f}% ({metrics['correct']}/{metrics['total']})")


if __name__ == "__main__":
    run_prompt_eval()
