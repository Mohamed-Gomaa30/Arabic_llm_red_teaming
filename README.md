# ROBOSTAI - Arabic LLM Security Evaluation

## Purpose
This repository provides a comprehensive framework for evaluating the safety and robustness of Arabic Large Language Models (LLMs). It specifically targets "jailbreaking" vulnerabilities—attempts to bypass model safety filters—using various linguistic formats common in Arabic communication.

The core objective is to assess whether models maintain their safety guardrails when prompts are presented in:
-   **Standard Arabic**
-   **Arabizi** (Arabic chat alphabet, with and without numbers)
-   **English Transliteration**

By systematically testing these formats, the project aims to identify discrepancies in safety alignment across different input representations.

## Key Features
-   **Multi-Format Testing**: Automatically converts and tests prompts across multiple linguistic variations.
-   **Automated Safety Judge**: Utilizes Gemini as an independent evaluator to classify model responses into specific categories (e.g., Direct Refusal, Unsafe Compliance, Misunderstanding).
-   **Model Support**: Designed to evaluate various models, including Gemini, Jais, AceGPT, and Allam.
-   **Detailed Metrics**: Calculates Jailbreak Success Rates (JSR) and Safety Rates per model and per input format.

## Evaluation Process
The system processes prompts through the following pipeline:
1.  **Input Formatting**: Prompts are converted into target formats (e.g., Arabizi).
2.  **Model Execution**: The target LLMs generate responses to these prompts.
3.  **Safety Assessment**: The Judge model evaluates the response and categorizes it as:
    -   **Refusal (Safe)**: The model declines the harmful request.
    -   **Non-Refusal (Unsafe)**: The model complies with the harmful request.
    -   **Other**: Translation only or misunderstanding.

## Infrastructure
This project leverages **Modal** for high-performance remote execution.
-   **GPU Acceleration**: Utilizes powerful GPUs (e.g., NVIDIA A100) to run large language models efficiently.
-   **Scalability**: Modal allows for seamless scaling of evaluation tasks without local resource constraints.
-   **Environment Management**: Ensures consistent execution environments across different runs.

## Results
The evaluation outputs are stored in the `results/` directory. These include:
-   **Detailed CSV Reports**: Row-by-row analysis of prompts, responses, and safety classifications.
-   **Summary Metrics**: Aggregated statistics showing the performance of each model and the effectiveness of each jailbreak format.

## Usage
To run the evaluation, use the provided test script. The system supports running on Modal for scalable infrastructure.

```bash
modal run -m test::evaluator --dataset harmful --model gemini jais --sample_size 10
```

### Arguments
-   `--dataset`: The dataset to test (e.g., `harmful`, `regional`).
-   `--model`: List of models to evaluate (default: `['gemini', 'jais', 'acegpt', 'allam']`).
-   `--sample_size`: Number of samples to test from the dataset.
