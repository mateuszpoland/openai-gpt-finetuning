### Fine Tuning ###

**Prerequisites**:

 - OpenAI account and OpenAI API Key. You should create `.env` file and put the key there. The template is `.env.local` file. 
 - Github Access Token. Do the same as with OpenAI key. Both should be environment variables.
 - optional(recommended): wandb account: https://wandb.ai/site/


**Resources:**
1. Preparing data for fine-tuning
https://cookbook.openai.com/examples/chat_finetuning_data_prep
https://www.pinecone.io/learn/fine-tune-gpt-3.5
https://wandb.ai/prompt-eng/openai-finetune-integration/reports/How-to-Fine-Tune-Your-OpenAI-GPT-3-5-and-GPT-4-Models-with-Weights-Biases--Vmlldzo2MDEwMjEw

To see how the fine-tuning files: `train/train_processed.jsonl` and `test/test_processed.jsonl` are created, run:

```
docker-compose up -d --build
docker-compose exec app bash

# Use the latest prompt version (default)
python3 data_pipeline.py --training_input_file_path train/fine-tuning/train.txt --test_input_file_path test/test.txt

# Or specify a specific prompt version
python3 data_pipeline.py --training_input_file_path train/fine-tuning/train.txt --test_input_file_path test/test.txt --prompt_version v1
```

This will create `.jsonl` files that are valid files for fine-tuning OpenAI Assistants v2.

Example files are production-ready and are used for **Named Entity Recognition** task.

2. Running a fine-tuning job

Once the files are created, you can start fine-tuning process by:

```
python3 fine_tuning.py
```

### Basic Monitoring and feedback loop for fine-tuning

**Weights and Biases tool**
https://wandb.ai/
**Data Version Control**
https://dvc.org/

### Prompt System

The project now includes a structured prompt system for managing different prompt versions:

1. Prompt class structure:
   - Each prompt version is defined as a Python class in `train/system_prompts/prompt_vX.py`
   - All prompt classes inherit from the base `Prompt` class
   - Each prompt class encapsulates both system message and user message templates

2. Using different prompt versions:
   - Use the command line argument `--prompt_version` to specify which prompt to use
   - If not specified, the latest version is used automatically
   - Available versions: v1, v4 (and any new versions you add)

3. Adding new prompts:
   - Create a new file in `train/system_prompts/` following the naming convention `prompt_vX.py`
   - Implement a class named `PromptVX` extending the base `Prompt` class
   - The new prompt will be automatically detected by the `PromptStrategy`

Example of creating a new prompt version:
```python
from .prompt_v4 import Prompt

class PromptV5(Prompt):
    def __init__(self):
        system_message = """Your system message here..."""
        user_message_template = """Your user message template here...
{input_address}"""
        
        super().__init__(
            version="v5",
            system_message=system_message,
            user_message_template=user_message_template
        )
```

### Cost Estimation and Model Comparison ###

The project includes a comprehensive cost estimation system that helps you understand both fine-tuning and inference costs before committing to a fine-tuning job.

1. Cost Estimation Features:
   - Fine-tuning cost calculation based on token count and model pricing
   - Inference cost estimation for fine-tuned models
   - Token distribution analysis (input/output token counts)
   - Monthly cost projections at different usage levels
   - Support for multiple OpenAI models (GPT-3.5 Turbo, GPT-4, GPT-4 Turbo)

2. Using the Cost Estimation:

   **With data pipeline:**
   ```bash
   # Cost estimation is included automatically when processing data
   python data_pipeline.py --training_input_file_path train/fine-tuning/train.txt --test_input_file_path test/test.txt --prompt_version v5 --model gpt-3.5-turbo
   ```

   **Standalone cost estimation:**
   ```bash
   # Estimate costs for a specific model
   python cost_estimation.py --data_path train/fine-tuning/train_processed.jsonl --model gpt-3.5-turbo
   
   # Compare costs across all supported models
   python cost_estimation.py --data_path train/fine-tuning/train_processed.jsonl --compare
   ```

   **Model comparison tool:**
   ```bash
   # Compare costs across all default models
   python model_cost_comparison.py --data_path train/fine-tuning/train_processed.jsonl
   
   # Compare specific models
   python model_cost_comparison.py --data_path train/fine-tuning/train_processed.jsonl --models "gpt-3.5-turbo,gpt-4-turbo"
   ```

3. Output Information:
   - Token count statistics and distributions
   - Estimated fine-tuning costs for the entire dataset
   - Per-request inference costs
   - Monthly cost projections at various usage levels (100, 1K, 10K, 100K requests)
   - JSON export of all cost estimates for further analysis

These cost estimation tools can help you make informed decisions about which models to use and budget appropriately for both fine-tuning and production inference costs.

Additional resources:
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) - Visualize token counting
- [OpenAI Pricing](https://openai.com/pricing) - Official pricing information

