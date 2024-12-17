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

python3 data_pipeline --training_input_file_path train/train.txt --test_input_file_path test/test.txt
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

### Estimating inference costs ###
https://platform.openai.com/tokenizer

