from wandb.integration.openai.fine_tuning import WandbLogger
from openai import OpenAI

client = OpenAI()

training_file_info = client.files.create(
    file=open("train/fine-tuning/train_03_processed.jsonl", "rb"),
    purpose="fine-tune"
)

validation_file_info = client.files.create(
    file=open("test/test_processed.jsonl", "rb"),
    purpose="fine-tune"
)

model = "gpt-4o-mini-2024-07-18"
n_epochs = 3

openai_ft_job_info = client.fine_tuning.jobs.create(
  training_file=training_file_info.id, 
  model=model,
  hyperparameters={"n_epochs": n_epochs},
  validation_file=validation_file_info.id,
  suffix="RO_address_processing_2025_05_02"
)

ft_job_id = openai_ft_job_info.id
ft_job_status = openai_ft_job_info.status

### sync with wandb
WandbLogger.sync(fine_tune_job_id=ft_job_id)
