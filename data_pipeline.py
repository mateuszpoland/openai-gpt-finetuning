import json
import tiktoken
import numpy as np
from collections import defaultdict
import argparse
from jsonl_training_data_processor import JSONLTrainingDataProcessor
from jsonl_test_data_processor import JSONLTestDataProcessor
from cost_estimation import CostEstimator, estimate_costs

encoding = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    """
    Calculate the total number of tokens in a list of messages
    
    Args:
        messages: List of message dictionaries
        tokens_per_message: Number of tokens per message (default: 3)
        tokens_per_name: Number of tokens per name (default: 1)
        
    Returns:
        Total token count
    """
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens
    
def num_assistant_tokens_from_messages(messages):
    """
    Calculate the number of tokens in assistant messages only
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Assistant token count
    """
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens    

def print_distribution(values, name):
    """
    Print statistical distribution of values
    
    Args:
        values: List of numeric values
        name: Name of the distribution for display
    """
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

# Data processing functions
def create_training_jsonl_file(input_file_path: str, output_file_path: str, prompt_version: str = None):
    """
    Create JSONL file for training data
    
    Args:
        input_file_path: Path to the input training data file
        output_file_path: Path where the processed JSONL will be saved
        prompt_version: Specific prompt version to use (optional)
    """
    training_data_processor = JSONLTrainingDataProcessor(input_file_path, output_file_path, prompt_version)
    training_data_processor.process()

def create_test_jsonl_file(input_file_path: str, output_file_path: str, prompt_version: str = None):
    """
    Create JSONL file for test data
    
    Args:
        input_file_path: Path to the input test data file
        output_file_path: Path where the processed JSONL will be saved
        prompt_version: Specific prompt version to use (optional)
    """
    test_data_processor = JSONLTestDataProcessor(input_file_path, output_file_path, prompt_version)
    test_data_processor.process()

def validate_data(data_path):
    """
    Validate the formatted JSONL dataset for OpenAI fine-tuning
    
    Args:
        data_path: Path to the JSONL data file to validate
    """
    with open(data_path, 'r', encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]
    
    print(f"Num examples: {len(dataset)}")
    print("First example: \n")
    for message in dataset[0]["messages"]:
        print(message)
    
    format_errors = defaultdict(int)
    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
            
            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
            
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1
    
    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

def count_tokens(data_path: str):
    """
    Count tokens in the first example of the dataset
    
    Args:
        data_path: Path to the JSONL data file
    """
    with open(data_path, 'r', encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    messages = dataset[0]["messages"]
    num_tokens = num_tokens_from_messages(messages)
    num_assistant_tokens = num_assistant_tokens_from_messages(messages)

    print(f"number of tokens: {num_tokens}")
    print(f"number of assistant tokens: {num_assistant_tokens}")

def cost_estimation(data_path: str, model_name: str = "gpt-3.5-turbo"):
    """
    Analyze dataset and estimate fine-tuning costs
    
    Args:
        data_path: Path to the JSONL data file
        model_name: Name of the model to use for fine-tuning (default: gpt-3.5-turbo)
        
    Returns:
        Dictionary with cost estimates
    """
    return estimate_costs(data_path, model_name)

    
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create a fine-tuning .jsonl formatted data for OpenAI")
    parser.add_argument("--training_input_file_path", type=str, help="Input file path for training data")
    parser.add_argument("--test_input_file_path", type=str, help="Input file path for test data")
    parser.add_argument("--prompt_version", type=str, help="Prompt version to use (e.g., v1, v4). If not specified, uses the latest.", default=None)
    parser.add_argument("--model", type=str, help="Model to use for fine-tuning (e.g., gpt-3.5-turbo, gpt-4-turbo)", default="gpt-3.5-turbo")
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_args()

    output_training_file_path = args.training_input_file_path.replace('.txt', '_processed.jsonl')
    output_test_file_path = args.test_input_file_path.replace('.txt', '_processed.jsonl')

    create_training_jsonl_file(args.training_input_file_path, output_training_file_path, args.prompt_version)
    create_test_jsonl_file(args.test_input_file_path, output_test_file_path, args.prompt_version)

    validate_data(output_training_file_path)
    count_tokens(output_training_file_path)
    cost_estimation(output_training_file_path, args.model)

if __name__ == "__main__":
    main()