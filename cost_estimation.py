"""
Cost estimation module for OpenAI fine-tuning.
Provides calculation of fine-tuning and inference costs for various OpenAI models.
"""

import json
import numpy as np
import tiktoken
from typing import Dict, List, Any, Tuple

class CostEstimator:
    """
    Cost estimator for OpenAI fine-tuning and inference.
    Calculates costs for different OpenAI models based on token counts.
    """
    
    # Encoding for tokenization
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Model pricing in USD per 1K tokens
    PRICING = {
        # Fine-tuning costs
        'fine_tuning': {
            'gpt-3.5-turbo': 0.0080,          # $0.0080 per 1K tokens for GPT-3.5 Turbo
            'gpt-4': 0.0800,                  # $0.0800 per 1K tokens for GPT-4
            'gpt-4-turbo': 0.0600             # $0.0600 per 1K tokens for GPT-4 Turbo
        },
        # Input inference costs
        'input': {
            'gpt-3.5-turbo': 0.0015,          # $0.0015 per 1K tokens
            'gpt-3.5-turbo-fine-tuned': 0.0030, # $0.0030 per 1K tokens
            'gpt-4': 0.0300,                  # $0.0300 per 1K tokens
            'gpt-4-turbo': 0.0100,            # $0.0100 per 1K tokens
            'gpt-4-fine-tuned': 0.0600,       # $0.0600 per 1K tokens
            'gpt-4-turbo-fine-tuned': 0.0150  # $0.0150 per 1K tokens
        },
        # Output inference costs
        'output': {
            'gpt-3.5-turbo': 0.0020,          # $0.0020 per 1K tokens
            'gpt-3.5-turbo-fine-tuned': 0.0060, # $0.0060 per 1K tokens
            'gpt-4': 0.0600,                  # $0.0600 per 1K tokens
            'gpt-4-turbo': 0.0300,            # $0.0300 per 1K tokens
            'gpt-4-fine-tuned': 0.1200,       # $0.1200 per 1K tokens
            'gpt-4-turbo-fine-tuned': 0.0600  # $0.0600 per 1K tokens
        }
    }
    
    # Constants for fine-tuning
    MAX_TOKENS_PER_EXAMPLE = 4096
    TARGET_EPOCHS = 3
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25
    
    def __init__(self, dataset: List[Dict[str, Any]]):
        """
        Initialize with a dataset.
        
        Args:
            dataset: List of training examples, each with "messages" key
        """
        self.dataset = dataset
        self.calculate_token_statistics()
    
    def print_distribution(self, values: List[int], name: str) -> None:
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
    
    def count_tokens_from_messages(self, messages: List[Dict[str, str]], 
                                  tokens_per_message: int = 3, 
                                  tokens_per_name: int = 1) -> int:
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
                num_tokens += len(self.encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens
    
    def count_role_tokens_from_messages(self, messages: List[Dict[str, str]], role: str) -> int:
        """
        Calculate the number of tokens for a specific role in messages
        
        Args:
            messages: List of message dictionaries
            role: The role to count tokens for (e.g., "assistant", "user", "system")
            
        Returns:
            Token count for the specified role
        """
        num_tokens = 0
        for message in messages:
            if message["role"] == role:
                num_tokens += len(self.encoding.encode(message["content"]))
        return num_tokens
    
    def calculate_token_statistics(self) -> None:
        """
        Calculate token statistics for the dataset
        """
        self.n_missing_system = 0
        self.n_missing_user = 0
        self.n_messages = []
        self.convo_lens = []
        self.assistant_message_lens = []
        self.user_message_lens = []
        self.system_message_lens = []

        for ex in self.dataset:
            messages = ex["messages"]
            if not any(message["role"] == "system" for message in messages):
                self.n_missing_system += 1
            if not any(message["role"] == "user" for message in messages):
                self.n_missing_user += 1
            
            self.n_messages.append(len(messages))
            self.convo_lens.append(self.count_tokens_from_messages(messages))
            
            # Track tokens by role
            assistant_tokens = self.count_role_tokens_from_messages(messages, "assistant")
            user_tokens = self.count_role_tokens_from_messages(messages, "user")
            system_tokens = self.count_role_tokens_from_messages(messages, "system")
            
            self.assistant_message_lens.append(assistant_tokens)
            self.user_message_lens.append(user_tokens)
            self.system_message_lens.append(system_tokens)
        
        self.n_too_long = sum(l > self.MAX_TOKENS_PER_EXAMPLE for l in self.convo_lens)
        
        # Calculate epochs
        self.n_train_examples = len(self.dataset)
        self.n_epochs = self.TARGET_EPOCHS
        
        if self.n_train_examples * self.TARGET_EPOCHS < self.MIN_TARGET_EXAMPLES:
            self.n_epochs = min(self.MAX_DEFAULT_EPOCHS, self.MIN_TARGET_EXAMPLES // self.n_train_examples)
        elif self.n_train_examples * self.TARGET_EPOCHS > self.MAX_TARGET_EXAMPLES:
            self.n_epochs = max(self.MIN_DEFAULT_EPOCHS, self.MAX_TARGET_EXAMPLES // self.n_train_examples)
        
        # Calculate total tokens for billing
        self.n_billing_tokens_in_dataset = sum(min(self.MAX_TOKENS_PER_EXAMPLE, length) for length in self.convo_lens)
        self.total_tokens = self.n_epochs * self.n_billing_tokens_in_dataset
        
        # Average tokens
        self.avg_input_tokens = np.mean(self.user_message_lens) + np.mean(self.system_message_lens)
        self.avg_output_tokens = np.mean(self.assistant_message_lens)
    
    def print_statistics(self) -> None:
        """Print basic statistics about the dataset"""
        print("Num examples missing system message:", self.n_missing_system)
        print("Num examples missing user message:", self.n_missing_user)
        self.print_distribution(self.n_messages, "num_messages_per_example")
        self.print_distribution(self.convo_lens, "num_total_tokens_per_example")
        self.print_distribution(self.assistant_message_lens, "num_assistant_tokens_per_example")
        self.print_distribution(self.user_message_lens, "num_user_tokens_per_example")
        
        print(f"\n{self.n_too_long} examples may be over the {self.MAX_TOKENS_PER_EXAMPLE} token limit, they will be truncated during fine-tuning")
    
    def estimate_cost(self, model_name: str = "gpt-3.5-turbo") -> Dict[str, float]:
        """
        Estimate costs for fine-tuning and inference
        
        Args:
            model_name: Name of the model to use for fine-tuning (default: gpt-3.5-turbo)
            
        Returns:
            Dictionary with cost estimates
        """
        # Calculate fine-tuning costs
        base_model = model_name
        if model_name not in self.PRICING['fine_tuning']:
            print(f"Warning: No pricing found for {model_name}. Using gpt-3.5-turbo pricing.")
            base_model = 'gpt-3.5-turbo'
        
        fine_tuning_cost = (self.total_tokens / 1000) * self.PRICING['fine_tuning'][base_model]
        
        # Get the appropriate model name for inference pricing
        inference_model = f"{base_model}-fine-tuned" if f"{base_model}-fine-tuned" in self.PRICING['input'] else base_model
        
        # Calculate per-request costs
        input_cost_per_request = (self.avg_input_tokens / 1000) * self.PRICING['input'][inference_model]
        output_cost_per_request = (self.avg_output_tokens / 1000) * self.PRICING['output'][inference_model]
        total_cost_per_request = input_cost_per_request + output_cost_per_request
        
        # Calculate monthly costs for various usage levels
        monthly_costs = {}
        for requests in [100, 1000, 10000, 100000]:
            monthly_costs[str(requests)] = requests * total_cost_per_request
        
        return {
            "model": base_model,
            "fine_tuning_cost": fine_tuning_cost,
            "inference_model": inference_model,
            "input_cost_per_1k": self.PRICING['input'][inference_model],
            "output_cost_per_1k": self.PRICING['output'][inference_model],
            "cost_per_request": total_cost_per_request,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "monthly_costs": monthly_costs,
            "total_tokens": self.total_tokens,
            "n_epochs": self.n_epochs
        }
    
    def print_cost_estimate(self, model_name: str = "gpt-3.5-turbo") -> Dict[str, float]:
        """
        Print cost estimates for fine-tuning and inference
        
        Args:
            model_name: Name of the model to use for fine-tuning (default: gpt-3.5-turbo)
            
        Returns:
            Dictionary with cost estimates
        """
        result = self.estimate_cost(model_name)
        
        # Print detailed cost estimates
        print("\n===== COST ESTIMATION =====")
        print(f"Dataset has ~{self.n_billing_tokens_in_dataset} tokens that will be charged for during training")
        print(f"By default, you'll train for {self.n_epochs} epochs on this dataset")
        print(f"By default, you'll be charged for ~{self.total_tokens} tokens")
        
        print(f"\n--- Fine-tuning Cost (Model: {result['model']}) ---")
        print(f"Cost per 1K tokens: ${self.PRICING['fine_tuning'][result['model']]}")
        print(f"Estimated fine-tuning cost: ${result['fine_tuning_cost']:.2f}")
        
        print(f"\n--- Inference Cost (Model: {result['inference_model']}) ---")
        print(f"Average input tokens per request: {result['avg_input_tokens']:.1f}")
        print(f"Average output tokens per request: {result['avg_output_tokens']:.1f}")
        print(f"Input cost per 1K tokens: ${result['input_cost_per_1k']}")
        print(f"Output cost per 1K tokens: ${result['output_cost_per_1k']}")
        print(f"Cost per request: ${result['cost_per_request']:.6f}")
        
        # Estimate costs for various usage levels
        print("\n--- Monthly Usage Estimates ---")
        for requests, cost in result['monthly_costs'].items():
            print(f"{requests} requests per month: ${cost:.2f}")
        
        return result

def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Load a dataset from a JSONL file
    
    Args:
        data_path: Path to the JSONL data file
        
    Returns:
        List of parsed JSON objects
    """
    with open(data_path, 'r', encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def estimate_costs(data_path: str, model_name: str = "gpt-3.5-turbo") -> Dict[str, float]:
    """
    Analyze dataset and estimate fine-tuning costs
    
    Args:
        data_path: Path to the JSONL data file
        model_name: Name of the model to use for fine-tuning
        
    Returns:
        Dictionary with cost estimates
    """
    dataset = load_dataset(data_path)
    estimator = CostEstimator(dataset)
    
    estimator.print_statistics()
    return estimator.print_cost_estimate(model_name)

def compare_models(data_path: str, models: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Compare costs across different OpenAI models
    
    Args:
        data_path: Path to the JSONL data file
        models: List of models to compare (default: ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
        
    Returns:
        Dictionary with cost estimates for each model
    """
    if models is None:
        models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    
    dataset = load_dataset(data_path)
    estimator = CostEstimator(dataset)
    
    results = {}
    
    print("\n===== MODEL COST COMPARISON =====")
    print(f"Dataset: {data_path}")
    print("\nEstimated costs for fine-tuning and inference:")
    print("=" * 80)
    print(f"{'Model':<15} | {'Fine-tuning':<15} | {'Per Request':<15} | {'1K Requests/mo':<15} | {'10K Requests/mo':<15}")
    print("-" * 80)
    
    for model in models:
        result = estimator.estimate_cost(model)
        results[model] = result
        
        fine_tuning_cost = result["fine_tuning_cost"]
        cost_per_request = result["cost_per_request"]
        monthly_1k = result["monthly_costs"]["1000"]
        monthly_10k = result["monthly_costs"]["10000"]
        
        print(f"{model:<15} | ${fine_tuning_cost:<14.2f} | ${cost_per_request:<14.6f} | ${monthly_1k:<14.2f} | ${monthly_10k:<14.2f}")
    
    print("=" * 80)
    print("\nToken usage statistics:")
    print(f"Average input tokens per request: {estimator.avg_input_tokens:.1f}")
    print(f"Average output tokens per request: {estimator.avg_output_tokens:.1f}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Estimate costs for OpenAI fine-tuning")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSONL data file")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model name (default: gpt-3.5-turbo)")
    parser.add_argument("--compare", action="store_true", help="Compare costs across different models")
    
    args = parser.parse_args()
    
    if args.compare:
        results = compare_models(args.data_path)
        
        # Save results to JSON file
        output_file = args.data_path.replace('.jsonl', '_cost_comparison.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")
    else:
        result = estimate_costs(args.data_path, args.model)
        
        # Save result to JSON file
        output_file = args.data_path.replace('.jsonl', f'_{args.model}_cost.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nDetailed result saved to: {output_file}")