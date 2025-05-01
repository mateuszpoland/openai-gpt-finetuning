#!/usr/bin/env python
"""
Model cost comparison tool for OpenAI fine-tuning.
This script estimates costs for fine-tuning and inference across different OpenAI models.
"""

import argparse
import json
from cost_estimation import compare_models

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Compare costs across different OpenAI models")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSONL data file")
    parser.add_argument("--models", type=str, help="Comma-separated list of models to compare (default: gpt-3.5-turbo,gpt-4,gpt-4-turbo)")
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_args()
    
    models = None
    if args.models:
        models = [model.strip() for model in args.models.split(',')]
    
    results = compare_models(args.data_path, models)
    
    # Save results to JSON file
    output_file = args.data_path.replace('.jsonl', '_cost_comparison.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()