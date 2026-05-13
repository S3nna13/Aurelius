# Example: using Pipeline for ETL-style data processing
# 
"""
Example: PowerShell-style object pipeline with Aurelius Pipeline.

Usage:
    python -m examples.pipeline_filter_map
"""

from aurelius_cli.pipeline_processor import Pipeline

if __name__ == "__main__":
    # Example: process a list of numbers
    numbers = list(range(1, 21))

    # Build a pipeline: filter evens, double them, take top 5
    result = (
        Pipeline(numbers)
        .filter(lambda x: x % 2 == 0)          # keep even numbers
        .map(lambda x: x * 2)                  # double
        .sort(reverse=True)                    # descending
        .head(5)                               # top 5
        .collect()
    )

    print("Pipeline result:", result)
    # Expected: [40, 38, 36, 34, 32]