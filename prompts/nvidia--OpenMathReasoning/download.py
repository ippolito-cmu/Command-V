from datasets import load_dataset
import json

# Load only the 'cot' split in streaming mode
dataset = load_dataset("nvidia/OpenMathReasoning", split="cot", streaming=True)

# Define filter function with proper type conversion
def filter_dataset(example):
    # Convert pass_rate_72b_tir to float before comparison
    try:
        pass_rate = float(example["pass_rate_72b_tir"])
        if not (0.75 <= pass_rate <= 0.9375):
            return False
    except (ValueError, TypeError):
        # Handle cases where conversion fails or value is None
        return False
    
    # Check problem length between 100 and 1000
    problem_length = len(example["problem"])
    if not (100 <= problem_length <= 1000):
        return False
    
    # Check generated_solution length less than 4000
    solution_length = len(example["generated_solution"])
    if solution_length >= 4000:
        return False
    
    return True

# Apply filters to get filtered dataset
filtered_dataset = dataset.filter(filter_dataset)

# Initialize lists to store the data
problem_solution_pairs = []
full_data = []

# Process the filtered dataset
for example in filtered_dataset:
    # Add problem-solution pair
    problem_solution_pairs.append([example["problem"], example["generated_solution"]])
    
    # Add full data
    full_data.append(example)
    
    # Add a progress indicator every 100 examples
    if len(problem_solution_pairs) % 100 == 0:
        print(f"Processed {len(problem_solution_pairs)} examples")
    
    # Set a limit of exactly 2000 examples
    if len(problem_solution_pairs) >= 2000:
        print("Reached 2000 examples")
        break

# Save the problem-solution pairs to train.json
with open("train.json", "w", encoding="utf-8") as f:
    json.dump(problem_solution_pairs, f, ensure_ascii=False, indent=2)

# Save the full data to full_data.json
with open("full_data.json", "w", encoding="utf-8") as f:
    json.dump(full_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(problem_solution_pairs)} examples to train.json and full_data.json")