from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk('datasets_serialized/ico_list')

# Check dataset info
print("Dataset info:")
print(dataset)
print(f"Number of examples: {len(dataset)}")
print(f"Features: {dataset.features}")

# View first few examples
print("\nFirst 3 examples:")
for i in range(3):
    print(f"Example {i}:")
    print(f"  note: {dataset[i]['note'][:200]}...")  # First 200 chars
    print(f"  label: {dataset[i]['label']}")
    print("-" * 50)