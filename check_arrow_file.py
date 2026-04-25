from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk('datasets_serialized/ico')

# Check dataset info
print("Dataset info:")
print(dataset)
print(f"Number of examples: {len(dataset)}")
print(f"Features: {dataset.features}")

# View first few examples
print("\nFirst 3 examples:")
for i in range(2):
    print(f"Example {i}:")
    print(f"  note: {dataset[i]['note'][:]}...")  # First 200 chars
    print(f"  riskLevel: {dataset[i]['riskLevel']}")
    print(f"  name: {dataset[i]['name']}")
    print(f"  token_symbol: {dataset[i]['token_symbol']}")
    print("-" * 50)