from datasets import load_from_disk

# Load the entire dataset
dataset = load_from_disk('C:/work/TabLLM/datasets_serialized/ico/ico_list/data.arrow')
print(dataset)

# View first few examples
print(dataset[:5])

# Convert to pandas for easier viewing
df = dataset.to_pandas()
print(df.head())