from query_gpt3 import read_dataset, configs
from string import Template as StringTemplate

# Test data loading
print("Testing data loading...")
dataset = read_dataset('ico', 'datasets_serialized/ico_list')
# print(f"Loaded {len(dataset)} examples")

# # Test first example
for i in range(0, min(10, len(dataset))):
    example = dataset[i]
    print(f"First example keys: {example.keys()}")
    print(f"Note preview: {example['note'][:]}...")
    print(f"Label: {example['label']}")
    print(f"Answer: {example['answer']}")


# Test prompt generation
# config = configs['ico']
# for i, prompt_temp in enumerate(config['prompts']):
#     prompt = prompt_temp.substitute(**example)
#     print(f"Generated prompt: {prompt}...")