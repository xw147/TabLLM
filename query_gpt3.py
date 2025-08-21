import argparse
import json
import math
import os
from copy import copy
from datetime import datetime
from string import Template as StringTemplate

import yaml

from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict, Dataset
from promptsource.templates import DatasetTemplates, Template


import requests
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

configs = {
    # 'income': {'prompts': [StringTemplate('${note}')]},
    # 'car': {'prompts': [StringTemplate('${note}')]},
    # 'heart': {'prompts': [StringTemplate('${note}')]},
    # 'diabetes': {'prompts': [StringTemplate('${note}')]},
    # 'blood': {'prompts': [StringTemplate('${note}')]},
    # 'bank': {'prompts': [StringTemplate('${note}')]},
    # 'creditg': {'prompts': [StringTemplate('${note}')]},
    # 'calhousing': {'prompts': [StringTemplate('${note}')]},
    # 'jungle': {'prompts': [StringTemplate('${note}')]},
    'ico': {'prompts': [StringTemplate('${note}')]},
}
public_tasks = ['ico', 'income', 'car', 'heart', 'diabetes', 'blood', 'bank', 'creditg', 'calhousing', 'jungle']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=999999)
    args = parser.parse_args()

    return args


def unpack_example(example, task):
    return example


###################################
# The probability is computed from the log-odds between the target tokens (e.g., “Yes” vs. “No”), 
# transformed with a sigmoid to yield a stable, interpretable class probability.
###################################
def post_request(example, model):
    text = json.dumps(example['prompt'])[1:-1]  # sanitize prompt

    print('-' * 80)
    print(text.replace('\\n', '\n'))

    if model != "gpt3":
        raise ValueError("Unexpected model")

    # API call
    url = "https://api.openai.com/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "model": "gpt-3.5-turbo-instruct",
        "prompt": text,
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": 5 # include logprobs for probability calculation
    }
    response = requests.post(url, headers=headers, json=data).json()
    
    if "error" in response:
        raise Exception("ERROR: " + response["error"]["message"])

    choice = response["choices"][0]
    output = choice["text"].strip() # the output token

    # normalize tokens
    def norm(tok): 
        return tok.strip().strip(".,!?;:").lower()

    yes_variants = {"yes", "y"}
    no_variants  = {"no", "n"}

    # get top_logprobs and chosen token
    top = choice.get("logprobs", {}).get("top_logprobs", [{}])[0]
    chosen = choice.get("logprobs", {}).get("tokens", [output])[0]
    chosen_lp = top.get(chosen, 0.0)

    # assign logprobs
    yes_lp = max((v for k,v in top.items() if norm(k) in yes_variants), default=None)
    no_lp  = max((v for k,v in top.items() if norm(k) in no_variants), default=None)

    # ensure chosen token is included
    if norm(chosen) in yes_variants:
        yes_lp = chosen_lp if yes_lp is None else max(yes_lp, chosen_lp)
    elif norm(chosen) in no_variants:
        no_lp = chosen_lp if no_lp is None else max(no_lp, chosen_lp)

    # compute probability
    if yes_lp is not None and no_lp is not None:
        log_odds = yes_lp - no_lp #log pro difference
        prob_yes = 1 / (1 + math.exp(-log_odds)) # sigmoid
    else:
        prob_yes = 1.0 if norm(output) in yes_variants else 0.0

    pred_label = 1 if norm(output) in yes_variants else 0

    print(f"Predicted label: {pred_label}, Probability: {prob_yes:.4f}")
    print('-' * 80)

    return pred_label, prob_yes




def submit_req(item, model, max_tries=300, sleep_sec=20):
    for i in range(max_tries):
        try:
            return post_request(item, model)
        except Exception as e:
            print(e)
            print(f"Request error; retrying in {sleep_sec} sec\n")
            time.sleep(sleep_sec)
    print("RAN OUT OF QUOTA or issues w/ API; quitting")
    return None, None


# From: https://stackoverflow.com/questions/2148119/how-to-convert-an-xml-string-to-a-dictionary
def dictify(r, root=True):
    if root:
        return {r.tag: dictify(r, False)}
    d = copy(r.attrib)
    if r.text:
        d["_text"] = r.text
    for x in r.findall("./*"):
        if x.tag not in d:
            d[x.tag] = []
        d[x.tag].append(dictify(x, False))
    return d


def read_dataset(task, input_file):
    # Get dataset as list of entities
    if task in public_tasks:
        # External dataset are not yet shuffled, so do it now
        orig_data = load_from_disk(input_file)

        # Without template
        # input_list = [{'note': x['note'], 'label': x['label']} for x in orig_data]

        # Load template
        yaml_dict = yaml.load(open('/work/TabLLM/templates/templates_' + task + '.yaml', "r"), Loader=yaml.FullLoader)
        prompts = yaml_dict['templates']
        # Return a list of prompts (usually only a single one with dataset_stash[1] name)
        templates_for_custom_tasks = {
            'income': '50000_dollars',
            'car': 'rate_decision',
            'heart': 'heart_disease',
            'diabetes': 'diabetes',
            'creditg': 'creditg',
            'bank': 'bank',
            'blood': 'blood',
            'jungle': 'jungle',
            'calhousing': 'calhousing',
            'ico': 'ico_fraud_detection',
        }
        temp = [t for k, t in prompts.items() if t.get_name() == templates_for_custom_tasks[task]][0]

        input_list = [{'note': temp.apply(x)[0], 'true_answer': temp.apply(x)[1], 'true_label': x['label']} for x in orig_data]
    else:
        raise ValueError("Invalid task name")

    dataset = [unpack_example(ex, task) for ex in input_list]
    return dataset


def main():
    time.sleep(0)
    args = parse_args()
    assert args.task in configs.keys()
    config = configs[args.task]
    outputs = pd.DataFrame()

    dataset = read_dataset(args.task, args.input)
    start_time = datetime.now().strftime("-%Y%m%d-%H%M%S")

    for k, example in enumerate(dataset):
        try:
            # if k >= 3:
            #     break
            # Only consider examples in provided range
            if k < args.start_index or k >= args.end_index:
                continue
            print(f"{k}/{len(dataset)} (from {args.start_index} to {args.end_index})")

            # Copy input into outputs
            output = example.copy()

            for i, prompt_temp in enumerate(config['prompts']):
                example['note'] = example['note'].strip()
                prompt = prompt_temp.substitute(**example)
                example['prompt'] = (prompt_temp.substitute(**example)).strip()
                output['prompt' + str(i)] = prompt
                if args.model == 'gpt3':
                    if args.task in public_tasks:
                        pred_label, pred_prob = submit_req(example, args.model)
                    else:
                        pred_label, pred_prob = submit_req(example, args.model)
                    output['pred_label'] = pred_label
                    output['pos_prob'] = pred_prob
                    time.sleep(0)
                outputs = pd.concat([outputs, pd.Series(output).to_frame(1).T], ignore_index=True)

                # if args.model == 'gpt3' and k % 50 == 0:
                if args.model == 'gpt3' and k % 50 == 0:
                    # Write temporary results out
                    outputs.to_csv('output/outputs-' + args.task + start_time + '.csv', index=False)

        except Exception as e:
            print("Error occurred: " + str(e))

    if args.model == 'gpt3':
        # Final output
        outputs.to_csv('output/outputs-' + args.task + start_time + '.csv', index=False)


if __name__ == '__main__':
    import sys
    INPUT_PATH = 'C:\\work\\TabLLM\\datasets_serialized\\ico_list'
    sys.argv = ['query_gpt3.py', '--task', 'ico', '--input', INPUT_PATH, '--model', 'gpt3']
    main()
