import argparse
import json
import pathlib
from tqdm import tqdm
import os
import math

# Prompt from stanford alpaca's training script
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def main(args):
    data = json.load(open(os.path.expanduser(args.data_path), "r"))
    # data_path = pathlib.Path(args.data_path)
    # with data_path.open() as f:
    #     data = json.load(f)
    data = get_chunk(data, args.num_chunks, args.chunk_idx)
    new_data = []
    cnt = 1
    for i, line in enumerate(tqdm(data[0:3])):
        if len(line['question_type'].split(', ')) > 1:
            line['question_type']='OTHER'

        image_info="<image>"
        p="Based on the medical image, please generate some Medical Community Question Answering conversation."
        prompt=PROMPT_DICT["prompt_input"]
        pp=prompt.format(instruction=p,input=image_info)

        new_data.append({
            'id': str(cnt),
            'image_organ': line['image_organ'],
            'answer_type': line['answer_type'],
            'question_type': line['question_type'],
            'image': line['image_name'],
            'conversations': [
                {
                    'from': 'human',
                    'value': pp,
                },
                {
                    'from': 'gpt',
                    'value': line['answer'],
                }
            ]
        })
        cnt += 1
        
    json.dump(new_data, open(args.output_path, 'w'), indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/coder/projects/SystemDataset/data_RAD/testset.json')
    parser.add_argument('--output_path', type=str, default='CQAvqarad-data-conversation.json')
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()
    main(args)

