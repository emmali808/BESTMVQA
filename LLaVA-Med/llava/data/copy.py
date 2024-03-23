import argparse
import json
import pathlib
from tqdm import tqdm
import os
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def main(args):
    # data = json.load(open(os.path.expanduser(args.data_path), "r"))
    # data = get_chunk(data, args.num_chunks, args.chunk_idx)
    # new_data = []
    # cnt = 1
    # with open('ovqa_output.txt','w') as f:
    #     for i, line in enumerate(tqdm(data)):
    #         f.write(line['question']+"\n")
    
    with open("/home/coder/projects/SystemDataset/VQA-Med-2019/ImageClef-2019-VQA-Med-Test/VQAMed2019_Test_Questions.txt", "r") as f:  # 打开文件
        data = f.readlines()
        print(data)
    newd=[]
    for d in data:
        newd.append(d.split('|')[1])
    print(newd)
    with open('medvqa_output.txt','w') as f:
        for i in newd:
            f.write(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/coder/projects/SystemDataset/data_OVQA_as_RAD/testset.json',choices=['/home/coder/projects/SystemDataset/data_OVQA_as_RAD/testset.json','/home/coder/projects/Med-VQA/data_PATH/testset.json','/home/coder/projects/SystemDataset/data_RAD/testset.json','/home/coder/projects/Med-VQA/data_SLAKE/train_test/testset.json'])
    parser.add_argument('--output_path', type=str, default='szx_rad_path.json')
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()
    main(args)

