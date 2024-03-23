import pandas as pd
import argparse
import sys
sys.path.append("/home/coder/projects/Demo/")
from mysql_connection import connect

dataset_list = [
    "/home/coder/projects/LLM_Data/vqa-rad/minigpt4_pred_prompt.csv",
    "/home/coder/projects/LLM_Data/vqa-med/minigpt4_pred_prompt.csv",
    "/home/coder/projects/LLM_Data/vqa-slake/minigpt4_pred_prompt.csv",
    "/home/coder/projects/LLM_Data/vqa-path/minigpt4_pred_prompt.csv"
] 
def evaluate_acc(ans_paths):
    # question,answer,image_name,pred_answer
    # dataset_ans_paths = [
    #     "/home/coder/projects/LLM_Data/vqa-rad/minigpt4_pred_prompt.csv",
    #     "/home/coder/projects/LLM_Data/vqa-med/minigpt4_pred_prompt.csv",
    #     "/home/coder/projects/LLM_Data/vqa-slake/minigpt4_pred_prompt.csv",
    #     "/home/coder/projects/LLM_Data/vqa-path/minigpt4_pred_prompt.csv"
    # ]
    for ans_path in ans_paths:
        df = pd.read_csv(ans_path)
        pred_ans_lists = df['pred_answer'].str.lower()
        ans_lists = df['answer'].str.lower()
        tmp = ans_lists
        closed_num = tmp[tmp== 'yes'].count() + tmp[tmp== 'no'].count()
        closed_corr_count = 0
        open_corr_count = 0
        for pred_ans, ans in zip(pred_ans_lists, ans_lists):
            ans = str(ans)
            pred_ans = str(pred_ans)
            pred_ans = pred_ans.rsplit(".")[0]
            if ans == 'yes' or ans == 'no':
                if pred_ans == ans:
                    closed_corr_count += 1
            else:
                corr_word_count = 0
                pred_words = list(pred_ans.split(" "))
                pred_words = [word.lower() for word in pred_words]
                ans_words = list(ans.split(" "))
                ans_words = [word.lower() for word in ans_words]
                for ans_word in ans_words:
                    if ans_word in pred_words:
                        corr_word_count += 1
                # if corr_word_count/len(ans_lists) >= 0.05:
                if corr_word_count >= 1:
                    open_corr_count += 1

        closed_corr_score = closed_corr_count/closed_num * 100
        open_corr_score = open_corr_count/(len(ans_lists) - closed_num) * 100
        all_corr_score = (closed_corr_count + open_corr_count)/len(ans_lists) * 100

        dataset_name = ans_path.split("/")[-2]
        print("{}'result:closed_corr_score:{}%, open_corr_score:{}%, all_corr_score:{}%".format(dataset_name, closed_corr_score, open_corr_score,all_corr_score))
        return closed_corr_score, open_corr_score,all_corr_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='dataset, choose vqa-rad, vqa-med, vqa-slake, vqa-path')
    parser.add_argument('--record_id', type=int, help='record', default=1)
    # Return args
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset_ans_paths = []
    data_dir = args.data_dir
    if 'vqa-rad' in data_dir:
        dataset_ans_paths.append(dataset_list[0])
    elif 'vqa-med' in data_dir:
        dataset_ans_paths.append(dataset_list[1])
    elif 'vqa-slake' in data_dir:
        dataset_ans_paths.append(dataset_list[2])
    else:
        dataset_ans_paths.append(dataset_list[3])
    temp_a, temp_b, temp_c = evaluate_acc(dataset_ans_paths)
    conn=connect()
    cursor=conn.cursor()
    # print("#######", temp_a, temp_b, temp_c)
    sql="UPDATE `record` SET closed=%f,open=%f,`all`=%f,`status`='%s' where id=%s" % (temp_a, temp_b, temp_c,'complete', args.record_id)
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()   