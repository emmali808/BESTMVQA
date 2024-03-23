import pandas as pd
import json

with open('/home/coder/projects/Demo/VQASystem/result/vqa_submit_last.json') as f:
    ans_js = json.load(f)

user_data_path = '/home/coder/projects/SystemDataset/robot/robot.csv'
final_data_df = pd.read_csv(user_data_path)
# img_path = str(final_data_df['img_path'][0])
# content = str(final_data_df['content'][0])

# save answer into csv
final_data_df['pre_ans'][0] = ans_js[0]['answer']
print("######", final_data_df['pre_ans'][0])
final_data_df.to_csv(user_data_path, index=False)        