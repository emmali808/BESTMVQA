import time
import change_str
import model_info
import re

# i=0
# while i<100:
#     i=i+1
#     print(i)
#     time.sleep(2)
models = model_info.models
total_strs = change_str.test_str

def check_change(content=""):
    for str in total_strs:
        match = re.search(str, content, re.I)
        # 符合句式
        if match:
            # 要修改模型
            if match.group(1).upper() in models.keys() and match.group(2).upper() in models.keys():
               print("************************")
               print(match.group(1).upper())
               print(match.group(2).upper())
            # 要修改数据集
            elif match.group(1).upper() in ["VQA-RAD", "PATHVQA", "SLAKE", "Med-VQA"] and match.group(2).upper() in ["VQA-RAD", "PATHVQA", "SLAKE", "Med-VQA"]:
                print("#####################")
                print(match.group(1).upper())
                print(match.group(2).upper())

check_change("change MEVF to MMBERT please")