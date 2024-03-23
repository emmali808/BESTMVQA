import subprocess
import time
import numpy as np
import pandas as pd
import streamlit as st
import model_info
import pymysql
import random
import change_str
import re
import os
import plotly.express as px


def check_change(content=""):
    for str in total_strs:
        match = re.search(str, content, re.I)
        # 符合句式
        if match:
            print("*******************")
            print(match.group(1).upper())
            print(match.group(2).upper())
            # 要修改模型
            if match.group(1).upper() in models.keys() and match.group(2).upper() in models.keys():
                st.session_state["params"][match.group(2).upper()] = {}
                st.session_state["params"][match.group(
                    2).upper()]["model_name"] = match.group(2).upper()
                if "datasets" in st.session_state["params"][match.group(1).upper()].keys() and len(st.session_state["params"][match.group(1).upper()]["datasets"])>0:
                    st.session_state["params"][match.group(2).upper(
                    )]["datasets"]=[]
                    st.session_state["params"][match.group(2).upper(
                    )]["datasets"] = st.session_state["params"][match.group(1).upper()]["datasets"]
                del (st.session_state["params"][match.group(1).upper()])
                return {
                    "if_change": True,
                    "ret": {
                        "content": [
                            {"data": "Update successfully!", "show_type": st.write},
                        ],
                        "next": st.session_state["routine_state"],
                    },
                }
            # 要修改数据集
            elif match.group(1).upper() in ["VQA-RAD", "PATHVQA", "SLAKE", "MED-VQA"] and match.group(2).upper() in ["VQA-RAD", "PATHVQA", "SLAKE", "MED-VQA"]:
                for model in st.session_state["params"]:
                    model = st.session_state["params"][model]
                    model["datasets"].remove(match.group(1).upper())
                    model["datasets"].append(match.group(2).upper())
                return {
                    "if_change": True,
                    "ret": {
                        "content": [
                            {"data": "Update successfully!", "show_type": st.write},
                        ],
                        "next": st.session_state["routine_state"],
                    },
                }
            else:
                return {
                    "if_change": True,
                    "ret": {
                        "content": [
                            {"data": " Sorry,I don't understand what you mean.If you want to :blue[modify the parameters selected earlier], please :blue[_reply with the one and what you would like to change to_]. If you wish to :green[answer the question], please :green[_reply directly to the options I have provided._]Maybe consider if you type the name right.", "show_type": st.write},
                        ],
                        "next": st.session_state["routine_state"],
                    },
                }
    return {"if_change": False}


def begin_ques():
    str = "Hello! What would you like me to do for you：⭐Data training⭐Generate evaluation report"
    return str


def begin_ans(content=""):
    if content == "Generate evaluation report":
        return {
            "content": [
                {"data": "Okay, we're about to start the training model process.",
                    "show_type": st.write},
            ],
            "next": "model",
        }
    elif content == "Data training":
        return {
            "content": [
                {"data": "we're about to start the test.", "show_type": st.write},
            ],
            "next": "datatraining",
        }
    # 正则表达式提取
    pattern_report = r"Generate evaluation report"
    matches_report = re.findall(pattern_report, content, re.I)
    if len(matches_report)==0:
        pattern_training =r"Data training"
        matches_training = re.findall(pattern_training, content, re.I)
        if len(matches_training)==0:
            msg = {
            "ret": {
                "content": [
                    {"data": " Sorry,I don't understand what you mean.Please reply :blue[_directly to the options I have provided._]For example,Data training.",
                     "show_type": st.write},
                ],
                "next": st.session_state["routine_state"],
            },
        }
        else:
            return {
            "content": [
                {"data": "we're about to start the test.", "show_type": st.write},
            ],
            "next": "datatraining",
        }
    else:
        return {
            "content": [
                {"data": "Okay, we're about to start the training model process.",
                    "show_type": st.write},
            ],
            "next": "model",
        }
    msg = {
        "ret": {
            "content": [
                {"data": " Sorry,I don't understand what you mean.Please reply :blue[_directly to the options I have provided._]For example,Data training.",
                    "show_type": st.write},
            ],
            "next": st.session_state["routine_state"],
        },
    }
    return msg["ret"]


def datatraining_ques():
    str = "Please upload the picture and the question you want to ask."
    return str


def get_models_pred_ans(img_path, content):
    save_path = '/home/coder/projects/SystemDataset/robot/robot.csv'
    data = {'img_path': [img_path], 'content': [content], 'pre_ans': [""]}
    data_df = pd.DataFrame(data)
    data_df.to_csv(save_path, index=False)
    sh_lists = [
        'bash /home/coder/projects/METER/demo_for_robot.sh',
        'bash /home/coder/projects/PTUnifier-share/run_scripts/demo_for_robot.sh',
        'bash /home/coder/projects/MMBERT/vqarad/demo_for_robot.sh',
        'bash /home/coder/projects/MiniGPT-4/demo_for_robot.sh',
    ]
    model_lists = [
        'METER',
        'PTUnifier',
        'MMBert',
        'MiniGPT-4',
    ]
    pre_ans = ""
    similar_report_js = {}
    for sh_list, model_list in zip(sh_lists, model_lists):
        os.system(sh_list)
        final_data_df = pd.read_csv(save_path)
        pre_ans = '{}:blue[_{}_]\'s answer is {}\n\n'.format(
            pre_ans, model_list, final_data_df['pre_ans'][0])
        if 'MiniGPT' in sh_list:
            similar_report_js['img_path'] = final_data_df['similar_img_path'][0]
            similar_report_js['similar_txt_content'] = final_data_df['similar_txt_content'][0]

    return pre_ans, similar_report_js

def myimage(str):
    st.image(image=str,width=250)
    # st.image(image=str,width=100)

def datatraining_ans(img_path, content=""):
    with st.spinner('testing...'): 
        # pre_ans,  similar_report_js = get_models_pred_ans(img_path, content)
        pre_ans,  similar_report_js = "dsf", "fasdfs"

    # test by hxj 记得删掉
    with open("/home/coder/projects/Demo/VQASystem/test01.txt","r", encoding="utf-8") as f:
        test_demo = f.read()
    return {
        "content": [
            {"data": pre_ans, "show_type": st.write},
            {"data": "By searching the database, :blue[_the most similar EMR_] is", "show_type": st.write},
            # {"data": similar_report_js['img_path'], "show_type": myimage},
            # {"data": similar_report_js['similar_txt_content'], "show_type": st.write},
            # 记得删掉
            {"data": '/home/coder/projects/SystemDataset/samples/images/synpic42202.jpg', "show_type": myimage},
            {"data": test_demo, "show_type": st.write},
        ],
        "next": "begin",
    }


def model_ques():
    str = "Please select the :blue[_models_] you want. The following models are available："
    str += "⭐"+" ⭐".join(models.keys())
    return str


def model_ans(content=""):
    content = content.upper()
    # 是否有更改
    check_result = check_change(content)
    if check_result["if_change"]:
        return check_result["ret"]

    # 正则表达式提取出和models相同的model
    pattern = r'\b(?:' + '|'.join(re.escape(model)
                                  for model in models.keys()) + r')\b'
    matches = re.findall(pattern, content, re.I)
    if len(matches)==0:
        return {
        "content": [
            {"data": "You seem to input the wrong model name,can you input the models correctly again?", "show_type": st.write},
        ],
        "next": "model",
        }   
    for match in matches:
        # 字典的嵌套，创建一个 st.session_state["params"]{"model1":{"model_name":mmq,}}的字典
        st.session_state["params"][match] = {}
        st.session_state["params"][match]["model_name"] = match

    answer = "You have chosen "
    answer += ",".join(matches)
    answer += " model"
    # 返回结果
    return {
        "content": [
            {"data": answer, "show_type": st.write},
        ],
        "next": "dataset",
    }


def dataset_ques():
    str = "Please select the :blue[_datasets_] you want. The following datasets are available："
    arr = ["VQA-RAD", "PATHVQA", "SLAKE", "MED-VQA"]
    str += "⭐"+" ⭐".join(arr)
    return str


def dataset_ans(content=""):
    content = content.upper()
    arr = ["VQA-RAD", "PATHVQA", "SLAKE", "MED-VQA"]

    # 是否有更改
    check_result = check_change(content)
    if check_result["if_change"]:
        return check_result["ret"]

    pattern = r'\b(?:' + '|'.join(re.escape(dataset)
                                  for dataset in arr) + r')\b'
    matches = re.findall(pattern, content, re.I)
    if len(matches)==0:
        return {
        "content": [
            {"data": "You seem to input the wrong dataset name,can you input the datasets correctly again?", "show_type": st.write},
        ],
        "next": "dataset",
        }   
    
    # match是用户选择的数据集之一
    for model in st.session_state['params']:
        model = st.session_state["params"][model]
        model["datasets"] = []
        for match in matches:
            model["datasets"].append(match)

    answer = "You have chosen "
    answer += ",".join(matches)
    answer += " dataset"

    # 返回结果
    return {
        "content": [
            {"data": answer, "show_type": st.write},
        ],
        "next": "result",
    }


def result_ques():
    str = "All parameters have been selected, whether to start training?Reply :blue[_yes_] or :blue[_no_]."
    return str


def result_ans(content=""):
    # 是否有更改
    check_result = check_change(content)
    if check_result["if_change"]:
        return check_result["ret"]

    conn = pymysql.connect(
        host=st.secrets["host"],
        user=st.secrets["username"],
        password=st.secrets["password"],
        database=st.secrets["database"]
    )

    auth_key = st.session_state['user'] + \
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    ids = []

    for model in st.session_state["params"]:
        model = st.session_state["params"][model]
        model_name = model["model_name"]
        record_name = model["model_name"]+"_MYTEST"
        if model["model_name"] == "MINIGPT-4":
            model_name = "MiniGPT4"
        elif model["model_name"] == "PTUNIFIER":
            model_name = "PTU"
        print(model)
        for dataset_name in model["datasets"]:
            if model["model_name"] not in best_params.keys():
                name = model["model_name"]+'_'+dataset_name
            else:
                name = model["model_name"]
            epoch = best_params[name]["epoch"]
            lr = best_params[name]["lr"]
            batchsize = best_params[name]["batchsize"]
            attention = best_params[name]["attention"]
            RNN = best_params[name]["RNN"]

            print("model:"+model_name)
            print("dataset:"+dataset_name)
            # 查找数据库model=model_name,dataset=dataset_name,max(open),max(closed)的一条记录，如果有,把记录保存到元组record=()中，列表records收集record,把该记录的id保存到数组ids中
            cursor = conn.cursor()
            if dataset_name.upper()=="MED-VQA":
                db_dataset_name="Med-VQA"
            else:
                db_dataset_name=dataset_name

            cursor.execute(
                "SELECT * FROM vqa.record WHERE model = '%s' AND dataset = '%s' ORDER BY 'all' DESC LIMIT 1;" % (model_name, db_dataset_name))
            record = cursor.fetchone()  # 获取一条记录，保存到元组中
            # 检测是否存在匹配的记录
            cursor.execute("SELECT COUNT(*) FROM detail WHERE record_id = '%d';" % (record[0]))
            count = cursor.fetchone()[0]
            cursor.close()
            if count>0:
                ids.append(record[0])  # 将记录的id添加到数组中
                print("Find one record in db!")
                continue
            
            cursor = conn.cursor()
            # 如果没有，使用以下插入语句运行
            cursor.execute("INSERT INTO vqa.record (auth_key,model,record_name,epoch,batch_size,dataset,attention,rnn,lr,status) VALUES ('%s','%s','%s',%d,%d,'%s','%s','%s','%f','%s');" % (
                auth_key, model_name, record_name, int(epoch), int(batchsize), db_dataset_name, attention, RNN, float(lr), "running"))
            record_id = conn.insert_id()
            ids.append(record_id)
            print("Add one record in db!")
            #print(len(ids))
            cursor.close()
            conn.commit()
            if model["model_name"] in ['PTUNIFIER', 'METER', 'TCL', 'MEDVINT']:
                subprocess.run(["bash", models[model["model_name"]]['path']
                                [dataset_name], epoch, lr, batchsize, str(record_id)])
                #print("*****************")
            else:
                subprocess.run(["bash", models[model["model_name"]]['path'][dataset_name], attention,
                                models[model["model_name"]]['dataset'][dataset_name], epoch, lr, batchsize, RNN, str(record_id)])
                #print("*****************")
    details = []
    # 此时，ids保存了：在数据库里查到已有的不用训练的记录的id，和没查到记录，插进数据库的id。在此处查询auth_key=auth_key,status="complete"的记录数是否等于ids的长度，否则while循环
    print('training...')
    print("ids:")
    print(ids)
    with st.spinner('training...'): 
        while True:
            cursor=conn.cursor()
            # 查询指定ID对应的记录的status字段
            query = f"SELECT status FROM vqa.record WHERE id IN ({', '.join(map(str, ids))})"
            cursor.execute(query)
            # 获取查询结果
            results = cursor.fetchall()
            print("results:")
            print(results)
            # 检查所有记录的state字段是否都为 "complete"
            if all(result[0] == 'complete' for result in results):
                break  # 结束循环
            else:
                print("Not all records have 'complete' state. Retrying in 5 seconds...")
                time.sleep(5)  # 等待五秒后再次查询
            cursor.close()

    cursor=conn.cursor()
    query = f"SELECT * FROM vqa.record WHERE id IN ({', '.join(map(str, ids))})"
    cursor.execute(query)
    # 这里应该是把cursor.fetchall()查到的所有记录加进records里去，records此前已经保存了一些已经在数据库查到的训练记录
    records = cursor.fetchall()
    print("records:")
    print(records)
    columnDes = cursor.description  # 获取连接对象的描述信息
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]  # 获取列名
    record_df = pd.DataFrame([list(i) for i in records], columns=columnNames)
    #st.dataframe(record_df[['model', 'dataset', 'open','closed','all']], use_container_width=True)  # 这里展示的表的列应该还要加上open,closed,all
    ids = record_df['id']
    cursor.close()

    cursor = conn.cursor()
    for id in ids:
        cursor.execute("SELECT t1.id, CONCAT(t1.dataset, ' ', t1.model) AS dataset_model, t2.epoch, t2.loss FROM vqa.record AS t1 JOIN vqa.detail AS t2 ON t1.id = t2.record_id where t1.id=%s;"%(id))
        ret = cursor.fetchall()
        for detail in ret:
            details.append(detail)
    print("details:")
    print(details)
    if len(details) > 0:
        columnDes = cursor.description
        columnNames = [columnDes[i][0] for i in range(len(columnDes))]
        detail_df = pd.DataFrame([list(i)
                                 for i in details], columns=columnNames)
        print(detail_df)
        epoch_fig = px.line(detail_df, x='epoch', y='loss', color='dataset_model',
                            
                            title='epoch-loss', log_y=True, markers=True)
        epoch_fig.update_layout(
            legend_title=None, xaxis_title='epoch', yaxis_title='loss')
        # st.plotly_chart(epoch_fig, use_container_width=True)
    # 返回结果
    return {
        "content": [
            {"data": "The results of your training are as follows：",
                "show_type": st.write},
            {"data": record_df[['model', 'dataset', 'open','closed','all']], "show_type": st.dataframe},
            {"data": epoch_fig, "show_type": st.plotly_chart},
        ],
        "next": "begin",
    }


models = model_info.models
total_strs = change_str.total_strs
half_strs = change_str.half_strs
best_params = {
    'MEVF': {
        "epoch": "20",
        "batchsize": "32",
        "attention": "BAN",
        "RNN": "LSTM",
        "lr": "0.005000",
    },
    'CR': {
        "epoch": "20",
        "batchsize": "64",
        "attention": "BAN",
        "RNN": "GRU",
        "lr": "0.005000",
    },
    'MMBERT': {
        "epoch": "60",
        "batchsize": "16",
        "attention": "",
        "RNN": "ResNet152",
        "lr": "0.000100",
    },
    'VQAMIX': {
        "epoch": "80",
        "batchsize": "32",
        "attention": "BAN",
        "RNN": "LSTM",
        "lr": "0.020000",
    },
    'MMQ': {
        "epoch": "60",
        "batchsize": "32",
        "attention": "SAN",
        "RNN": "LSTM",
        "lr": "0.005000",
    },
    'CSMA': {
        "epoch": "60",
        "batchsize": "32",
        "attention": "CMSA",
        "RNN": "LSTM",
        "lr": "0.005000",
    },
    'PTUNIFIER_VQA-RAD': {
        "epoch": "15",
        "batchsize": "16",
        "attention": "",
        "RNN": "",
        "lr": "0.001000",
    },
    'PTUNIFIER_Med-VQA': {
        "epoch": "15",
        "batchsize": "8",
        "attention": "",
        "RNN": "",
        "lr": "0.001000",
    },
    'PTUNIFIER_SLAKE': {
        "epoch": "15",
        "batchsize": "8",
        "attention": "",
        "RNN": "",
        "lr": "0.001000",
    },
    'PTUNIFIER_PATHVQA': {
        "epoch": "15",
        "batchsize": "16",
        "attention": "",
        "RNN": "",
        "lr": "0.001000",
    },
    'TCL_VQA-RAD': {
        "epoch": "15",
        "batchsize": "16",
        "attention": "",
        "RNN": "",
        "lr": "0.001000",
    },
    'TCL_Med-VQA': {
        "epoch": "15",
        "batchsize": "16",
        "attention": "",
        "RNN": "",
        "lr": "0.001000",
    },
    'TCL_SLAKE': {
        "epoch": "15",
        "batchsize": "8",
        "attention": "",
        "RNN": "",
        "lr": "0.001000",
    },
    'TCL_PATHVQA': {
        "epoch": "15",
        "batchsize": "4",
        "attention": "",
        "RNN": "",
        "lr": "0.001000",
    },
    'METER_VQA-RAD': {
        "epoch": "15",
        "batchsize": "16",
        "attention": "",
        "RNN": "",
        "lr": "0.001000",
    },
    'METER_Med-VQA': {
        "epoch": "15",
        "batchsize": "16",
        "attention": "",
        "RNN": "",
        "lr": "0.001000",
    },
    'METER_SLAKE': {
        "epoch": "15",
        "batchsize": "8",
        "attention": "",
        "RNN": "",
        "lr": "0.001000",
    },
    'METER_PATHVQA': {
        "epoch": "15",
        "batchsize": "4",
        "attention": "",
        "RNN": "",
        "lr": "0.001000",
    },
    'MEDVINT': {
        "epoch": "",
        "batchsize": "",
        "attention": "",
        "RNN": "",
        "lr": "",
    },
    'MINIGPT-4': {
        "epoch": "15",
        "batchsize": "16",
        "attention": "",
        "RNN": "",
        "lr": "0.001000",
    },
}

routines = {
    "begin": {
        "question": begin_ques,
        "answer": begin_ans
    },
    # 使用函数返回样例（提供调用chatGPT或其他复杂处理的能力）
    "model": {
        "question": model_ques,
        "answer": model_ans,
    },
    # 使用字符串直接返回样例
    "dataset": {
        "question": dataset_ques,
        "answer": dataset_ans,
    },
    # 使用表格返回样例（提供调用chatGPT或其他复杂处理的能力）
    "result": {
        "question": result_ques,
        "answer": result_ans,
    },
    "datatraining": {
        "question": datatraining_ques,
        "answer": datatraining_ans,
    },
}
