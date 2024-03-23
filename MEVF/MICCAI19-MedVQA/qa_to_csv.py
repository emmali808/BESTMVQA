import json
import pandas as pd

#VQARAD
with open('/home/coder/projects/MEVF/MICCAI19-MedVQA/data_RAD/testset.json','r') as f:
    rad=json.load(f)
print(rad[0])
id=[]
ques=[]
ans=[]
image_id=[]
for i in range(len(rad)):
    # if rad[i]['answer_type']=='CLOSED' and rad[i]['answer'].lower() not in ['yes','no']:
    #     print(i,rad[i]['answer'],rad[i]['answer_type'])
    #     print(rad[i])
    quest=rad[i]['question']
    if rad[i]['answer_type']=='CLOSED' and rad[i]['answer'].lower() in ['yes','no']:
        print(type(rad[i]['question']))
        quest=rad[i]['question']+' '+'Yes/No'
        quest.strip()
        print(quest)
        print(rad[i])
    ques.append(quest)
    ans.append(rad[i]['answer'])
    id.append()

dt=pd.DataFrame({"Question":ques,"Answer":ans})
dt.to_csv("VQA-RAD-TEST.csv",index=False,sep=',')


#MED-VQA
# data=pd.read_csv(r'/home/coder/projects/MMBERT/VQA-Med-2019/ImageClef-2019-VQA-Med-Test/testdf.csv',sep= ',')
# for i in range(len(data)):
#     if data.loc[i,'answer'] == 'yes' or data.loc[i,'answer'] == 'no':
#         data.loc[i,'question']=str(data.loc[i,'question'])+" "+"Yes/No"
#         print(data.loc[i,'question'])
# meddt=pd.DataFrame({"Question":data['question'],"Answer":data['answer']})
# meddt.to_csv("VQA-Med-TEST.csv",index=False,sep=',')

#SLAKE
# with open('/home/coder/projects/Med-VQA/data_SLAKE/testset.json','r') as f:
#     rad=json.load(f)
# print(rad[0])
# ques=[]
# ans=[]
# for i in range(len(rad)):
#     quest=rad[i]['question']
#     if rad[i]['answer_type']=='CLOSED' and rad[i]['answer'].lower() in ['yes','no']:
#         print(type(rad[i]['question']))
#         quest=rad[i]['question']+' '+'Yes/No'
#         quest.strip()
#         print(quest)
#         print(rad[i])
#     ques.append(quest)
#     ans.append(rad[i]['answer'])

# dt=pd.DataFrame({"Question":ques,"Answer":ans})
# dt.to_csv("SLAKE-TEST.csv",index=False,sep=',')

#pathVQA
# with open('/dev/shm/data_PATH/testset.json','r') as f:
#     rad=json.load(f)
# print(rad[0])
# ques=[]
# ans=[]
# for i in range(len(rad)):
#     quest=rad[i]['question']
#     if rad[i]['answer_type']=='yes/no':
#         print(type(rad[i]['question']))
#         quest=rad[i]['question']+' '+'Yes/No'
#         quest.strip()
#         print(quest)
#         print(rad[i])
#     ques.append(quest)
#     ans.append(rad[i]['answer'])

# dt=pd.DataFrame({"Question":ques,"Answer":ans})
# dt.to_csv("PathVQA-TEST.csv",index=False,sep=',')


import pickle

# rb是2进制编码文件，文本文件用r
f = open(r'/home/coder/projects/MEVF/MICCAI19-MedVQA/data_RAD/cache/trainval_label2ans.pkl','rb')
data = pickle.load(f)
print(data)
