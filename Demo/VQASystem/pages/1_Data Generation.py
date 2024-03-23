import time
import streamlit as st
import pandas as pd
import zipfile
import os
import hydralit_components as hc
from mysql_connection import connect
import sys
sys.path.append("/home/coder/projects/Demo/VQASystem")
from ner import ner
conn = connect()
hc.hydralit_experimental(True)
st.set_page_config(layout="wide")

if 'user' not in st.session_state:
    st.warning("Login First!")
    user=st.text_input('Auth Key: ')
    st.write('Create your auth-key if you have none. You can use this auth-key to find training records in this system.')
    if not user:  
        st.stop()
    st.session_state['user']=user
    st.success(st.session_state['user']+',welcome!')
    time.sleep(1)
    st.experimental_rerun()

st.title("Data Generation")
upload, label = st.tabs(["Data Upload", "Data Labeling"])

with upload:
    st.subheader("My Datasets")
    container = st.empty()
    cursor=conn.cursor()
    cursor.execute("SELECT * FROM vqa.dataset")
    records=cursor.fetchall()
    columnDes = cursor.description # 获取连接对象的描述信息
    columnNames = [columnDes[i][0] for i in range(len(columnDes))] # 获取列名
    dataset_df = pd.DataFrame([list(i) for i in records],columns=columnNames)
    print(list(dataset_df['name']))

    data_df = pd.DataFrame(
        {
            "Dataset": dataset_df['name'],
            "Available": dataset_df['available'],
            "Download": dataset_df['download'],
        }
    )
    st.subheader("Upload Your Dataset")
    st.markdown('###### Choose a zip file that contains:')
    st.markdown('###### &ensp;&ensp;a) A folder contains Images;')
    st.markdown('###### &ensp;&ensp;b) A folder contains Medical Case Text;')
    st.markdown('###### &ensp;&ensp;c) A csv file that show the relationship between Image and Text.')
    uploaded_file = st.file_uploader(
        "Choose a zip file that contains: "
        "  a) A folder contains Images. "
        "  b) A folder contains Medical Case Text."
        "  c) A csv file that show the relationship between Image and Text.",label_visibility="collapsed" )
    if uploaded_file is not None:
        global zip_file
        zip_file = zipfile.ZipFile(uploaded_file)
        st.write("Your zip file structure:")
        st.write(zip_file.namelist())
        if not zip_file.filelist[len(zip_file.filelist) - 1].filename.endswith(".csv"):
            st.error("Lack csv file!")
        if not zip_file.filelist[0].filename.endswith("image/"):
            st.error("Lack “image” folder!")
        if 'txt/' not in zip_file.namelist():
            st.error("Lack “txt” folder!")
        dataset_name=str(zip_file.filename).split(".")[0]
        new_dataset_path="/home/coder/projects/SystemDataset/" + dataset_name
        
        os.makedirs(new_dataset_path,exist_ok=True)
        zip_file.extractall(new_dataset_path)

        cursor.execute("INSERT INTO vqa.dataset (name,available,download,support) VALUES ('%s',%d,'%s',%d);"%(dataset_name,0,"https://streamlit.io",0))
        conn.commit()

        cursor=conn.cursor()
        cursor.execute("SELECT * FROM vqa.dataset")
        records=cursor.fetchall()
        dataset_df = pd.DataFrame([list(i) for i in records],columns=columnNames)
        data_df = pd.DataFrame(
            {
                "Dataset": dataset_df['name'],
                "Available": dataset_df['available'],
                "Download": dataset_df['download'],
            }
        )
        f=zip_file.open(zip_file.filelist[len(zip_file.filelist) - 1].filename)
        vqa_df=pd.read_csv(f)
        st.write("VQA pairs:")
        st.dataframe(vqa_df, width=1300)
        vqa_list=list(vqa_df.values)
        for vqa in vqa_list:
            vqa[0]=vqa[0].replace("'", "''")
            vqa[1]=vqa[1].replace("'", "''")                
            cursor.execute("INSERT INTO vqa.label (`image`,txt,`dataset`) VALUES ('%s','%s','%s');" %(vqa[0],vqa[1],dataset_name))

        conn.commit()
        cursor.close()
        st.toast("Success! Please label the dataset you upload on the next page!",icon='🎉')
        st.success("Success! Please go to the Data Labeling page to label the dataset!")

def generateVQA(io,ip,it,idir,id,img,txt,name):
    q1 = "Is this a ct image or xray image?"
    if(it=="CT"):
        a1='CT'
    else:
        a1='X-Ray'
    answer_type = 'CLOSE'
    question_type = 'Modality'
    phrase_type = 'fixed'
    cursor=conn.cursor()
    cursor.execute("INSERT INTO vqa.qa (`dataset`,question_type,answer_type,phrase_type,image,question,answer,txt) VALUES ('%s','%s','%s','%s','%s','%s','%s','%s');" %(name,question_type,answer_type,phrase_type,img,q1,a1,txt))
    conn.commit()
    return q1,a1
    

def generateDISnCHM(dic):
    QAlist = []
    for x in dic['DISEASE']:
        for y in dic['CHEMICAL']:
            q = 'I have ' + x + ', what treatment is available?'
            a = y
            QAlist.append([q,a])
    return QAlist

def generateQA(file_data,img,txt,name):
    entities = ner(file_data)
    entDic={'DISEASE':[],'CHEMICAL':[]}
    for entity in entities:
        if entity[1]=='DISEASE':
            entDic['DISEASE'].append(entity[0])
        elif entity[1]=='CHEMICAL':
            entDic['CHEMICAL'].append(entity[0])
    answer_type = 'OPEN'
    question_type = 'Disease'
    phrase_type = 'unfixed'
    QAlist = generateDISnCHM(entDic)
    for QA in QAlist:
        cursor.execute("INSERT INTO vqa.qa (`dataset`,question_type,answer_type,phrase_type,image,question,answer,txt) VALUES ('%s','%s','%s','%s','%s','%s','%s','%s');" %(name,question_type,answer_type,phrase_type,img,QA[0],QA[1],txt))
    return QAlist[0][0],QAlist[0][1]
global is_avail
with label:
    is_avail=False
    cursor=conn.cursor()
    cursor.execute("select name from vqa.dataset where support=%d" % (0))
    non_support_dataset=cursor.fetchall()
    non_support_dataset_df=pd.DataFrame(non_support_dataset,columns=["name"])
    print(non_support_dataset_df)

    option = st.selectbox(
        'Select the dataset you want to label:', non_support_dataset_df['name'])


    cursor.execute("select * from vqa.label where dataset='%s'" % (option))
    unlabeled_dataset=cursor.fetchall()
    columnDes = cursor.description # 获取连接对象的描述信息
    columnNames = [columnDes[i][0] for i in range(len(columnDes))] # 获取列名
    unlabeled_dataset_df=pd.DataFrame(unlabeled_dataset,columns=columnNames)
    print(unlabeled_dataset_df)
    for idx, data in unlabeled_dataset_df.iterrows():
        with st.expander("Medical Case  "+str(idx)):
            is_avail=True
            form = st.form("example"+str(idx))
            txt_path = "/home/coder/projects/SystemDataset/" + option + "/txt/" + data['txt']
            with open(txt_path) as file:
                file_data=file.read()
            with open(txt_path) as file:
                for item in file:
                    form.write(item)
            image_path = "/home/coder/projects/SystemDataset/" + option + "/image/" + data['image']
            form.image(image_path,caption="Image "+str(idx),width=350)
            image_organ = form.selectbox('Label the image organ:', ('CHEST', 'HEAD', 'HAND','LEG'),key=option + "v" + str(idx))
            image_plane = form.selectbox('Label the image_plane:', ('Axial', 'Coronal', 'Sagittal'),key=option + "q" + str(idx))
            image_type = form.selectbox('Label the image_type:', ('CT', 'X-rays'),key=option + "a" + str(idx))
            image_direction = form.selectbox('Label the image_direction:', ('A-P', 'Lateral', 'Lordotic', 'A-P Supine','P-A'),key=option + "d" + str(idx))
            cursor=conn.cursor()
            submitted = form.form_submit_button("Generate QA Pairs")
            if submitted:
                sql="UPDATE vqa.label SET image_organ='%s',image_plane='%s',image_type='%s',image_direction='%s' where id=%s" % (image_organ,image_plane,image_type,image_direction,data['id'])
                cursor.execute(sql)
                conn.commit()
                with st.spinner("generating..."):
                    q1,a1=generateVQA(io=image_organ,ip=image_plane,it=image_type,idir=image_direction,id=data['id'],img=data['image'],txt=data['txt'],name=option)
                    q2,a2=generateQA(file_data,data['image'],data['txt'],option)
                    qa_data = {'question':[q1,q2],'answer':[a1,a2]}
                    qa_df = pd.DataFrame(qa_data)
                    st.dataframe(qa_df,width=1300)
                    msg=st.toast("Generate successfully for case "+str(idx)+" !",icon='🎉')
        if is_avail:
            cursor.execute("UPDATE vqa.dataset SET available=1 where name='%s'" % (option))
        conn.commit()
    # 定义modal
    modal_code = """
    <div>
    <!-- Button trigger modal -->
    <button type="button" class="btn btn-primary" data-toggle="modal" data-target="#exampleModal">
    Submit
    </button>

    <!-- Modal -->
    <div class="modal fade" id="exampleModal" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel" aria-hidden="true">
    <div class="modal-dialog" role="document">
    <div class="modal-content">
    <div class="modal-header">
      <h5 class="modal-title" id="exampleModalLabel">Whether to submit all the changes?</h5>
      <button type="button" class="close" data-dismiss="modal" aria-label="Close">
        <span aria-hidden="true">&times;</span>
      </button>
    </div>
    <div class="modal-body">
      <div class="container">
    <h6>Click the Submit Button to conduct model practice, else to relabel the dataset!</h6>
    </div>
    </div>
    <div class="modal-footer">
    <form class="form-horizontal" action="/Model_Practice">
          <button type="submit" class="btn btn-default">Submit</button>
          <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
    </form>    
    </div>
    </div>
    </div>
    </div>
    </div>
    """
    # if option is not None:
    st.markdown(modal_code, unsafe_allow_html=True)


    # 更新数据
    while True:
        # 添加所有数据到容器中
        container.dataframe(
            data=data_df, width=1300,
            column_config={
                "Available": st.column_config.CheckboxColumn(
                    help="Please label the dataset if it is not available!"
                ),
                "Download": st.column_config.LinkColumn(
                    help="To download the dataset"
                )
            }
        )
        time.sleep(1)