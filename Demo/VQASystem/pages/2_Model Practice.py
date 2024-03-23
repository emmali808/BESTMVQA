import streamlit as st
import pandas as pd
import numpy as np
import model_info
import time
import json
import subprocess
import altair as alt
import pymysql
import plotly.express as px
import plotly.graph_objects as go
conn = pymysql.connect(
                        host=st.secrets["host"],
                        user=st.secrets["username"],
                        password=st.secrets["password"],
                        database=st.secrets["database"]
                    )
st.set_page_config(layout='wide')
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
    
st.title('Model Practice')

models=model_info.models

st.markdown("#### Choose a Model: ")
model_option = st.selectbox(
    'Choose a Model: ',
    models.keys(),label_visibility="collapsed")
model=models[model_option]



with st.container():
    st.subheader('_'+model['title']+'_')
    st.write('Click '+f'<a target="_blank" href="{model["url"]}">_here_</a>'+' to the original paper.',
                    unsafe_allow_html=True)
    pretrain,tuning=st.tabs(['Pre-Train','Fine-Tune'])
    with pretrain:
        if 'pretrain_dataset' in model.keys():
            with st.form(model_option+" pretrain form"):
                pretrain_dataset_option = st.multiselect(
                    'Dataset',
                    model['pretrain_dataset'].keys())
                pretrain_submitted = st.form_submit_button("Pre-Train")
                if pretrain_submitted:
                    for po in pretrain_dataset_option:
                        with st.spinner('Pre-training dataset: %s...'%(po)):
                            subprocess.run(["bash",model['pretrain_path'][po],model['pretrain_dataset'][po]])
                    st.success("success!")
        else:
            st.info('Currently Not Support!')
            
    with tuning:
        if model_option=='MiniGPT-4':
            st.info('Currently Not Support! We offer the fine-tune results in the Evaluation page!')
        else:
            with st.form(model_option+" tuning form"):
                dataset_option = st.selectbox(
                    'Choose a Dataset: ',
                    model['dataset'].keys())
                epoch = st.number_input('Epoch:',1,None,1,1)
                lr = st.number_input('Learning Rate:',0.00001,1.00)
                # batchsize = st.number_input('Batch Size:',1,None,1,1)
                batchsizes = st.multiselect(
                    'Batch Size: ',
                    [1, 2, 4, 8,16,32,64,128,256],
                    [1])
                attention=''
                if 'attention' in model.keys():
                    attention=st.selectbox('Attention: ',model['attention'])
                rnn=''
                if 'RNN' in model.keys():
                    rnn=st.selectbox('RNN: ',model['RNN'])
                record_name=st.text_input('Record Name: ')
                if 'attention' in model.keys():
                    model_name=model_option+'+'+attention
                else:
                    model_name=model_option
                curr,history=st.tabs(['Current','History'])
                with curr:
                    if not record_name:
                        st.info('Fill in the record name, the fine-tune results can be searched by it')
                    submitted = st.form_submit_button("Fine-Tune")
                if submitted:
                    for batchsize in batchsizes:
                        cursor=conn.cursor()
                        cursor.execute("INSERT INTO vqa.record (auth_key,model,record_name,epoch,batch_size,dataset,attention,rnn,lr,status) VALUES ('%s','%s','%s',%d,%d,'%s','%s','%s','%f','%s');"%(st.session_state['user'],model_name,record_name,epoch,batchsize,dataset_option,attention,rnn,lr,"running"))
                        record_id=conn.insert_id()
                        cursor.close()
                        with st.spinner('Traning Record ID: %d,Epoch :%d,Batch Size:%d...'%(record_id,epoch,batchsize)):
                            time.sleep(2)
                            conn.commit()
                            if model_option in['PTUnifier','METER','TCL','MedVInT']:
                                subprocess.run(["bash",model['path'][dataset_option],str(epoch),str(lr),str(batchsize),str(record_id)])
                            else:
                                subprocess.run(["bash",model['path'][dataset_option],str(attention),str(model['dataset'][dataset_option]),str(epoch),str(lr),str(batchsize),str(rnn),str(record_id)])
                    st.success("Success! You can view the details in the History tab!")
                with history:
                    if not record_name:
                        st.info('You can use the record name as a search criterion')
                    search=st.form_submit_button("Search")
                    details=[]
                    if search:
                        with st.spinner("searching"):
                            cursor=conn.cursor()
                            # st.write("SELECT * FROM vqa.record WHERE model='%s' and dataset='%s' and (record_name='%s' or '%s'='') and attention='%s' and rnn='%s' and lr=%f and auth_key='%s'"%(model_name,dataset_option,record_name,record_name,attention,rnn,lr,st.session_state['user']))
                            cursor.execute("SELECT * FROM vqa.record WHERE model='%s' and dataset='%s' and (record_name='%s' or '%s'='') and attention='%s' and rnn='%s' and lr=%f and auth_key='%s'"%(model_name,dataset_option,record_name,record_name,attention,rnn,lr,st.session_state['user']))
                            records=cursor.fetchall()
                            columnDes = cursor.description #获取连接对象的描述信息
                            columnNames = [columnDes[i][0] for i in range(len(columnDes))] #获取列名
                            record_df = pd.DataFrame([list(i) for i in records],columns=columnNames)
                            st.dataframe(record_df[['id','record_name','model','dataset','epoch','batch_size','status']],use_container_width=True)
                            ids=record_df['id']
                            cursor.close()
                            cursor=conn.cursor()
                    
                            for id in ids:
                                cursor.execute("SELECT * FROM vqa.detail WHERE record_id=%d"%(id))
                                ret = cursor.fetchall()
                                for detail in ret:
                                    details.append(detail)
                            # st.write(details)
                        if len(details)>0:
                            columnDes = cursor.description
                            columnNames = [columnDes[i][0] for i in range(len(columnDes))]
                            detail_df = pd.DataFrame([list(i) for i in details],columns=columnNames)
                            epoch_fig = px.line(detail_df, x='epoch', y='loss', color='batch_size', title='epoch-loss', log_y=True,markers=True)
                            epoch_fig.update_layout(legend_title=None, xaxis_title='epoch', yaxis_title='loss')
                            st.plotly_chart(epoch_fig, use_container_width=True)
                            batch_detail_df=detail_df.groupby(['batch_size']).apply(lambda t: t[(t['loss']==t['loss'].min())])
                            batchsize_fig=px.line(batch_detail_df, x='batch_size', y='loss', title='batchsize-loss', log_y=True,markers=True)
                            batchsize_fig.update_layout(legend_title=None, xaxis_title='batch_size', yaxis_title='loss')
                            st.plotly_chart(batchsize_fig, use_container_width=True)
                

            