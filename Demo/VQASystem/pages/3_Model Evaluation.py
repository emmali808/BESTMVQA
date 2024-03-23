import streamlit as st
import pandas as pd
import numpy as np
import model_info
import decimal
import time
import json
import pymysql
import plotly.subplots as psb
import plotly.express as px
import plotly.graph_objects as go
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
st.title("Test Results of User _"+st.session_state['user']+"_")
with st.container():
    all,best=st.tabs(['All Records','Best Performance Record'])
    with all:
        conn = pymysql.connect(
            host=st.secrets["host"],
            user=st.secrets["username"],
            password=st.secrets["password"],
            database=st.secrets["database"]
        )
        cursor=conn.cursor()
        cursor.execute("SELECT * FROM vqa.record WHERE auth_key='%s'"%(st.session_state['user']))
        records=cursor.fetchall()
        columnDes = cursor.description #获取连接对象的描述信息
        columnNames = [columnDes[i][0] for i in range(len(columnDes))] #获取列名
        df = pd.DataFrame([list(i) for i in records],columns=columnNames)
        df=df[['record_name','model','dataset','epoch','batch_size','attention','rnn','lr','open','closed','all','status']]
        df['open'] = df['open'].astype(float)
        df['open'] = df['open'].apply(lambda x:round(x,1))
        df['closed'] = df['closed'].astype(float)
        df['closed'] = df['closed'].apply(lambda x:round(x,1))
        df['all'] = df['all'].astype(float)
        df['all'] = df['all'].apply(lambda x:round(x,1))
        df['lr'] = df['lr'].astype(float)
        df['lr'] = df['lr'].apply(lambda x:round(x,3))
        st.dataframe(df,width=1300)
        st.download_button(
        label="Download data as CSV",
        data=df.to_csv(),
        file_name=st.session_state['user']+'_records.csv',
        mime='text/csv',
        )
    with best:
        st.subheader("Bar Graph Comparison for Best Performance")
        best_sql=(
            "SELECT  t1.* FROM  vqa.record  t1 INNER  JOIN (SELECT  model,  dataset,  MAX(`all`)  AS  max_all FROM vqa.record WHERE AUTH_KEY='%s' AND STATUS = 'complete' GROUP  BY  model,  dataset)  t2 ON  t1.model  =  t2.model  AND  t1.dataset  =  t2.dataset  AND  t1.all  =  t2.max_all;"
            %(st.session_state['user'])
        )
        # st.write(best_sql)
        cursor=conn.cursor()
        cursor.execute(best_sql)
        best_records=cursor.fetchall()  
        # st.write(best_records)
        columnDes = cursor.description #获取连接对象的描述信息
        columnNames = [columnDes[i][0] for i in range(len(columnDes))] #获取列名
        best_df = pd.DataFrame([list(i) for i in best_records],columns=columnNames)
        best_df=best_df[['model','dataset','open','closed','all']]
        
        # 假设数据存储在一个DataFrame对象df中
        grouped = best_df.groupby('model')
        grouped_dict = {key: value for key, value in grouped}
        selected=st.multiselect(
            'Models',
            grouped_dict.keys(),
            # grouped_dict.keys()
            )
        # 存储每个小组的plotly图形到一个列表中
        figs = []
        for name in selected:
            group=grouped_dict[name]
            fig = go.Figure()
            for col in ['open', 'closed', 'all']:
                fig.add_trace(go.Bar(x=group['dataset'], y=group[col], name=col))
            fig.update_layout(title=name, barmode='group', xaxis_tickangle=-45)
            figs.append(fig)
        
        per_row=st.number_input('Graphs Per Row:',1,len(figs),3,1)
        # columns=st.columns(per_row)
        for i in range(len(figs)):
            if i%per_row==0:
                columns=st.columns(per_row)
            with columns[i%per_row]:
                st.plotly_chart(figs[i],use_container_width=True)
        st.subheader("Data Table Comparison for Best Performance")
        best_df['open'] = best_df['open'].astype(float)
        best_df['open'] = best_df['open'].apply(lambda x:round(x,1))
        best_df['closed'] = best_df['closed'].astype(float)
        best_df['closed'] = best_df['closed'].apply(lambda x:round(x,1))
        best_df['all'] = best_df['all'].astype(float)
        best_df['all'] = best_df['all'].apply(lambda x:round(x,1))
        st.dataframe(best_df,use_container_width=True,width=1000)
        st.download_button(
            label="Download data as CSV",
            data=best_df.to_csv(),
            file_name=st.session_state['user']+'_best_records.csv',
            mime='text/csv',
        )