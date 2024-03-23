import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(layout='wide')
# st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)
if 'user' not in st.session_state:
    st.warning("Login First!")
    user=st.text_input('Auth Key: ')
    st.write('Create your auth-key if you have no one . You can use this auth-key to find training records in this system')
    if not user:  
        st.stop()
    st.session_state['user']=user
    st.success(st.session_state['user']+',welcome!')
    time.sleep(1)
    st.experimental_rerun()
    
st.title('ASMVQA: An Assistant System for Medical Visual Question Answering')
st.markdown('##### &ensp;&ensp; **ASMVQA** is an assistant system for Medical Visual Question Answering.The system architecture is shown below:')
st.image("/home/coder/projects/SystemDataset/fig1.png",caption="The Architecture of ASMVQA")
st.markdown('##### &ensp;&ensp; From the frontend, we designed the system into four pages: Data Generation, Model Practice, Model Evaluation and Robot Assistant.')
st.markdown('##### **Data Preparation Page:**')
st.markdown('##### &ensp;&ensp;(1) Provides users with publicly available high-quality VQA datasets;')
st.markdown('##### &ensp;&ensp;(2) Helps users to build customized datasets through auxiliary labeling tools.')
st.markdown('##### **Model Practice Page:**')
st.markdown('##### &ensp;&ensp;(1) Integrates the latest state-of-the-art models from medical domain, general domain(including a multimodal large language model);')
st.markdown('##### &ensp;&ensp;(2) Accessible for users to pre-train, fine-tune and test the model in the model repository by simple configuration.')
st.markdown('##### **Model Evaluation Page:**')
st.markdown('##### &ensp;&ensp;(1) Comprehensively compares user-selected models by tables and bar graphs;')
st.markdown('##### &ensp;&ensp;(2) Records the optimal performance of each model.')
st.markdown('##### **Robot Assistant Page:**')
st.markdown('##### &ensp;&ensp;(1) Accessible for users to conduct data testing;')
st.markdown('##### &ensp;&ensp;(2) Indicates users to configure the hyperparameter in \"One Guide Stop\" form.')
st.markdown('##### &ensp;&ensp; The demonstration video of our system can be found at '+f'<a target="_blank" href="https://github.com/shidaihxj/shared">https://github.com/emmali808/ASVQAS</a>'+ '.',unsafe_allow_html=True)
