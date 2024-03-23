import pandas as pd
import streamlit as st
#import Demo.VQASystem.routine as routine
import routine
import numpy as np
from PIL import Image
import time

st.title("Chatbot")

def clear_history():
    del st.session_state["history"]
    del st.session_state["routine_state"]


def init():
    if 'user' not in st.session_state:
        st.session_state['user']="nyy"
    if "routine_state" not in st.session_state:
        st.session_state["routine_state"] = "begin"
    if "params" not in st.session_state:
        st.session_state["params"] = {}
    if "model_index" not in st.session_state:
        st.session_state["model_index"]=0
    print("load history")
    if "history" in st.session_state:
        for message in st.session_state["history"]:
            show(message)
    else:
        st.session_state["history"] = []


def show(message):
    with st.chat_message(message["role"]):
        for msg in message["content"]:
            time.sleep(0.01)
            msg["show_type"](msg["data"])


def user_answer(content=""):
    user_message = {
        "role": "user",
        "content": [
            {"data": content, "show_type": st.write},
        ],
    }
    show(user_message)
    st.session_state["history"].append(user_message)


def assistant_answer(user_content=""):
    result = curr_routine["answer"](user_content)
    result_message = {"role": "assistant", "content": result["content"]}
    show(result_message)
    st.session_state["history"].append(result_message)
    st.session_state["routine_state"] = result["next"]

def assistant_answer_pic(uploaded_picture,user_content=""):
    result = curr_routine["answer"](uploaded_picture,user_content)
    content=[]
    content.append(result["content"][0])
    result_message = {"role": "assistant", "content": content}
    show(result_message)
    st.session_state["history"].append(result_message)

    result["content"].remove(result["content"][0])
    ret_message={"role": "assistant", "content": result["content"]}
    show(ret_message)
    st.session_state["history"].append(ret_message)
    st.session_state["routine_state"] = result["next"]


def image(image):
    st.image(image,channels="RGB")


init()
curr_state = st.session_state["routine_state"]
print("state " + str(curr_state))
curr_routine = routine.routines[st.session_state["routine_state"]]
question_message = {
    "role": "assistant",
    "content": [
        {"data": curr_routine["question"](), "show_type": st.write},
    ],
}
show(question_message)


if st.session_state["routine_state"]=="datatraining":
    uploaded_picture = st.file_uploader("Choose a picture")
    if uploaded_picture is not None:
        img_path = '/home/coder/projects/SystemDataset/robot/upload.jpg'
        img = Image.open(uploaded_picture)
        # get original size
        img_width, img_height = img.size
        # decrease the image's size
        img = img.resize((int(img_width/4), int(img_height/4)))
        image_ = np.array(img)
        img.save(img_path)
        message = {
            "role": "user",
            "content": [
                {"data": image_, "show_type": image},
            ],
        }
        show(message)
        if content := st.chat_input():
            st.session_state["history"].append(question_message)
            st.session_state["history"].append(message)
            user_answer(content)
            # assistant_answer_pic(uploaded_picture, content)
            assistant_answer_pic(img_path, content)
            st.experimental_rerun()
else:
    if content := st.chat_input():
        st.session_state["history"].append(question_message)
        user_answer(content)
        assistant_answer(content)
        st.experimental_rerun()
