from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import nltk
import cv2
import os

# def calculate_images_similarity(image1_path, image2_path):
#     # 读取图像
#     image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
#     image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

#     # 创建SIFT特征提取器
#     sift = cv2.SIFT_create()

#     # 检测并计算图像的特征点和描述符
#     keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
#     keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

#     # 创建FLANN匹配器
#     matcher = cv2.FlannBasedMatcher()

#     # 使用KNN算法进行特征点匹配
#     matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

#     # 根据Lowe's比率进行匹配筛选
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.7 * n.distance:
#             good_matches.append(m)

#     # 计算相似度得分
#     similarity = len(good_matches) / max(len(keypoints1), len(keypoints2))

#     return similarity

# 另外一种图像相似度计算方式
def calculate_images_similarity(image1_path, image2_path):
    # 读取医学图像
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # 初始化ORB特征检测器
    orb = cv2.ORB_create()

    # 寻找关键点和描述符
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # 创建BFMatcher对象
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 使用BFMatcher进行特征匹配
    matches = bf.match(descriptors1, descriptors2)

    # 按照特征点之间的距离进行排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 计算匹配的相似度得分
    similarity_score = len(matches)/200
    return similarity_score

def calculate_texts_similarity(text1, text2):
    # 文本预处理和特征提取
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    # 计算余弦相似度
    cosine_sim_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1]).flatten()[0]
    return cosine_sim_score

def get_text(file_path):
    # 打开txt文件并读取内容
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def similarity_report_retrieval(minigpt_generate_text):
    data_csv_path = '/home/coder/projects/SystemDataset/robot/report_retrieval/case_samples.csv'
    compare_data_root_path = '/home/coder/projects/SystemDataset/robot/report_retrieval'
    image01_path = '/home/coder/projects/SystemDataset/robot/upload.jpg'
    total_similarity_score = []

    # file_path = os.path.join(compare_data_root_path, 'original_text.txt')  
    # text01 = get_text(file_path)
    text01 = minigpt_generate_text
    df = pd.read_csv(data_csv_path)

    for img_path, txt_path in zip(df['image'],df['txt']):
        
        image02_path = os.path.join(compare_data_root_path, 'image', img_path)
        text02 = get_text(os.path.join(compare_data_root_path, 'txt', txt_path))

        # 比较两张图片的相似度
        images_similarity_score = calculate_images_similarity(image01_path, image02_path)
        # print("图片相似度：", similarity_score)

        texts_similarity_score = calculate_texts_similarity(text01, text02)
        # print("文本段之间的余弦相似度：", texts_similarity_score)
        
        # 相似度权重
        k = 0.2
        score_tmp = k*images_similarity_score + (1-k)*texts_similarity_score
        total_similarity_score.append(score_tmp)
        print("#######", image02_path, text02)
        print("######img_path, txt_path",images_similarity_score, texts_similarity_score, score_tmp)

    # 获取电子病历报告最大相似度的索引
    max_idx = max(enumerate(total_similarity_score), key=lambda x: x[1])[0]
    max_img_path = df['image'][max_idx]
    max_txt_path = df['txt'][max_idx]
    max_img_full_path = os.path.join(compare_data_root_path, 'image', max_img_path)
    max_txt_full_path = os.path.join(compare_data_root_path, 'txt', max_txt_path)
    max_txt_content = get_text(max_txt_full_path)
    return max_img_full_path, max_txt_content


# img_path, txt_content = similarity_report_retrieval("The image shows an X - ray of a person's chest. The lungs, heart and ribs can be seen in the image. The bones are white and the lungs are black. The heart is in the center of the chest and is visible as a dark spot.")
# print(img_path, txt_content)