import os
import pandas as pd
import numpy as np
from img_to_vct import img_vect
import operator

# 测试图片与训练集的距离
def compare_vect(test_img_vect):
    train_imgs_vect = pd.read_csv("train_img_csv.csv", header=None)
    distance_dict = {}
    for i in range(train_imgs_vect.shape[0]):  # train_imgs_vect.shape[0]
        train_img_vect = train_imgs_vect.ix[i].values
        vect_dis = np.sqrt(np.sum(np.square(test_img_vect - train_img_vect)))
        distance_dict[i] = vect_dis
    return distance_dict

def predict_img_num():

    test_img_vect = pd.read_csv('train_img_csv.csv', header=None)
    for i in range(2):  # test_img_vect.shape[0]
        # vect_distance = compare_vect(test_img_vect)
        print('')

    print("第 {} 个预测结果为：{}")



if __name__ == '__main__':

    # test_img_path = 'F:/python/TensorFlow/mnistasjpg/tensor-data-set/'
    predict_img_num()
