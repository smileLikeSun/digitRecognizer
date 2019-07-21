import os
import pandas as pd
import numpy as np
from img_to_vct import img_vect
import operator

def img_labels():
    with open("digit_label.txt", 'r') as fi:
        labels = fi.read().split(',')
        return labels

# 测试图片与训练集的距离
def compare_vect(test_img_vect):
    train_imgs_vect = pd.read_csv("digit_vect.csv", header=None)
    distance_dict = {}
    for i in range(train_imgs_vect.shape[0]):  # train_imgs_vect.shape[0]
        train_img_vect = train_imgs_vect.ix[i].values
        vect_dis = np.sqrt(np.sum(np.square(test_img_vect - train_img_vect)))
        distance_dict[i] = vect_dis
    return distance_dict

def predict_img_num(img_dir, res_label):
    if os.path.isdir(img_dir):
        wrong_count = 0
        digit_imgs = os.listdir(img_dir)
        digit_imgs.sort(key=lambda x: int(x[4:-4]))
        for i in range(len(res_label)):
            test_img_vect = img_vect(img_dir + digit_imgs[i])
            distance_dict = compare_vect(test_img_vect)
            sorted_distance = sorted(distance_dict.items(), key=operator.itemgetter(1))
            labels = img_labels()
            class_count = {}
            for j in range(5):
                class_count[labels[sorted_distance[j][0]]] = class_count.get(labels[sorted_distance[j][0]], 0) + 1
            class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
            print("预测结果为：{} 正确值为：{}".format(class_count[0][0], res_label[i]))
            if(class_count[0][0] != res_label[i]):
                wrong_count += 1
        print("识别正确率为：{}".format((len(res_label) - wrong_count) / len(res_label)))




if __name__ == '__main__':

    # test_img_path = 'testSample/'
    # res_label = "2,0,9,0,3,7,0,3,0,3,5,7,7,0,4,3,3,1,9,0".split(',')
    res_label = "7,3,4".split(',')
    test_img_path = "test/"
    predict_img_num(test_img_path, res_label)

    # a = np.array([[3,3,3,2],[8,8,8,4]])
    # a = np.mat(a)
    # print(type(a))
    # print(a)
    # for i in range(a.shape[0]):
    #     print("a.shape[0] = {}".format(a.shape[0]))
    #     print("a[i] = {}".format(a[i]))
    #     print("type a[i]: {}".format(type(a[i])))
    #     a[i,:] = a[i,:] / np.tile((2), 4)
    # print("***********")
    # print(a)
