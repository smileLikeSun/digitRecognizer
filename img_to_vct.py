from PIL import Image
import numpy as np
import csv
import os


def img_vect(img_path):
    img = Image.open(img_path)

    img = img.convert("L")

    data = img.getdata()

    data = np.matrix(data)

    data = np.reshape(data, (28, 28))
    for i in range(28):
        data[i, :] = data[i, :] / np.tile((255), 28)

    # 多维向量转换为一维向量
    returnVect = np.zeros((1, 784))
    for i in range(28):
        for j in range(28):
            returnVect[0, 28 * i:28 * (i + 1)] = data[i, :]
    return returnVect

def write_vect_label(img_path, label):
    returnVect = img_vect(img_path)
    digit_csv_path = "digit_vect.csv"
    with open(digit_csv_path, 'a', newline='') as fi:
        csv_write = csv.writer(fi)
        csv_write.writerow(returnVect[0, :])
    digit_label_path = "digit_label.txt"
    with open(digit_label_path, 'a', encoding='utf-8') as fi:
        fi.write(label + ',')
    global count
    count = count + 1
    if count % 5 == 0:
        print("第{}张图片读写完成！".format(count))

def read_img_path(img_dir, num_img):
    if os.path.isdir(img_dir):
        digit_imgs = os.listdir(img_dir)
        digit_imgs.sort(key=lambda x: int(x[4:-4]))
        for i in range(num_img):
            write_vect_label(img_dir + digit_imgs[i], labels[i])

labels = "1,0,1,4,0,0,7,3,5,3,8,9,1,3,3,1,2,0,7,5,8,6,2,0,2,3,6,9,9,7,8,9,4,9,2," \
           "1,3,1,1,4,9,1,4,4,2,6,3,7,7,4,7,5,1,9,0,2,2,3,9,1,1,1,5,0,6,3,4,8,1,0," \
           "3,9,6,2,6,4,7,1,4,1,5,4,8,9,2,9,9,8,9,4,3,6,4,6,2,9,1,2,0,5,"
labels = labels.split(',')
count = 0

if __name__ == '__main__':
    img_dir = "/trainingSample"
    num_img = 100
    read_img_path(img_dir, num_img)

