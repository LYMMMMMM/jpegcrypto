#coding:utf-8
import cv2
import numpy as np
import math


def encrypt(input):
    img=input
    for i in range (1,4):
        img = transform(img,i)
    return img

def transform(img, num):

    [rows,cols] = img.shape
    if (rows == cols):
        n = rows
        img2 = np.zeros([rows, cols])

        for x in range(0, rows):
            for y in range(0, cols):
                #a=2,b=2
                img2[x][y] = img[(x+2*y)%n][(2*x+5*y)%n]
        
        return img2

    else:
        if(rows > cols):
            H=rows
        else:
            H=cols
        
        padded_img = np.zeros([H,H])
        
        padded_img[0:rows,0:cols] = img[0:rows,0:cols]
        
        img3=transform(padded_img,num)
        
        return img3

def imgreshape(details):
    #从标记中获取整数：h和w是图像的高度和宽度（前两项）
    h = int(''.join(filter(str.isdigit, details[0])))
    w = int(''.join(filter(str.isdigit, details[1])))

    #声明一个零数组（它有助于重建更大的数组，DCT和所有必须应用的数组）
    array = np.zeros(h*w).astype(int)

    #一些循环变量的初始化
    k = 0
    i = 2
    x = 0
    j = 0

    #这个循环为我们提供了重建的图像大小数组

    while k < array.shape[0]:
        #图像的结尾
        if(details[i] == ';'):
            break
        #请注意，要在数组中获取负数，请检查字符串中的“-”字符
        if "-" not in details[i]:
            array[k] = int(''.join(filter(str.isdigit, details[i])))        
        else:
            array[k] = -1*int(''.join(filter(str.isdigit, details[i])))        

        if(i+3 < len(details)):
            j = int(''.join(filter(str.isdigit, details[i+3])))

        if j == 0:
            k = k + 1
        else:                
            k = k + j + 1        

        i = i + 2

    array = np.reshape(array,(h,w))
    return array

#-----------读取图片--------------------------

block_size = 8
# 取image.text将其解码为图像
with open('imageR.txt', 'r') as myfile:
    imageR=myfile.read()
with open('imageB.txt', 'r') as myfile:
    imageB=myfile.read()
with open('imageG.txt', 'r') as myfile:
    imageG=myfile.read()

# 分成由空格字符分隔的标记
Rdetails = imageR.split()
Bdetails = imageB.split()
Gdetails = imageG.split()

# 将image.txt文件转为图片image
imageR=imgreshape(Rdetails)
imageB=imgreshape(Bdetails)
imageG=imgreshape(Gdetails)

#加密编码后的图片
R=encrypt(imageR)
B=encrypt(imageB)
G=encrypt(imageG)

IMG=cv2.merge([R,B,G])

cv2.imwrite('ENCRYPT.jpg', np.uint8(IMG))