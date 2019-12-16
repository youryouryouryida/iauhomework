import math
import queue
import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2




def Canny_l(img_path,TL=80,TH=150):
    # 1.读取图片并转为灰度图
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    #2.高斯滤波器
    gaussian = np.zeros([5, 5])
    sigma1=sigma2=1.4
    sum=0
    for i in range(5):
        for j in range(5):
            gaussian[i, j] = math.exp(-1 / 2 * (np.square(i - 2) / np.square(sigma1)  + (np.square(j - 2) / np.square(sigma2)))) / (2 * math.pi * sigma1 * sigma2)
            sum = sum + gaussian[i, j]

    gaussian = gaussian / sum
    W, H = img.shape
    new_gray = np.zeros([W - 5, H - 5])
    for i in range(W - 5):
        for j in range(H - 5):
            new_gray[i, j] = np.sum(img[i:i + 5, j:j + 5] * gaussian)  # 与高斯矩阵卷积实现滤波

    # plt.imshow(new_gray, cmap="gray")
    # plt.show()
    # 3.增强 通过求梯度幅值

    W1, H1 = new_gray.shape
    dx = np.zeros([W1 - 2, H1 - 2])
    dy = np.zeros([W1 - 2, H1 - 2])
    d = np.zeros([W1 - 2, H1 - 2])
    theta=np.zeros([W1 - 2, H1 - 2])
    for i in range(W1 - 2):
        for j in range(H1 - 2):
            x=i+1
            y=j+1
            dx[i, j] = 2*(new_gray[x+1, y] - new_gray[x-1, y])+(new_gray[x+1, y-1] - new_gray[x-1, y-1])+(new_gray[x+1, y + 1] - new_gray[x-1, y+1])
            dy[i, j] = 2*(new_gray[x ,y-1] - new_gray[x, y+1])+(new_gray[x+1, y-1] - new_gray[x+1, y+1])+(new_gray[x-1, y - 1] - new_gray[x-1, y+1])
            d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))  # 图像梯度幅值作为图像强度值
            angle=math.atan2(dy[i,j],dx[i,j])
            angle=math.degrees(angle)%180
            theta[i,j]=angle


    #4.非极大值抑制
    #        梯度方向分为四类 0-45度  45-90度  90-135度  135-180度
    NMS = np.copy(d)
    W2, H2 = d.shape
    NMS[0, :] = NMS[W2 - 1, :] = NMS[:, 0] = NMS[:, H2 - 1] = 0
    for i in range(1, W2 - 1):
        for j in range(1, H2 - 1):
            temp1=temp2=0
            if d[i, j] == 0:
                NMS[i, j] = 0
            else:

                if theta[i,j]==0:
                    temp1=d[i+1,j]
                    temp2=d[i-1,j]
                if theta[i,j]==90:
                    temp1=d[i,j+1]
                    temp2=d[i,j-1]

                if theta[i,j]>0 and theta[i,j]<=45:
                    ang=theta[i,j]
                    temp1=d[i+1,j]+math.tan(math.radians(ang))*(d[i+1,j-1]-d[i+1,j])
                    temp2=d[i-1,j]+math.tan(math.radians(ang))*(d[i-1,j+1]-d[i-1,j])
                if theta[i,j]>45 and theta[i,j]<90:
                    ang=90-theta[i,j]
                    temp1=d[i,j-1]+math.tan(math.radians(ang))*(d[i+1,j-1]-d[i,j-1])
                    temp2=d[i,j+1]+math.tan(math.radians(ang))*(d[i-1,j+1]-d[i,j+1])
                if theta[i,j]>90 and theta[i,j]<=135:
                    ang=theta[i,j]-90
                    temp1=d[i,j-1]+math.tan(math.radians(ang))*(d[i-1,j-1]-d[i,j-1])
                    temp2=d[i,j+1]+math.tan(math.radians(ang))*(d[i+1,j+1]-d[i,j+1])
                if theta[i,j]>135 and theta[i,j]<180:
                    ang=180-theta[i,j]
                    temp1=d[i+1,j]+math.tan(math.radians(ang))*(d[i+1,j+1]-d[i+1,j])
                    temp2=d[i-1,j]+math.tan(math.radians(ang))*(d[i-1,j-1]-d[i-1,j])
            if d[i,j]>=temp1 and d[i,j]>=temp2:
                NMS[i,j]=d[i,j]
            else:
                NMS[i,j]=0
    #5.双阈值检测
    DT = np.zeros([W2, H2])
    # MAX=np.max(NMS)
    TL=TL#np.average(NMS)+np.std(NMS)
    TH=TH
    for i in range(0, W2 - 1):
        for j in range(0, H2 - 1):
            # NMS[i,j]=255*(NMS[i,j]/MAX)
    #        if (NMS[i - 1, j - 1:j + 2]==0).all and (NMS[i + 1, j - 1:j + 2]==0).all and NMS[i - 1, j]==0 and NMS[i + 1, j]==0:
    #            DT[i,j]=0
    #        else:
            if NMS[i,j]<TL:
                DT[i,j]=0
            elif NMS[i,j]>=TH:
                DT[i,j]=1
            else:
                DT[i,j]=-1
    #TH=

    # plt.imshow(NMS, cmap="gray")
    # plt.show()

    #6.滞后边界检测
    #LAB=np.copy(DT)
    LAB=np.zeros([W2 , H2])
    stack=[]
    q=queue.Queue()
    connected=False
    for i in range(0, W2 - 1):
        for j in range(0, H2 - 1):

            if DT[i,j]==-1 and LAB[i,j]==0 :
                LAB[i,j]=1
                stack.append([i,j])
                q.put([i,j])
    #            print(i,j,DT[i,j],LAB[i,j])
                while(stack!=[]):
                    temp_i,temp_j=stack.pop()
                    if DT[temp_i-1,temp_j-1]==-1 and LAB[temp_i-1,temp_j-1]==0:##左上点
                        LAB[temp_i-1,temp_j-1]=1
                        stack.append([temp_i-1,temp_j-1])
                        q.put([temp_i-1,temp_j-1])
                    if DT[temp_i-1,temp_j-1]==1:
                        connected=True

                    if DT[temp_i+1,temp_j-1]==-1 and LAB[temp_i+1,temp_j-1]==0 :##右上点
                        LAB[temp_i+1,temp_j-1]=1
                        stack.append([temp_i+1,temp_j-1])
                        q.put([temp_i+1,temp_j-1])
                    if DT[temp_i+1,temp_j-1]==1:##右上点
                        connected=True


                    if DT[temp_i-1,temp_j+1]==-1 and LAB[temp_i-1,temp_j+1]==0 :##左下点
                        LAB[temp_i-1,temp_j+1]=1
                        stack.append([temp_i-1,temp_j+1])
                        q.put([temp_i-1,temp_j+1])
                    if DT[temp_i-1,temp_j+1]==1:
                        connected=True


                    if DT[temp_i+1,temp_j+1]==-1 and LAB[temp_i+1,temp_j+1]==0 :##右下点
                        LAB[temp_i+1,temp_j+1]=1
                        stack.append([temp_i+1,temp_j+1])
                        q.put([temp_i+1,temp_j+1])
                    if DT[temp_i+1,temp_j+1]==1:
                        connected=True

                    if DT[temp_i-1,temp_j]==-1 and LAB[temp_i-1,temp_j]==0:##左点
                        LAB[temp_i-1,temp_j]=1
                        stack.append([temp_i-1,temp_j])
                        q.put([temp_i-1,temp_j])
                    if DT[temp_i-1,temp_j]==1:
                        connected=True

                    if DT[temp_i+1,temp_j]==-1 and LAB[temp_i+1,temp_j]==0:##右点
                        LAB[temp_i+1,temp_j]=1
                        stack.append([temp_i+1,temp_j])
                        q.put([temp_i+1,temp_j])
                    if DT[temp_i+1,temp_j]==1:
                        connected=True

                    if DT[temp_i,temp_j-1]==-1 and LAB[temp_i,temp_j-1]==0:##上点
                        LAB[temp_i,temp_j-1]=1
                        stack.append([temp_i,temp_j-1])
                        q.put([temp_i,temp_j-1])
                    if DT[temp_i,temp_j-1]==1:
                        connected=True

                    if DT[temp_i,temp_j+1]==-1 and LAB[temp_i,temp_j+1]==0:##下点
                        LAB[temp_i,temp_j+1]=1
                        stack.append([temp_i,temp_j+1])
                        q.put([temp_i,temp_j+1])
                    if DT[temp_i,temp_j+1]==1:
                        connected=True
                if connected==False:
                    while(q.empty()==False):
                        q_i,q_j=q.get()
                        DT[q_i,q_j]=0
                elif connected==True:
                    while(q.empty()==False):
                        q_i,q_j=q.get()
                        DT[q_i, q_j] = 1
                        # DT[q_i,q_j]=1
                connected==False
                q.queue.clear()
            else:
                continue
    return DT

DT=Canny_l('soccer.jpg',)
plt.imshow(DT, cmap="gray")
plt.show()            
                








'''
# step4. 双阈值算法检测、连接边缘
W3, H3 = NMS.shape
DT = np.zeros([W3, H3])
# 定义高低阈值
#TL = 0.8 * np.mean(NMS)
#TH = 1 * np.mean(NMS)+np.std(NMS)
TL = 0.5 * np.max(NMS)
TH = 0.99 * np.max(NMS)
# 这里用于滞后边界跟踪  
for i in range(1, W3 - 1):
    for j in range(1, H3 - 1):
        if (NMS[i, j] > TH):
            DT[i, j] = 1
        else:
            DT[i,j]=0


        # elif (NMS[i, j] > TH):
        #     DT[i, j] = 1
        # elif ((NMS[i - 1, j - 1:j + 1] < TH).any() or (NMS[i + 1, j - 1:j + 1]).any()
        #       or (NMS[i, [j - 1, j + 1]] < TH).any()):
        #     DT[i, j] = 1
plt.imshow(DT, cmap="gray")
plt.show()
# plt.imshow(DT, cmap="gray")
# input()


#plt.imshow(img, cmap="gray")
'''