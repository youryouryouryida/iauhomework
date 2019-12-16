import cv2
import numpy as np
import matplotlib.pyplot as plt

o_img=cv2.imread('lena.jpg')

img=cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)
img=np.array(img)
h=img.shape[0]
w=img.shape[1]
label=np.zeros((h,w))


print(label.shape)

pix1=np.random.randint(0,90)
pix2=np.random.randint(90,180)
pix3=np.random.randint(180,255)
print(pix1,pix2)


# np.random.randint(0,w)
# init_index=[(np.random.randint(0,w),np.random.randint(0,h)),(np.random.randint(0,w),np.random.randint(0,h)),(np.random.randint(0,w),np.random.randint(0,h))]
# print(init_index)
# point=[]
# for each in init_index:
#     point.append(img[each[1]][each[0]])
#     print(point)
#
for num in range(5):
    sum1=0
    sum2=0
    sum3=0
    num1=0
    num2=0
    num3=0
    print(pix1, pix2,pix3)
    for x in range(w):
        for y in range(h):
            if abs(img[y][x]-pix1)<=abs(img[y][x]-pix2) and abs(img[y][x]-pix1)<=abs(img[y][x]-pix3):
                label[y][x]=255
                sum1=sum1+img[y][x]
                num1=num1+1
            else:
                if abs(img[y][x]-pix2)<=abs(img[y][x]-pix3):
                    label[y][x] =150
                    sum2 = sum2 + img[y][x]
                    num2=num2+1
                else:
                    label[y][x] =0
                    sum3 = sum3 + img[y][x]
                    num3=num3+1

    pix1=round(sum1/num1)
    pix2=round(sum2/num2)
    pix3=round(sum3/num3)


print(pix1,pix2)
plt.figure("label") # 图像窗口名称
plt.imshow(label,'gray')
plt.show()