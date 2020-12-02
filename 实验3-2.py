# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

#绘制变换函数图像
def linearPlot(x1, x2):
    plt.plot([x1, x2], [0, 255], 'r')
# 第一个方框代表x的各个标记点，第二个方框代表y的各个标记点，第三个显示风格
    plt.plot([x2, x2], [0, 255], '--')
    plt.xlim(0, 255), plt.ylim(0, 255)
    plt.show()

def linearStretch(src):#线性拉伸
    for i in range(1,3):
        src[::i] = np.absolute(
            (src[::i] - np.min(src[::i])) / (np.max(src[::i]) - np.min(src[::i]))) * 255.0
    src = np.uint8(src + 0.5)
    return src

def linearPercentStretch(src, percent):#裁剪百分比拉伸
    src = np.float64(src)
    cut = np.floor(256 * percent + 0.5)  #计算需要裁减的数量；floor()向下取整
    minvalue = np.min(src)
    maxvalue = np.max(src)

    newminvalue = minvalue + cut
    newmaxvalue = maxvalue - cut

    row, col = src.shape
    ans = np.array((row, col), dtype = np.float64)
    ans = 255.0 * (src - newminvalue) / (newmaxvalue - newminvalue)
    ans = ans + 0.5
    ans[ans > 255] = 255
    ans[ans < 0] = 0
    ans = np.uint8(ans)
    return ans

def LDSPlot(x1, y1, x2, y2):
    plt.plot([0, x1, x2, 255], [0, y1, y2, 255], 'r' , label = 'Targetline', linewidth=2.5) # 红色实线，定义label，线宽
    plt.plot([0, 255], [0, 255], 'g' , label = 'Guideline', linewidth=2.5)
    plt.plot([0, x1, x1], [0, 0, x1], 'y--') # 黄色虚线
    plt.plot([0, x2, x2], [0, 0, y2], '--') # 虚线
    plt.xlim(0, 255), plt.ylim(0, 255)
    plt.legend() # 显示图例
    plt.show()

def linearDividedStretch(src, x1, y1, x2, y2):
    #以查找表的方式做分段线性变换
    LDSPlot(x1, y1, x2, y2)
    x1, x2, y1, y2 = np.float(x1), np.float(x2), np.float(y1), np.float(y2)

    x = np.arange(256) # 列数为256的一维数组
    lut = np.zeros(256, dtype = np.float64)  #定义一个空表作为容器，储存每一灰度值对应变换后的值

    #填写内容
    for i in x:
        if(i < x1):
            lut[i] = (y1 * 1.0 / x1) * i
        elif(i < x2):
            lut[i] = (y2 - y1) / (x2 - x1) * (i - x1) + y1
        else:
            lut[i] = (255 - y2) / (255 - x2) * (i - x2) + y2

    lut = np.uint8(lut + 0.5)
    dst = cv2.LUT(src, lut)
    return dst

img = cv2.imread('subset.tif',3 )
cv2.imshow('input', img)
img2 = linearStretch(img)
(B,G,R) = cv2.split(img2)
B = linearPercentStretch(B, 0.05)#裁剪百分比拉伸
G = linearPercentStretch(G, 0.05)
R = linearPercentStretch(R, 0.05)
B = linearDividedStretch(B, 20, 10, 200, 250)#分段线性拉伸
G = linearDividedStretch(G, 20, 10, 200, 250)
R = linearDividedStretch(R, 20, 10, 200, 250)
merged = cv2.merge([R,G,B])#合并R、G、B分量
cv2.imshow("output",merged)
cv2.waitKey(0)
# 计算直方图，比较拉伸前后直方图变化
histb0 = cv2.calcHist([img], [0], None, [256], [0, 255])
histb = cv2.calcHist([img2], [0], None, [256], [0, 255])
hista1 = cv2.calcHist(B, [0], None, [256], [0, 255])
hista2 = cv2.calcHist(G, [0], None, [256], [0, 255])
hista3 = cv2.calcHist(R, [0], None, [256], [0, 255])

# 对第1子图进行设定
plt.subplot(4, 1, 1)
plt.plot(histb, 'y', label = 'Before')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
# 对第2子图进行设定
plt.subplot(4, 1, 2)
plt.plot(hista1, 'c', label = 'GREEN')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
# 对第3子图进行设定
plt.subplot(4, 1, 3)
plt.plot(hista2, 'r', label = 'RED')
plt.xlabel('x')
plt.ylabel('y')
plt.legend() 
# 对第4子图进行设定
plt.subplot(4, 1, 4)
plt.plot(hista3, 'b', label = 'NIR')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()
#cv2.imwrite(r'NJU_output.tif', img2)
cv2.destroyAllWindows() # 删除所有窗口；若要删除特定的窗口，往输入特定的窗口值
print('done!')