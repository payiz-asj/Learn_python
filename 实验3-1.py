
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 

# 幂变换(伽马变换)
def fun(src, c, gamma):
    src = np.float32(src)
    dst = c * np.power(src / 255, gamma) * 255
    dst = np.uint8(dst + 0.5)
    return dst


if __name__ == '__main__':
    # 读取图片
    img = cv2.imread('city.tif', cv2.IMREAD_GRAYSCALE)
    c = float(input("请输入c值："))
    gamma = float(input("请输入伽马值："))
    img2 = fun(img, c, gamma)

    # 两张图片合并一起显示
    img3 = cv2.resize(img, None, fx=0.5, fy=0.5) # 缩小一半，方便显示
    img4 = cv2.resize(img2, None, fx=0.5, fy=0.5) # 缩小一半，方便显示
    img5 = np.hstack([img3,img4]) # 横向合并

    # 输出，左边为原图，右边为处理后的图
    cv2.namedWindow('compare', 0)
    cv2.imshow('compare', img5)



    # 计算直方图
    histb = cv2.calcHist([img], [0], None, [256], [0, 255])
    hista = cv2.calcHist([img2], [0], None, [256], [0, 255])

    # 对第1子图进行设定
    plt.subplot(2, 1, 1)
    plt.plot(histb, 'y', label='Before')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    # 对第2子图进行设定
    plt.subplot(2, 1, 2)
    plt.plot(hista, 'c', label='After')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
