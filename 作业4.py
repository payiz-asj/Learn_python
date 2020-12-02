import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_hist(src, name):
    hist_1 = cv2.calcHist([src], [0], None, [256], [0, 255])
    plt.plot(hist_1, 'k', label=name)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(name + ' histogram')
    plt.tight_layout()
    # plt.show()
    return


# 统计积累直方图
def cumuFre(src):
    row = src.shape[0]
    col = src.shape[1]
    hist = np.zeros(256, dtype=np.float32)
    cumuHist = np.zeros(256, dtype=np.float32)
    for i in range(row):
        for j in range(col):
            index = src[i, j]
            hist[index] += 1
    cumuHist[0] = hist[0]
    for i in range(1, 256):
        cumuHist[i] = cumuHist[i - 1] + hist[i]
    cumuHist = cumuHist / np.float32(row * col)
    return cumuHist


# 直方图匹配
def histMatching(oriImg, tarImg):
    oriCumHist = cumuFre(oriImg)  #
    tarCumHist = cumuFre(tarImg)  #
    lut = np.ones(256, dtype=np.uint8) * (256 - 1)  # 新查找表
    start = 0
    for i in range(256 - 1):
        temp = (tarCumHist[i + 1] - tarCumHist[i]) / 2.0 + tarCumHist[i]
        for j in range(start, 256):
            if oriCumHist[j] <= temp:
                lut[j] = i
            else:
                start = j
                break

    dst = cv2.LUT(oriImg, lut)
    return dst


if __name__ == '__main__':
    # 读取目标图片
    img1 = cv2.imread('1.jpg', cv2.IMREAD_UNCHANGED)  # 读入完整图，全通道
    # 读取参考图片
    img2 = cv2.imread('2.jpg', cv2.IMREAD_UNCHANGED)  # 读入完整图，全通道
    # 两张图片合并
    img3 = cv2.resize(img1, None, fx=0.5, fy=0.5)
    cv2.putText(img3, text='Target_Img', org=(50, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(0, 255, 0),
                thickness=2, lineType=cv2.LINE_AA)
    img4 = cv2.resize(img2, None, fx=0.5, fy=0.5)
    cv2.putText(img4, text='Original_Reference', org=(50, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2,
                color=(0, 255, 0),
                thickness=2, lineType=cv2.LINE_AA)
    img5 = np.hstack([img3, img4])
    cv2.namedWindow('compare', 0)
    cv2.imshow('compare', img5)
    cv2.waitKey(0)

    # 规定化
    print('正在规定化，别急...')
    outImg = histMatching(img1, img2)
    print('处理完毕，请看结果!')
    # 显示目标（要改的图片）、原始（参考图）、规定化后的三个直方图
    plt.subplot(3, 1, 1)
    show_hist(img1, 'Target')
    plt.subplot(3, 1, 2)
    show_hist(img2, 'Reference')
    plt.subplot(3, 1, 3)
    show_hist(outImg, 'out')
    plt.show()

    # 显示规定化后的图片
    cv2.namedWindow('out', 0)
    cv2.imshow('out', outImg)
    cv2.waitKey(0)
