import numpy as np
import cv2


def main():
    red_val = 0
    green_val = 114

    # 为视频Feed创建一个窗口
    # cv2.namedWindow('frame',cv2.CV_WINDOW_AUTOSIZE)

    # 使轨迹栏用于HSV屏蔽
    # cv2.createTrackbar('HSV','frame',0,255,getVal)

    # 从网络摄像头捕获视频
    cap = cv2.VideoCapture(0)

    while (True):
        # 逐帧捕获
        ret, frame = cap.read()

        cv2.imshow("original", frame)

        # 命名用于掩码边界的变量
        # j = cv2.getTrackbarPos('HSV','frame')

        # 将BGR转换为HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # d限定在HSV红色的范围
        red_lower = np.array([red_val - 10, 100, 100])
        red_upper = np.array([red_val + 10, 255, 255])

        # 限定在HSV绿色的范围
        green_lower = np.array([green_val - 10, 100, 100])
        green_upper = np.array([green_val + 10, 255, 255])

        # T阈值HSV图像以仅获得所选颜色
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        # Bitwise-AND屏蔽原始图像
        red_res = cv2.bitwise_and(frame, frame, mask=red_mask)
        green_res = cv2.bitwise_and(frame, frame, mask=green_mask)

        # 结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # 形态学结束
        red_closing = cv2.morphologyEx(red_res, cv2.MORPH_CLOSE, kernel)
        green_closing = cv2.morphologyEx(green_res, cv2.MORPH_CLOSE, kernel)

        # 转换为黑白图像
        red_gray = cv2.cvtColor(red_closing, cv2.COLOR_BGR2GRAY)
        green_gray = cv2.cvtColor(green_closing, cv2.COLOR_BGR2GRAY)
        (thresh1, red_bw) = cv2.threshold(red_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        (thresh2, green_bw) = cv2.threshold(green_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 计算像素变化
        red_black = cv2.countNonZero(red_bw)
        if red_black > 20000:
            print("RED")

        green_black = cv2.countNonZero(green_bw)
        if green_black > 18000:
            print("BLUE")

        # 显示结果帧
        both = np.hstack((red_bw, green_bw))
        cv2.imshow('RED_GREEN', both)

        # 按q退出
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

    # 完成所有操作后，释放捕获
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
