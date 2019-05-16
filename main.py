import numpy as np
import cv2

def main():
    red_val = 0
    blue_val = 114

    # 从网络摄像头捕获视频
    cap = cv2.VideoCapture(0)

    while (True):
        # 逐帧捕获
        ret, frame = cap.read()

        cv2.imshow("original", frame)
        # 将BGR转换为HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 限定在HSV绿色的范围
        blue_lower = np.array([blue_val - 10, 100, 100])
        blue_upper = np.array([blue_val + 10, 255, 255])

        # T阈值HSV图像以仅获得所选颜色
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        # Bitwise-AND屏蔽原始图像
        blue_res = cv2.bitwise_and(frame, frame, mask=blue_mask)

        # 结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # 形态学结束
        blue_closing = cv2.morphologyEx(blue_res, cv2.MORPH_CLOSE, kernel)

        # 转换为黑白图像
        green_gray = cv2.cvtColor(blue_closing, cv2.COLOR_BGR2GRAY)
        (thresh2, blue_bw) = cv2.threshold(green_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 计算像素变化
        blue_black = cv2.countNonZero(blue_bw)
        if blue_black > 18000:
            print("BLUE")

        # 显示结果帧
        both = np.hstack(blue_bw)
        cv2.imshow('RED_GREEN', both)

        # 按q退出
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

    # 完成所有操作后，释放捕获
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
