import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_colors(img):
    # Min-max values for HSV-colors
    _GREEN_HSV = [
        np.array([75, 50, 50], np.uint8),
        np.array([120, 255, 255], np.uint8)
    ]
    _YELLOW_HSV = [
        np.array([0, 100, 100]),
        np.array([75, 255, 255])
    ]
    _Y_FRONT_HSV = [
        np.array([0, 0, 200]),
        np.array([255, 99, 255])
    ]

    input_img = cv2.imread(img, cv2.IMREAD_COLOR)

    hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv_img, *_GREEN_HSV)
    yellow = cv2.inRange(hsv_img, *_YELLOW_HSV)
    front = cv2.inRange(hsv_img, *_Y_FRONT_HSV)
    front = cv2.bitwise_and(front, front, mask=cv2.bitwise_not(yellow))
    nothing = cv2.bitwise_or(yellow, green, mask=front)

    return {
        'green': green,
        'yellow': yellow,
        'front': front,
        'nothing': nothing
    }


if __name__ == '__main__':

    obj = detect_colors('./dataset/100/1.jpg')

    colors = {}

    for color, value in obj.items():
        colors.update({color: cv2.countNonZero(value)})

    ALL = 0
    for v in colors.values():
        ALL += v

    for key, color in colors.items():
        colors.update({key: round(color / ALL, 4)})
        print(f'{key}: {round(color, 4)}')

    plt.bar(
        range(obj.__len__()),
        colors.values(),
        align='center'
    )
    plt.xticks(
        range(obj.__len__()),
        colors.values()
    )

    plt.show()
