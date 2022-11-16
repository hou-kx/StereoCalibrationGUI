import cv2

path_BM_left = 'ceshi/left/'
path_BM_right = 'ceshi/right/'
count = 0

cv2.moveWindow('left', 0, 0)
cv2.moveWindow('right', 640, 0)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    _, frame = cap.read()
    frame1 = frame[0:480, 0:640]

    _, frame = cap.read()
    frame2 = frame[0:480, 640:1280]

    # imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    imgL = frame1
    imgR = frame2

    cv2.imshow('left', frame1)
    cv2.imshow('right', frame2)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(path_BM_left+'left'+str(count)+'.jpg', imgL)
        cv2.imwrite(path_BM_right+'right'+str(count)+'.jpg', imgR)
        count += 1
        pass
cap.release()
cv2.destroyAllWindows()
