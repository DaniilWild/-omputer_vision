import cv2 as cv

img1 = cv.imread('/home/strokovrg/Документы/ComputerVision/HW_2/Pictures/shrek.jpg',cv.IMREAD_GRAYSCALE)
img2 = cv.imread('/home/strokovrg/Документы/ComputerVision/HW_2/Pictures/shrek90.jpeg',cv.IMREAD_GRAYSCALE)

orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

img1 = cv.drawKeypoints(img1, kp1, None, color=(125, 255, 0))
img2 = cv.drawKeypoints(img2, kp2, None, color=(125, 255, 0))
img3 = cv.drawMatches(img1, kp1, img2, kp2,  matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('final', img3)
cv.waitKey(0)
cv.destroyAllWindows()


