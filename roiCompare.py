# 두 이미지의 가운데 부분에 roi를 추출하고 두 roi에 대한 색상 정보와 roi 간 차이를 출력

import cv2
import numpy as np

# 첫 번째 이미지 파일 읽기
image1 = cv2.imread("./assets/over_70.jpeg")

# 두 번째 이미지 파일 읽기
image2 = cv2.imread("./assets/30.jpeg")

# 이미지 크기 및 중앙 좌표 가져오기
height, width, _ = image1.shape
center_x, center_y = width // 2, height // 2

# 중앙 rect 좌표 설정
rect_size = 1000  # rect의 크기
x1 = center_x - rect_size // 2
y1 = center_y - rect_size // 2
x2 = center_x + rect_size // 2
y2 = center_y + rect_size // 2

# tempx1 = x1 - 170
# tempx2 = x2 - 170
# tempy1 = y1 + 230
# tempy2 = y2 + 230
# 첫 번째 이미지의 중앙 rect 추출
roi1 = image1[y1:y2, x1:x2]
# roi1 = image1[tempy1:tempy2, tempx1:tempx2]

# 두 번째 이미지의 중앙 rect 추출
roi2 = image2[y1:y2, x1:x2]

# roi1 내의 모든 픽셀의 B, G, R (파란색, 녹색, 빨간색) 색상 값을 평균 계산
average_color1 = np.mean(np.mean(roi1, axis=0), axis=0)
# average_color를 정수 데이터 형식으로 변환
average_color1 = average_color1.astype(int)

# roi2 내의 모든 픽셀의 B, G, R (파란색, 녹색, 빨간색) 색상 값을 평균 계산
average_color2 = np.mean(np.mean(roi2, axis=0), axis=0)
# average_color를 정수 데이터 형식으로 변환
average_color2 = average_color2.astype(int)

print("image1: ", average_color1)
print("image2: ", average_color2)
print("img1 - img2: ", average_color1 - average_color2)

# ROI 뺄셈
result_roi = cv2.absdiff(roi1, roi2)
# print(result_roi)

# 이미지 리사이즈
resized_image1 = cv2.resize(image1, (0, 0), fx=0.5, fy=0.5)  # 이미지 크기를 50%로 조절
resized_image2 = cv2.resize(image2, (0, 0), fx=0.5, fy=0.5)  # 이미지 크기를 50%로 조절

# 결과 이미지에 rectangle 그리기
# cv2.rectangle(image1, (x1, y1), (x2, y2), (0, 255, 0), 2)
# cv2.rectangle(image2, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.rectangle(resized_image1, (x1 // 2, y1 // 2), (x2 // 2, y2 // 2), (0, 255, 0), 2)
# cv2.rectangle(resized_image1, (tempx1 // 2, tempy1 // 2), (tempx2 // 2, tempy2 // 2), (0, 255, 0), 2)
cv2.rectangle(resized_image2, (x1 // 2, y1 // 2), (x2 // 2, y2 // 2), (0, 255, 0), 2)

# 결과 이미지 화면에 표시
cv2.imshow("Image 1 with Rectangle", resized_image1)
cv2.imshow("Image 2 with Rectangle", resized_image2)
cv2.imshow("ROI Difference", result_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
