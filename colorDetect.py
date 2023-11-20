# 간단한 이미지 색상 디텍팅! 색상 정보를 return

import cv2
import numpy as np

# 이미지 파일 읽기
image = cv2.imread("./assets/70.jpeg")

# 이미지 크기 및 중앙 좌표 가져오기
height, width, _ = image.shape
center_x, center_y = width // 2, height // 2

# 중앙 rect 좌표 설정
rect_size = 300  # rect의 크기
x1 = center_x - rect_size // 2
y1 = center_y - rect_size // 2
x2 = center_x + rect_size // 2
y2 = center_y + rect_size // 2

# 중앙 rect 추출
roi = image[y1:y2, x1:x2]

# rect 내의 평균 색상 계산
# roi 내의 모든 픽셀의 B, G, R (파란색, 녹색, 빨간색) 색상 값을 평균 계산
average_color = np.mean(np.mean(roi, axis=0), axis=0)
# average_color를 정수 데이터 형식으로 변환
average_color = average_color.astype(int)

# 평균 색상 출력 (BGR)
print("Average Color (BGR):", average_color)

# 빨간색에 가까운지 확인
blue, green, red = average_color
if red > 100 and blue < 50 and green < 50:
    print("주의: 빨간색에 가까운 색상을 감지했습니다!")

# 이미지 리사이즈
resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)  # 이미지 크기를 50%로 조절

# # 이미지에 rectangle 그리기
# cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
# 이미지에 rectangle 그리기
cv2.rectangle(resized_image, (x1 // 2, y1 // 2), (x2 // 2, y2 // 2), (0, 255, 0), 2)

# 이미지 화면에 표시
cv2.imshow("Image with Rectangle", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()