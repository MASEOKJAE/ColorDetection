# 두 이미지의 가운데 roi를 추출하여 두 roi의 패턴과 색상 명도에 따른 매칭 정보를 선으로 표현
# 이를통해 두 이미지를 매칭 정보를 통해 색상 정보에 대해서 패턴이 완전히 일치하지 않는 이미지라도 정보를 얻을 수 있음

import cv2
import numpy as np 

img1 = cv2.imread('./assets/60.jpeg')
img2 = cv2.imread('./assets/over_70.jpeg')

# 이미지 크기 및 중앙 좌표 가져오기
height, width, _ = img1.shape  # 첫 번째 이미지의 높이와 너비를 가져옴
center_x, center_y = width // 2, height // 2  # 이미지의 중앙 좌표를 계산

# 중앙 rect 좌표 설정 (ROI 범위)
rect_size = 1000  # rect의 크기를 설정
x1 = center_x - rect_size // 2  # rect의 왼쪽 위 x 좌표를 계산
y1 = center_y - rect_size // 2  # rect의 왼쪽 위 y 좌표를 계산
x2 = center_x + rect_size // 2  # rect의 오른쪽 아래 x 좌표를 계산
y2 = center_y + rect_size // 2  # rect의 오른쪽 아래 y 좌표를 계산


# 템플릿 이미지로 사용할 부분 추출 (ROI)
basicImg = img1[y1:y2, x1:x2]  # 첫 번째 이미지에서 rect 부분을 잘라서 basicImg에 저장


# 두 번째 이미지의 중앙 rect 추출
compareImg = img2[y1:y2, x1:x2]  # 두 번째 이미지에서 rect 부분을 잘라서 compareImg에 저장


# 두 이미지를 gray scale로 변환하는 이유는 계산 비용을 줄이고 명암 정보를 통해 색상 일치 여부를 판단하기 위함
gray1 = cv2.cvtColor(basicImg, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(compareImg, cv2.COLOR_BGR2GRAY)


# SIFT 객체 생성
# SIFT 알고리즘?: 이미지의 크기나 회전 등에 불변인 특징점을 추출하는 알고리즘, 이 특징점들은 객체 인식, 이미지 매칭 등의 작업에서 사용되며, 각 특징점은 그 위치에서의 이미지의 방향과 크기 정보를 포함
#               - 이 정보를 통해 각 특징점에서 128차원의 디스크립터를 생성함. 이 디스크립터는 그 특징점 주변의 명암 패턴을 나타내는데, 이 디스크립터를 기반으로 두 이미지의 특징점들 사이에서 매칭을 찾음
detector = cv2.SIFT_create()


# 이미지 특징 점 추출, 특징 디스크럽터 계산
# detect And compute
#   - detect: 이미지에서 특징점(키포인트, keypoints) 찾음, 이 특징점들은 이미지에서 유니크한 구조를 가진 지점들을 나타냄 (지점: 에지, 코너 등)
#   - compute: 각 특징점에서의 SIFT 디스크립터를 계산, 디스크립터는 해당 특징점 주변의 이미지 패턴을 나타내는 벡터 -> 이 디스크립터는 이미지의 회전이나 스케일 변화에도 불변인 특징을 가짐
kp1, desc1 = detector.detectAndCompute(gray1, None) # (gray1)에서 특징점을 찾고, 해당 특징점의 디스크립터를 계산하여 kp1과 desc1에 저장
kp2, desc2 = detector.detectAndCompute(gray2, None) # (gray2)에서 특징점을 찾고, 해당 특징점의 디스크립터를 계산하여 kp2과 desc2에 저장
# more info: 
#   - kp1: 첫 번째 이미지의 특징점들을 나타내는 리스트, 각 특징점은 cv2.KeyPoint 객체로 표현되며, 이 객체는 특징점의 위치(x, y), 크기, 방향 등의 정보를 포함
#   - desc1: 첫 번째 이미지의 각 특징점의 디스크립터를 나타내는 numpy 배열, 각 행은 하나의 특징점 디스크립터를 나타내며, 각 디스크립터는 128차원의 벡터

# BFMatcher 생성, L1 거리 측정 기준 사용
# BFMatcher(Brute-Force Matcher): 두 개의 디스크립터를 직접 비교하는 가장 간단한 매칭 메소드
#   - cv2.NORM_L1: 매칭 간의 거리를 측정하는 방식, L1 norm은 두 디스크립터 간의 절대값 차이의 합을 계산
#   - crossCheck=True: 이와 반대 방향의 가장 가까운 매칭도 찾아서 두 방향이 서로 일치하는 매칭만을 반환
matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)


# 두 이미지 특징 디스크럽터 매칭
# 첫번째 matches: 첫 번째 이미지의 각 디스크립터와 두 번째 이미지의 디스크립터를 비교해서 매칭을 찾는 과정, matches는 매칭 결과를 담은 DMatch 객체의 리스트
matches = matcher.match(desc1, desc2)
# 두번째 matches: 매칭 결과를 distance 속성에 따라 정렬하는 과정, distance는 매칭된 두 디스크립터 간의 거리를 의미하며, 거리가 짧을수록 두 디스크립터는 더 유사하다는 것을 의미
matches = sorted(matches, key = lambda x:x.distance)
# 정렬된 매칭 결과에서 가장 거리가 먼 매칭(max_dist)과 가장 거리가 짧은 매칭(min_dist)의 거리를 찾는 과정, 추후 좋은 매칭 선별 시 사용
max_dist, min_dist = matches[-1].distance, matches[0].distance


# 최소 거리의 ratio 지점을 임계점으로 설정
# ratio와 boundary는 매칭의 "질"을 결정하는 데 사용
#   - ratio가 높을수록, 더 많은 매칭을 "좋은" 매칭으로 간주하므로, 더 많은 선이 그려짐
#   - ratio가 낮으면, 오직 가장 좋은 매칭만이 선택되므로 선의 수가 줄어듬
#   - 두 이미지 사이의 유사성을 시각적으로 나타내는 도구
ratio = 0.7 # 실험 값 => 이 값만 바꾸어 가면서 최선의 지점을 찾음
# boundary: '좋은 매칭'을 선별하기 위한 거리 임계값을 계산하는 과정, 임계값은 최소 거리(min_dist)와 최대 거리(max_dist) 사이의 거리에 ratio를 곱한 후 최소 거리를 더한 값
#   - 이렇게 하면 거리가 짧은 매칭들부터 차례대로 선별할 수 있음
boundary = (max_dist - min_dist) * ratio + min_dist
# correct_matches: 파이썬의 리스트 컴프리헨션(list comprehension) 문법을 사용하여 matches 리스트에서 조건에 맞는 원소만을 선택하여 새로운 리스트 correct_matches를 만드는 과정
#   - m for m in matches:       matches라는 리스트에 있는 각 원소(여기서는 m)를 순차적으로 접근하는 구문
#   - if m.distance < boundary: m.distance는 매칭된 두 디스크립터 사이의 거리를 나타냄, 이 거리가 boundary보다 작은 경우에만 해당 매칭(m)을 선택
#   - m:                        이 부분은 최종적으로 선택된 매칭을 나타냄. 조건문을 만족하는 매칭(m)만을 새로운 리스트 correct_matches에 추가
correct_matches = [m for m in matches if m.distance < boundary]


# 매치 된 결과 시각화
# cv2.drawMatches: 두 이미지 사이의 매칭 결과를 시각화하는 과정
#   - 첫 번째 이미지와 두 번째 이미지에 있는 매칭된 특징점들을 선으로 연결하여 그려주는 함수
#   - basicImg: 첫 번째 이미지, 이 이미지의 특징점들이 두 번째 이미지의 특징점들과 매칭
#   - kp1: 첫 번째 이미지의 특징점들
#   - compareImg: 두 번째 이미지, 이 이미지의 특징점들이 첫 번째 이미지의 특징점들과 매칭
#   - kp2: 두 번째 이미지의 특징점들
#   - correct_matches: 두 이미지 사이의 매칭 정보,이 정보에 따라 두 이미지 사이의 매칭된 특징점들을 선으로 연결하여 그림
#   - None: 이 부분은 선택적인 마스크 이미지를 입력하는 파라미터, 여기서는 사용하지 않으므로 None으로 설정
#   - flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS: 이 부분은 매칭되지 않은 특징점들은 그리지 않도록 설정하는 플래그
res = cv2.drawMatches(basicImg, kp1, compareImg, kp2, correct_matches, None, \
flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# 결과 출력
cv2.imshow("result", res)
cv2.waitKey()
cv2.destroyAllWindows()