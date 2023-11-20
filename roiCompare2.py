import cv2  # OpenCV 라이브러리를 가져옵니다. OpenCV는 컴퓨터 비전을 위한 라이브러리입니다.
import numpy as np  # numpy 라이브러리를 가져옵니다. numpy는 다차원 배열을 처리하는데 사용되는 라이브러리입니다.

def main():
    # 이미지 파일 읽기
    image1 = cv2.imread("./assets/50.jpeg")  # 첫 번째 이미지 파일을 읽어옵니다.
    image2 = cv2.imread("./assets/60.jpeg")  # 두 번째 이미지 파일을 읽어옵니다.

    # 이미지 크기 및 중앙 좌표 가져오기
    height, width, _ = image1.shape  # 첫 번째 이미지의 높이와 너비를 가져옵니다.
    center_x, center_y = width // 2, height // 2  # 이미지의 중앙 좌표를 계산합니다.

    # 중앙 rect 좌표 설정 (ROI 범위)
    rect_size = 1000  # rect의 크기를 설정합니다.
    x1 = center_x - rect_size // 2  # rect의 왼쪽 위 x 좌표를 계산합니다.
    y1 = center_y - rect_size // 2  # rect의 왼쪽 위 y 좌표를 계산합니다.
    x2 = center_x + rect_size // 2  # rect의 오른쪽 아래 x 좌표를 계산합니다.
    y2 = center_y + rect_size // 2  # rect의 오른쪽 아래 y 좌표를 계산합니다.

    # 템플릿 이미지로 사용할 부분 추출 (ROI)
    basicImg = image1[y1:y2, x1:x2]  # 첫 번째 이미지에서 rect 부분을 잘라서 basicImg에 저장합니다.

    # 두 번째 이미지의 중앙 rect 추출
    compareImg = image2[y1:y2, x1:x2]  # 두 번째 이미지에서 rect 부분을 잘라서 compareImg에 저장합니다.

    # 이미지에서 특징점 찾기
    sift = cv2.SIFT_create()  # SIFT(Scale-Invariant Feature Transform)를 사용하여 이미지의 특징점을 추출합니다.
    kp1, descriptors1 = sift.detectAndCompute(basicImg, None)  # 첫 번째 이미지에서 특징점을 찾습니다.
    kp2, descriptors2 = sift.detectAndCompute(compareImg, None)  # 두 번째 이미지에서 특징점을 찾습니다.

    # 특징점 매칭
    bf = cv2.BFMatcher()  # BFMatcher 객체를 생성합니다. BFMatcher는 두 이미지 사이의 특징점 매칭을 수행합니다.
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)  # 첫 번째 이미지와 두 번째 이미지 사이의 특징점을 매칭합니다.

    # 좋은 매치 선택
    good_matches = []  # 좋은 매치를 저장할 리스트를 생성합니다.
    for m, n in matches:  # 각 매치에 대해
        if m.distance < 0.75 * n.distance:  # 첫 번째 가장 가까운 매치(m)의 거리가 두 번째 가장 가까운 매치(n)의 거리의 75%보다 작으면
            good_matches.append(m)  # m을 좋은 매치로 간주하고 리스트에 추가합니다.

    res = cv2.drawMatches(image1, kp1, image2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 호모그래피 계산
    if len(good_matches) > 10:  # 좋은 매치가 10개 이상이면
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 첫 번째 이미지의 좋은 매치 좌표를 가져옵니다.
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 두 번째 이미지의 좋은 매치 좌표를 가져옵니다.
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # 호모그래피를 계산합니다. 호모그래피는 한 이미지에서 다른 이미지로 좌표를 변환하는 변환 행렬입니다.
    else:
        raise AssertionError("Can't find enough good matches!")  # 좋은 매치가 충분하지 않으면 에러를 발생시킵니다.

    # 이미지 레지스트레이션
    registered_image = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]))  # 첫 번째 이미지를 두 번째 이미지에 맞게 변환합니다.

    # ROI 추출
    roi1 = registered_image[y1:y2, x1:x2]  # 변환된 첫 번째 이미지에서 rect 부분을 잘라서 roi1에 저장합니다.
    roi2 = image2[y1:y2, x1:x2]  # 두 번째 이미지에서 rect 부분을 잘라서 roi2에 저장합니다.

    # ROI 내의 모든 픽셀의 B, G, R (파란색, 녹색, 빨간색) 색상 값을 평균 계산
    average_color1 = np.mean(np.mean(roi1, axis=0), axis=0).astype(int)  # roi1의 평균 색상을 계산합니다.
    average_color2 = np.mean(np.mean(roi2, axis=0), axis=0).astype(int)  # roi2의 평균 색상을 계산합니다.

    print("image1: ", average_color1)  # roi1의 평균 색상을 출력합니다.
    print("image2: ", average_color2)  # roi2의 평균 색상을 출력합니다.
    print("img1 - img2: ", average_color1 - average_color2)  # 두 이미지의 평균 색상 차이를 출력합니다.

    # ROI 뺄셈
    result_roi = cv2.absdiff(roi1, roi2)  # 두 이미지의 차이를 계산합니다.

    # 이미지 리사이즈
    resized_image1 = cv2.resize(image1, (0, 0), fx=0.5, fy=0.5)  # 첫 번째 이미지를 절반 크기로 리사이즈합니다.
    resized_image2 = cv2.resize(image2, (0, 0), fx=0.5, fy=0.5)  # 두 번째 이미지를 절반 크기로 리사이즈합니다.
    resized_res = cv2.resize(res, (0, 0), fx=0.5, fy=0.5)  # 두 번째 이미지를 절반 크기로 리사이즈합니다.

    # 결과 이미지에 rectangle 그리기
    cv2.rectangle(resized_image1, (x1 // 2, y1 // 2), (x2 // 2, y2 // 2), (0, 255, 0), 2)  # 첫 번째 이미지에 rect를 그립니다.
    cv2.rectangle(resized_image2, (x1 // 2, y1 // 2), (x2 // 2, y2 // 2), (0, 255, 0), 2)  # 두 번째 이미지에 rect를 그립니다.

    # 결과 이미지 화면에 표시
    cv2.imshow("Image 1 with Rectangle", resized_image1)  # 첫 번째 이미지를 화면에 표시합니다.
    cv2.imshow("Image 2 with Rectangle", resized_image2)  # 두 번째 이미지를 화면에 표시합니다.
    cv2.imshow("ROI Difference", result_roi)  # 두 이미지의 차이를 화면에 표시합니다.
    cv2.imshow("Matched Features", resized_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()