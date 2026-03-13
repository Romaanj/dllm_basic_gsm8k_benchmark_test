import cv2
import os

# 1. 파일 경로 설정
video_path = './file_example_MP4_480_1_5MG.mp4' # 영상 파일 경로
output_folder = 'frames'       # 저장할 폴더 이름

# 저장 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. 영상 불러오기
cap = cv2.VideoCapture(video_path)
count = 0

while cap.isOpened():
    # 프레임을 하나씩 읽기
    ret, frame = cap.read()
    
    # 더 이상 읽을 프레임이 없으면 종료
    if not ret:
        break
    
    # 3. 프레임을 이미지로 저장 (파일명: frame_0001.jpg 형식)
    file_name = os.path.join(output_folder, f"frame_{count:04d}.jpg")
    cv2.imwrite(file_name, frame)
    
    count += 1

# 4. 자원 해제
cap.release()
print(f"작업 완료! 총 {count}개의 프레임이 저장되었습니다.")
