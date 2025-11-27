Bước 1: Tải thư mục chứa mẫu biển báo (ClassifiSigns) được chứa trong link github hoặc từ google drive templates.txt.
Bước 2: Tải 2 video đầu vào video1.mp4 từ video1.txt và video2.mp4 từ video2.txt.
Bước 3: Mở thư mục main.py nơi chứa mã nguồn của code có 3 biến nền tảng cần thực thi:
student_id = "52300045_52300124" (ký tự sẽ được in góc dưới phải video)
input_video = 'video1.mp4' (đường dẫn video input đầu vào)
output_video = f'{student_id}_video1.mp4' (đường dẫn video output)
Bước 4: Tìm kiếm:
scale = 0.7 (khung hình hiển thị)
small_frame = cv.resize(frame, (int(W * scale), int(H * scale))) (resize video)
cv.imshow("Final", small_frame) (hiển thị video theo từng frame đã xử lý)
Bước 5: Thực thi mã nguồn.
