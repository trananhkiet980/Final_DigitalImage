import cv2 as cv
import numpy as np
import math
import time
import os

# =============================================================================
# GLOBAL CONFIGURATION (GIỮ NGUYÊN THAM SỐ GỐC)
# =============================================================================
student_id = "52300045_52300124"
input_video = 'video1.mp4'
output_video = f'{student_id}_video1.mp4'

# === THAM SỐ GIỚI HẠN KHUNG HÌNH (ROI) ===
ROI_X_MIN, ROI_X_MAX = 0.43, 0.75
ROI_Y_MIN, ROI_Y_MAX = 0.00, 0.48

# === NGƯỠNG PHÁT HIỆN CHÍNH (DIỆN TÍCH & KÍCH THƯỚC) ===
MIN_AREA = 550
MIN_SIGN_SIZE = 15
MAX_SIGN_SIZE = 550

# === NGƯỠNG PHÁT HIỆN HÌNH DẠNG (TRÒN & TAM GIÁC) ===
MIN_CIRCULARITY = 0.50 
MAX_AREA_ERROR = 0.25  
MIN_TRIANGLE_ASPECT, MAX_TRIANGLE_ASPECT = 0.5, 1.2 

# === NGƯỠNG PHÁT HIỆN MÀU TRẮNG/BIỂN PHỤ ===
MIN_AREA_WHITE = 100
ASPECT_MIN_WHITE, ASPECT_MAX_WHITE = 0.8, 3.5 

# === TRACKING & NMS ===
NMS_IOU = 0.3
IOU_TRACK = 0.3
MAX_MISS = 5

# === THAM SỐ PHÂN LOẠI BIỂN BÁO ===
CLASSIFY_SIZE = (32, 32)
SIMILARITY_THRESHOLD = 0.1

# =============================================================================
# HELPER CLASS: TRACK
# =============================================================================
class Track:
    def __init__(self, box, color, id, conf=1.0):
        self.box = box
        self.color = color
        self.id = id
        self.miss = 0
        self.conf = conf
    
    # Cập nhật vị trí dùng phương pháp làm mịn (exponential smoothing)
    def update(self, box, color, conf):
        a = 0.8
        self.box = [int(a*b + (1-a)*s) for b,s in zip(box, self.box)]
        self.color = color
        self.conf = conf
        self.miss = 0

# =============================================================================
# CLASS 1: PHÁT HIỆN BIỂN BÁO (TrafficSignDetector)
# Bao gồm: Xử lý mask màu, tìm contour, lọc hình dạng, NMS và Tracking
# =============================================================================
class TrafficSignDetector:
    def __init__(self):
        self.next_id = 0

    def iou(self, b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xi1, yi1 = max(x1,x2), max(y1,y2)
        xi2, yi2 = min(x1+w1,x2+w2), min(y1+h1,y2+h2)
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        union = w1*h1 + w2*h2 - inter
        return inter / union if union > 0 else 0

    def get_mask(self, hsv, color):
        hsv = hsv.copy()
        hsv[:, :, 2] = cv.equalizeHist(hsv[:, :, 2])
        masks = {
            'red':   [([0, 20, 20], [20, 255, 255]), ([110, 40, 40], [180, 255, 255])],
            'yellow': ([20, 150, 150], [40, 255, 255]),
            'blue':      ([95, 130, 130], [140, 250, 250]),
            'white':    ([0, 0, 160], [180, 80, 255])
        }
        if color == 'red':
            m1 = cv.inRange(hsv, np.array(masks['red'][0][0]), np.array(masks['red'][0][1]))
            m2 = cv.inRange(hsv, np.array(masks['red'][1][0]), np.array(masks['red'][1][1]))
            return m1 | m2
        else:
            lower = np.array(masks[color][0])
            upper = np.array(masks[color][1])
            return cv.inRange(hsv, lower, upper)

    def clean(self, m):
        m = cv.GaussianBlur(m, (5,5), 0)
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        m = cv.morphologyEx(m, cv.MORPH_OPEN, k)
        m = cv.morphologyEx(m, cv.MORPH_CLOSE, k)
        return m

    def white_sign_confidence(self, roi_hsv):
        if roi_hsv.size == 0: return 0
        bgr = cv.cvtColor(roi_hsv, cv.COLOR_HSV2BGR)
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150)
        return np.sum(edges > 0) / edges.size

    def is_circular_or_triangular(self, cnt):
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        if area == 0 or perimeter == 0: return False
        
        x, y, w, h = cv.boundingRect(cnt)

        # 1. Kiểm tra Hình Tròn
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity >= MIN_CIRCULARITY: 
            aspect = w / h
            if aspect >= 0.8 and aspect <= 1.2:
                 return True
            
        # 2. Kiểm tra Hình Tam Giác
        approx = cv.approxPolyDP(cnt, 0.03 * perimeter, True) 
        
        if len(approx) == 3:
            aspect = w / h
            if aspect < MIN_TRIANGLE_ASPECT or aspect > MAX_TRIANGLE_ASPECT:
                 return False
            approx_area = cv.contourArea(approx)
            if approx_area > 0 and abs(area - approx_area) / area < MAX_AREA_ERROR:
                 return True

        return False

    def detect_main(self, frame, hsv, x1, y1, x2, y2, mask_roi):
        dets = []
        for c in ['red', 'yellow', 'blue']:
            m = self.get_mask(hsv, c)
            m = self.clean(m)
            m = cv.bitwise_and(m, m, mask=mask_roi) 
            cnts, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv.contourArea(cnt)
                if area < MIN_AREA: continue 
                x,y,w,h = cv.boundingRect(cnt)
                if w < MIN_SIGN_SIZE or h < MIN_SIGN_SIZE or w > MAX_SIGN_SIZE or h > MAX_SIGN_SIZE: continue 
                
                if not self.is_circular_or_triangular(cnt): continue
                if not (x >= x1 and y >= y1 and x+w <= x2 and y+h <= y2): continue
                
                dets.append((x, y, w, h, c, 1.0))
        return dets

    def detect_white(self, frame, hsv, x1, y1, x2, y2, mask_roi):
        dets = []
        m = self.get_mask(hsv, 'white')
        m = self.clean(m)
        m = cv.bitwise_and(m, m, mask=mask_roi) 
        
        cnts, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv.contourArea(cnt)
            if area < MIN_AREA_WHITE: continue
            x,y,w,h = cv.boundingRect(cnt)
            if w < 10 or h < 10: continue
            
            asp = w / h
            if asp < ASPECT_MIN_WHITE or asp > ASPECT_MAX_WHITE: continue
            
            if not (x >= x1 and y >= y1 and x+w <= x2 and y+h <= y2): continue
            
            roi = hsv[y:y+h, x:x+w]
            conf = self.white_sign_confidence(roi)
            if conf < 0.08: continue
            dets.append((x, y, w, h, 'white', conf))
        return dets

    def nms(self, dets, thresh):
        if not dets: return []
        boxes = np.array([d[:4] for d in dets])
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,0]+boxes[:,2], boxes[:,1]+boxes[:,3]
        areas = (x2-x1)*(y2-y1)
        idx = np.argsort(y2)
        keep = []
        while len(idx)>0:
            i = idx[-1]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[idx[:-1]])
            yy1 = np.maximum(y1[i], y1[idx[:-1]])
            xx2 = np.minimum(x2[i], x2[idx[:-1]])
            yy2 = np.minimum(y2[i], y2[idx[:-1]])
            inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
            iou_vals = inter / (areas[i] + areas[idx[:-1]] - inter)
            idx = np.delete(idx[:-1], np.where(iou_vals > thresh)[0])
        return [dets[i] for i in keep]

    def track_update(self, new_dets, tracks):
        matched = [False]*len(new_dets)
        new_tracks = []
        for t in tracks:
            best_iou, best_idx = 0, -1
            for i, d in enumerate(new_dets):
                if matched[i]: continue
                iou_val = self.iou(t.box, d[:4])
                if iou_val > best_iou:
                    best_iou, best_idx = iou_val, i
            if best_iou > IOU_TRACK:
                cx_old = t.box[0] + t.box[2] / 2
                cy_old = t.box[1] + t.box[3] / 2
                cx_new = new_dets[best_idx][0] + new_dets[best_idx][2] / 2
                cy_new = new_dets[best_idx][1] + new_dets[best_idx][3] / 2
                move_dist = math.hypot(cx_new - cx_old, cy_new - cy_old)

                if move_dist > 10:
                    continue
                t.update(new_dets[best_idx][:4], new_dets[best_idx][4], new_dets[best_idx][5])
                matched[best_idx] = True
                new_tracks.append(t)
            else:
                t.miss += 1
                if t.miss < MAX_MISS:
                    new_tracks.append(t)
        for i, d in enumerate(new_dets):
            if not matched[i]:
                new_tracks.append(Track(d[:4], d[4], self.next_id, d[5]))
                self.next_id += 1
        return new_tracks

    def detect_and_track(self, frame, hsv, tracks):
        H, W = frame.shape[:2]
        x1 = int(W * ROI_X_MIN)
        y1 = int(H * ROI_Y_MIN)
        x2 = int(W * ROI_X_MAX)
        y2 = int(H * ROI_Y_MAX)
        mask_roi = np.zeros(frame.shape[:2], np.uint8)
        mask_roi[y1:y2, x1:x2] = 255
        roi_coords = (x1, y1, x2, y2)
        
        # Detect Main
        dets_main = self.detect_main(frame, hsv, *roi_coords, mask_roi)
        dets_main = self.nms(dets_main, NMS_IOU)

        # Detect White
        dets_white = self.detect_white(frame, hsv, *roi_coords, mask_roi)
        dets_white = self.nms(dets_white, NMS_IOU)

        # Tracking
        all_dets = dets_main + dets_white
        updated_tracks = self.track_update(all_dets, tracks)
        
        return updated_tracks

# =============================================================================
# CLASS 2: PHÂN LOẠI BIỂN BÁO (TrafficSignClassifier)
# Bao gồm: Cắt ảnh, phân tích tỉ lệ màu, so sánh template
# =============================================================================
class TrafficSignClassifier:
    def get_color_ratio(self, roi_bgr):
        # 1. Kiểm tra đầu vào
        if roi_bgr is None or roi_bgr.size == 0:
            return {'red': 0, 'black': 0, 'yellow': 0, 'blue': 0, 'white': 0}
        
        # 2. Chuyển sang HSV
        hsv = cv.cvtColor(roi_bgr, cv.COLOR_BGR2HSV)
        
        # 3. Cân bằng sáng
        hsv[:, :, 2] = cv.equalizeHist(hsv[:, :, 2])
        
        total_pixels = roi_bgr.shape[0] * roi_bgr.shape[1]
        
        # 4. Định nghĩa các khoảng màu
        color_ranges = {
            'red': [
                ([0, 20, 20], [20, 255, 255]),
                ([110, 30, 30], [170, 255, 255])
            ],
            'yellow': [([20, 70, 70], [40, 255, 255])],
            'blue': [([95, 130, 130], [140, 250, 250])],
            'black': [([0, 0, 0], [180, 70, 70])], 
            'white': [([0, 0, 200], [180, 40, 255])]
        }
        
        ratios = {}
        
        # --- LOGIC ƯU TIÊN (PRIORITY MASKING) ---
        # B1: Tạo mask tổng hợp cho các màu "có sắc" (Red, Yellow, Blue, White) trước
        chromatic_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        priority_colors = ['red', 'yellow', 'blue', 'white']
        
        # Tính mask cho các màu ưu tiên và lưu tạm vào ratios (tạm thời chưa chia total_pixels)
        temp_masks = {}
        
        for color_name in priority_colors:
            ranges = color_ranges[color_name]
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv.inRange(hsv, np.array(lower), np.array(upper))
            
            temp_masks[color_name] = mask
            chromatic_mask |= mask # Cộng dồn vào mask tổng hợp
            
        # B2: Tính mask cho màu Đen (Black) và LOẠI TRỪ các màu kia
        black_mask_raw = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges['black']:
            black_mask_raw |= cv.inRange(hsv, np.array(lower), np.array(upper))
            
        # Chỉ nhận là Đen nếu KHÔNG PHẢI là các màu kia
        temp_masks['black'] = cv.bitwise_and(black_mask_raw, cv.bitwise_not(chromatic_mask))
        
        # 5. Tính toán tỉ lệ cuối cùng
        all_colors = ['red', 'yellow', 'blue', 'white', 'black']
        for color_name in all_colors:
            ratios[color_name] = np.sum(temp_masks[color_name] > 0) / total_pixels
        
        return ratios

    def get_folder_by_color(self, ratios):
        red = ratios['red']
        black = ratios['black']
        yellow = ratios['yellow']
        blue = ratios['blue']

        if yellow > 0.08  and red>0.04:
            return "ClassifiSigns/RedBlackYellow"

        if blue > 0.4:
            if red > 0.2:
                    return "ClassifiSigns/RedBlue"
            return "ClassifiSigns/Blue"
        elif red > 0.6 and black<0.2:
            return "ClassifiSigns/Red"
        elif red>0.15 and black>0.1:
            return "ClassifiSigns/Red"
        elif red > 0.2 and blue < 0.4: 
            if blue > 0.1:
                return "ClassifiSigns/RedBlue"
            elif black > 0.05 and black<0.1:
                return "ClassifiSigns/RedBlack"
        return None

    def standardize_template(self, img, target_size):
        h, w = img.shape[:2]
        S = max(h, w)
        dw = (S - w) // 2
        dh = (S - h) // 2

        padded = cv.copyMakeBorder(
            img, dh, S - h - dh, dw, S - w - dw,
            cv.BORDER_CONSTANT, value=[255, 255, 255]
        )
        return cv.resize(padded, target_size, interpolation=cv.INTER_AREA)

    def compare_images(self, img1, img2):
        g1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        g2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        
        res = cv.matchTemplate(g1, g2, cv.TM_CCOEFF_NORMED)
        score_shape = res[0][0]
        score_shape = max(0, score_shape)

        hsv1 = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
        hsv2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)
        
        hist1 = cv.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist2 = cv.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])
        
        cv.normalize(hist1, hist1, 0, 1, cv.NORM_MINMAX)
        cv.normalize(hist2, hist2, 0, 1, cv.NORM_MINMAX)
        
        score_color = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
        score_color = max(0, score_color)

        final_score = 0.6 * score_shape + 0.4 * score_color
        return final_score

    def classify_sign(self, frame, x, y, w, h):
        try:
            os.makedirs("cropped_sign", exist_ok=True)

            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                return None

            timestamp = int(time.time() * 1000)
            # cv.imwrite(f"cropped_sign/{timestamp}_original.jpg", roi)

            roi_blur = cv.GaussianBlur(roi, (3, 3), 0)

            h0, w0 = roi_blur.shape[:2]
            S = max(h0, w0)
            dw = (S - w0) // 2
            dh = (S - h0) // 2

            roi_square = cv.copyMakeBorder(
                roi_blur, dh, S - h0 - dh, dw, S - w0 - dw,
                cv.BORDER_CONSTANT, value=[255, 255, 255]
            )

            target_size = (roi_square.shape[1], roi_square.shape[0])
            roi_resized = roi_square 

            # cv.imwrite(f"cropped_sign/{timestamp}_square.jpg", roi_resized)

            color_ratios = self.get_color_ratio(roi)
            folder = self.get_folder_by_color(color_ratios)

            if folder is None or not os.path.exists(folder):
                return None

            best_similarity = -1.0
            best_match = None

            for filename in os.listdir(folder):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue

                template_path = os.path.join(folder, filename)
                template = cv.imread(template_path)
                if template is None:
                    continue

                template_resized = self.standardize_template(template, target_size)
                similarity = self.compare_images(roi_resized, template_resized)
                # cv.imwrite(f"cropped_sign/{timestamp}_template.jpg", template_resized)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = os.path.splitext(filename)[0]

            if best_similarity > SIMILARITY_THRESHOLD:
                return best_match

            return None

        except Exception as e:
            print(f"Lỗi trong classify_sign: {e}")
            return None

# =============================================================================
# CLASS 3: XỬ LÝ VIDEO (VideoProcessor)
# Bao gồm: Vòng lặp chính, vẽ bounding box, hiển thị và lưu video
# =============================================================================
class VideoProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.detector = TrafficSignDetector()
        self.classifier = TrafficSignClassifier()
        self.tracks = []

    def process_frame(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        # 1. Gọi Detector để phát hiện và tracking
        self.tracks = self.detector.detect_and_track(frame, hsv, self.tracks)

        # 2. Vẽ kết quả và Phân loại
        H, W = frame.shape[:2]
        for t in self.tracks:
            if t.miss > 0: 
                continue
            
            if t.color != 'white':
                x, y, w, h = t.box
                # Gọi Classifier để lấy tên biển báo
                sign_name = self.classifier.classify_sign(frame, x, y, w, h)

                if sign_name==None:
                    continue

                color_map = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'yellow': (0, 255, 255)}
                box_color = color_map.get(t.color, (0, 255, 0))
                
                cv.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                
                display_text = sign_name if sign_name else t.color
                
                if display_text:
                    display_text = display_text.replace("_", " ")
                    display_text = " ".join(display_text.split())
                    
                    font = cv.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    
                    (text_w, text_h), baseline = cv.getTextSize(display_text, font, font_scale, thickness)
                    
                    # === SỬA ĐỔI TẠI ĐÂY ===
                    # Tính vị trí bên phải: Cạnh phải (x+w) + khoảng cách (5px)
                    text_x = x + w + 5 
                    text_y = y + text_h + 5
                    
                    # Kiểm tra: Nếu text tràn ra khỏi mép phải màn hình
                    # frame.shape[1] là chiều rộng của ảnh
                    if text_x + text_w > frame.shape[1]:
                        # Thì đẩy ngược lại về bên trái
                        text_x = x - text_w - 5 
                    # =======================
                    
                    cv.rectangle(frame, 
                                 (text_x - 2, text_y - text_h - 2), 
                                 (text_x + text_w + 2, text_y + baseline), 
                                 (0, 0, 0), -1)
                    
                    cv.putText(frame, display_text, (text_x, text_y), 
                               font, font_scale, (255, 255, 255), thickness)
        
        cv.putText(frame, student_id, (20, H - 20), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame

    def run(self):
        cap = cv.VideoCapture(self.input_path)
        if not cap.isOpened():
            print("Không mở video!")
            return
        
        fps = int(cap.get(cv.CAP_PROP_FPS)) or 30
        W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(self.output_path, fourcc, fps, (W,H))
        
        fcount = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Dừng đọc video tại frame {fcount} (ret=False)")
                break

            if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                print(f"Bỏ qua frame lỗi tại {fcount}")
                continue 

            frame = self.process_frame(frame)
            out.write(frame)

            # scale = 0.7 
            # small_frame = cv.resize(frame, (int(W * scale), int(H * scale)))
            # cv.imshow("Final", small_frame)

            fcount += 1
            if fcount % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {fcount} frames | FPS: {fcount/elapsed:.2f}")

            if cv.waitKey(1) == 27: 
                break

        cap.release()
        out.release()
        cv.destroyAllWindows()
        print(f"HOÀN THÀNH: {self.output_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    processor = VideoProcessor(input_video, output_video)
    processor.run()

if __name__ == "__main__":
    main()