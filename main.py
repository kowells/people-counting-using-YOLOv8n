import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time

# Inisialisasi model dan input
model = YOLO("model12kcln.pt")
cap = cv2.VideoCapture("data1.mp4")  

# Tentukan penggunaan perangkat GPU jika tersedia
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


# Ukuran frame
new_width = 1280
new_height = 720

# Koordinat garis horizontal untuk deteksi masuk/keluar
cy1 = 340
cy2 = 300
offset = 6

# Inisialisasi variabel untuk perhitungan
person_in = {}
counter_in = []
person_out = {}
counter_out = []

# Variabel untuk menghitung FPS
fps = 30  # FPS video input
frame_time = 1.0 / fps  # Waktu per frame dalam detik
last_frame_time = time.time()

while cap.isOpened():
    current_time = time.time()
    elapsed_time = current_time - last_frame_time
    
    # Hanya proses frame jika waktu yang berlalu sudah cukup
    if elapsed_time >= frame_time:
        last_frame_time = current_time
        
        success, frame = cap.read()
        if success:
            # Ubah ukuran frame
            frame = cv2.resize(frame, (new_width, new_height))
            frame_height, frame_width, _ = frame.shape
            
            # Deteksi dan pelacakan objek menggunakan ByteTrack
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            
            # Hitung FPS
            fps_display = 1.0 / elapsed_time
            
            # Pastikan results dan results[0].boxes tidak None
            if results and results[0] is not None and results[0].boxes is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id
                # Pastikan track_ids tidak None
                if track_ids is not None:
                    track_ids = track_ids.int().cpu().tolist()
                else:
                    track_ids = []  # Jika track_ids adalah None, gunakan list kosong
                
                annotated_frame = results[0].plot()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    cx, cy = int(x), int(y)

                    # Deteksi orang masuk
                    if cy1 < (cy + offset) and cy1 > (cy - offset):
                        person_in[track_id] = cy
                    if track_id in person_in:
                        if cy2 < (cy + offset) and cy2 > (cy - offset):
                            cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)
                            cv2.putText(annotated_frame, str(track_id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                            if counter_in.count(track_id) == 0:
                                counter_in.append(track_id)

                    # Deteksi orang keluar
                    if cy2 < (cy + offset) and cy2 > (cy - offset):
                        person_out[track_id] = cy
                    if track_id in person_out:
                        if cy1 < (cy + offset) and cy1 > (cy - offset):
                            cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)
                            cv2.putText(annotated_frame, str(track_id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                            if counter_out.count(track_id) == 0:
                                counter_out.append(track_id)

                # Menampilkan garis dan informasi jumlah orang masuk/keluar
                cv2.line(annotated_frame, (150, cy1), (900, cy1), (255, 255, 255), 1)
                cv2.putText(annotated_frame, 'in', (140, 318), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

                cv2.line(annotated_frame, (150, cy2), (900, cy2), (255, 255, 255), 1)
                cv2.putText(annotated_frame, 'out', (140, 363), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

                # Menampilkan jumlah orang masuk, keluar, dan di dalam ruangan
                cv2.putText(annotated_frame, 'masuk= ' + str(len(counter_in)), (60, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(annotated_frame, 'keluar= ' + str(len(counter_out)), (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(annotated_frame, 'di dalam= ' + str(len(counter_in) - len(counter_out)), (60, 120), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

                # Menampilkan FPS di pojok kanan atas
                #cv2.putText(annotated_frame, f'FPS: {fps_display:.2f}', (new_width - 150, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

                # Tampilkan frame yang sudah dianotasi
                cv2.imshow("YOLOv8 People Counting", annotated_frame)

            # Tekan tombol ESC untuk keluar
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            break
    else:
        # Jika waktu belum cukup, tunggu sebentar
        time.sleep(frame_time - elapsed_time)

cap.release()
cv2.destroyAllWindows()

# Print jumlah orang masuk, keluar, dan di dalam ruangan
print("masuk: ", len(counter_in))
print("keluar: ", len(counter_out))
print("didalam: ", len(counter_in) - len(counter_out))
