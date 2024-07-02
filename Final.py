import cv2
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
import cvzone

def compute_mhi_and_optical_flow(video_path, duration=1.0, max_value=255):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None, None

    ret, frame = cap.read()
    if not ret:
        print("Error reading video")
        return None, None

    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_frame.shape
    mhi = np.zeros((h, w), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_diff = cv2.absdiff(gray, prev_frame)
        _, motion_mask = cv2.threshold(motion_diff, 30, 1, cv2.THRESH_BINARY)
        mhi = np.maximum(mhi - 1.0/duration, 0) + (motion_mask * max_value)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev_frame = gray

    cap.release()
    return mhi, flow

def visualize_mhi(mhi):
    plt.figure(figsize=(10, 10))
    plt.imshow(mhi, cmap='jet')
    plt.colorbar()
    plt.title("Motion History Image")
    plt.show()

def visualize_optical_flow(flow):
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 1
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title("Optical Flow Magnitude and Direction")
    plt.show()

def detect_objects(video_path):
    model = YOLO("best.pt")
    class_names = model.names
    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue

        img = cv2.resize(img, (1020, 500))
        h, w, _ = img.shape
        results = model.predict(img)

        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y,x1,y1 = cv2.boundingRect(contour)
                    cv2.polylines(img, [contour],True, color=(0, 0, 255), thickness=2)
    #                cv2.rectangle(img,(x,y),(x1+x,y1+y),(255,0,0),2)
                    cv2.putText(img, c, (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main process
video_path = 'cv.mp4'
mhi, flow = compute_mhi_and_optical_flow(video_path)

if mhi is not None and flow is not None:
    visualize_mhi(mhi)
    visualize_optical_flow(flow)
    detect_objects(video_path)
