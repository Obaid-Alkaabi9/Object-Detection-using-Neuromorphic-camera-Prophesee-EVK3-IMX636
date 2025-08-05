import numpy as np
import cv2
import time
from metavision_core.event_io import EventsIterator
from ultralytics import YOLO

# === Settings ===
MAX_EVENTS = 50000
DELTA_T = 100000  # microseconds
MODEL_PATH = "runs/detect/train/weights/best.pt"  # Update if needed
FRAME_SKIP = 1  # Run YOLO every frame or increase to reduce load

# === Load YOLOv8 Model ===
print("📦 Loading YOLO model...")
model = YOLO(MODEL_PATH)

# === Init Event Stream ===
print("📡 Starting live camera stream...")
stream_obj = EventsIterator(input_path="", delta_t=DELTA_T)
stream = iter(stream_obj)
first_events = next(stream)
height, width = stream_obj.get_size()
print(f"🖼️ Resolution: {width}x{height}")

frame_index = 1
print("🚀 Running live detection... Press 'q' to quit.")

# === Main Loop ===
while True:
    try:
        start_total = time.time()

        # Get next events
        if frame_index == 1:
            events = first_events
        else:
            events = next(stream)

        if len(events) > MAX_EVENTS:
            events = events[:MAX_EVENTS]

        x = events['x']
        y = events['y']
        p = events['p']
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y, p = x[valid], y[valid], p[valid]

        # Reconstruct frame (same as training)
        accum_on = np.zeros((height, width), dtype=np.int16)
        accum_off = np.zeros((height, width), dtype=np.int16)
        np.add.at(accum_on, (y[p == 1], x[p == 1]), 1)
        np.add.at(accum_off, (y[p == 0], x[p == 0]), 1)
        frame = accum_on - accum_off

        # Normalize and convert to BGR
        img = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Run YOLO inference every FRAME_SKIP frames
        if frame_index % FRAME_SKIP == 0:
            results = model.predict(source=rgb_frame, imgsz=1280, conf=0.6, stream=False, verbose=False) #4 good but can have many FP 6 conf is pretty good
            annotated = results[0].plot()
        else:
            annotated = rgb_frame

        # Display the result
        cv2.imshow("YOLOv8 Event Detection", annotated)

        print(f"\n--- Frame {frame_index} ---")
        print(f"  Inference time      : {time.time() - start_total:.4f}s")

        frame_index += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except StopIteration:
        print("✅ Event stream ended.")
        break
    except Exception as e:
        print(f"⚠️ Error at frame {frame_index}: {e}")
        frame_index += 1
        continue

# === Cleanup ===
cv2.destroyAllWindows()
print("🛑 Detection stopped.")

