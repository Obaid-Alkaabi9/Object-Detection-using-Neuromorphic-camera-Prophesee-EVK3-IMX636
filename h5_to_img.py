import h5py, numpy as np, cv2, os

input_h5 = "/home/emad/projects/INTERSHIP/development/Dataset_Human/output_of_raw/Human_Mycamera.h5"
output_dir = "/home/emad/projects/INTERSHIP/development/Dataset_Human/images_for_labeling"
os.makedirs(output_dir, exist_ok=True)

with h5py.File(input_h5, "r") as f:
    data = f["data"]
    for i in range(data.shape[0]):
        # Combine ON/OFF polarity channels into a single frame
        frame = data[i][0] - data[i][1]  # you can also try + instead of -
        img = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(f"{output_dir}/frame_{i:05d}.jpg", img)

