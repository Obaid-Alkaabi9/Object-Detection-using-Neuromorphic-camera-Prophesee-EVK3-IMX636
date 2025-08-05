# 🧠 Event-Based Object Detection with YOLOv8

This project demonstrates how to use a **Neuromorphic (event-based) camera** to detect humans using the **YOLOv8** object detection model. Event-based cameras, unlike traditional frame-based cameras, capture changes in the scene asynchronously, enabling high-speed and low-power vision systems.

---

## 📌 Objectives

- Utilize a **neuromorphic camera** for human/person detection.
- Record event-based data using **Metavision Studio**.
- Convert event-based `.raw` recordings into images for YOLOv8 training.
- Train a custom YOLOv8 model and deploy it for **live stream detection**.

---

## 🛠️ Setup Instructions

### 1. Install Prophesee SDK & Metavision Studio

- Download and install the **Metavision SDK**:  
  [Metavision SDK Installation Guide](https://docs.prophesee.ai/stable/installation/index.html)

- Download and install **Metavision Studio** software:  
  [Metavision Studio Guide](https://docs.prophesee.ai/stable/metavision_studio/)

---

## 🎥 Data Collection

1. Open **Metavision Studio**.
2. Record a video using your neuromorphic camera.
3. The recorded video will be saved in `.raw` format.

---

## 🔄 Convert RAW to HDF5 (.h5)

Use the following command to convert the `.raw` file to `.h5` format:

```bash
python3 /usr/share/metavision/sdk/ml/python_samples/generate_hdf5/generate_hdf5.py \
    --input-raw /path/to/your_file.raw \
    --output-folder /path/to/output_dir \
    --delta-t 50000  # microseconds (e.g., 50ms)
```  
## 🖼️ Convert HDF5 to Images (JPG)
Run the script h5_to_img.py to convert the .h5 dataset to images: 
```bash   
python3 h5_to_img.py --input /path/to/your_file.h5 --output /path/to/images
```
## 🏷️ Annotate Images
- Use LabelImg or any annotation tool that supports YOLO format.

- Annotate the images and save labels in YOLO format (.txt).

- Organize your dataset into train, valid, and test splits.




## 📄 Prepare Data YAML

```bash
train: path/to/train/images
val: path/to/valid/images
test: path/to/test/images

nc: 1
names: ['person']
```
## 🏋️‍♂️ Train YOLOv8

```bash    
python3 train.py
```
After training, the best model will be saved as:
```bash
runs/detect/train/weights/best.pt
```
## ✅ Validate Model
### Use the following command to validate the trained model:
```bash
yolo task=detect mode=val \
    model=runs/detect/train/weights/best.pt \
    data=data.yaml \
    split=test
```
## 📡 Live Stream Detection
Run live detection using your neuromorphic camera:

```bash
python3 final_stream.py
```
## 📚 Notes
- Make sure all paths in scripts and YAML files are correctly set.

- You may need to adjust parameters such as delta-t or YOLO image size based on your application.

- Refer the "sample_output.mp4" for the sample output.

## 🎬 Sample Output

![Sample Output](sample_output.gif)

