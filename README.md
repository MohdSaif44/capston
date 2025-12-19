# AI-Powered PPE Compliance & Access Control System

**A real-time Computer Vision system that enforces safety gear compliance before granting physical access.**

This project integrates **Deep Learning (YOLOv11)** with **Heuristic Image Processing (OpenCV)** to create a robust safety checkpoint. It verifies not just the *presence* of safety gear, but its *correct usage* (e.g., helmet strictly on the head, specific gear colors) and triggers an IoT-enabled door lock via an ESP32 microcontroller upon compliance.

---

## Key Features

* **Real-Time Detection:** Utilizes **YOLOv11** for high-speed detection of Persons, Helmets, and Gloves.
* **Advanced Logic Validation (Anti-Spoofing):**
    * **Spatial Association:** Ensures PPE is detected strictly *inside* the worker's bounding box (ignores gear sitting on tables).
    * **Geometric Constraints:** Validates that the helmet is positioned in the **top 20%** of the person's frame (Head Area) and matches realistic shoulder-width ratios to prevent background errors.
    * **Color Authentication:** Uses **HSV Color Thresholding** to enforce specific gear standards (e.g., **Yellow Helmets** and **Blue Gloves**), automatically rejecting skin tones or incorrect equipment.
* **IoT Access Control:** Sends asynchronous HTTP commands to an **ESP32** to trigger relays/solenoid locks when compliance is met.
* **Crowd Management:** Enforces a strictly "One Person at a Time" rule to prevent tailgating.
* **Kiosk Dashboard:** A custom-styled **Streamlit** interface with high-contrast visual feedback (Red/Green/Yellow states) designed for industrial monitors.

---

## Tech Stack

* **Language:** Python 3.9+
* **AI/ML:** Ultralytics YOLOv8
* **Computer Vision:** OpenCV (`cv2`), NumPy
* **Frontend:** Streamlit (Custom CSS)
* **IoT/Hardware:** ESP32 (Wi-Fi), Relay Module, 12V Solenoid Lock/Buzzer
* **Concurrency:** Python Threading (for non-blocking Network I/O)

---

## ⚙️ Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/MohdSaif44/capston.git
    cd ppe-detection-system
    ```

2.  **Install Dependencies**
    ```bash
    pip install ultralytics streamlit opencv-python numpy requests
    ```

3.  **Setup the Model**
    * Place your trained YOLOv11 weights file in the directory: `runs/detect/train/weights/best.pt`.
    * *Note: If you do not have a custom model, the system will attempt to download `yolov11n.pt`, but accurate class detection for PPE requires custom training.*

---

## 🖥️ Usage

1.  **Connect your Camera** (Webcam or USB Camera).
2.  **Connect the ESP32** to the same Wi-Fi network as the host PC.
3.  **Run the Application**:
    ```bash
    streamlit run main.py
    ```
4.  The application will launch in your default web browser. Press `F11` for full-screen Kiosk mode.

---

## 🔧 Configuration

You can adjust key parameters at the top of `main.py` to fit your specific environment:

```python
# Network & Hardware
ESP_URL = "[http://192.168.1.200](http://192.168.1.200)"  # IP Address of your ESP32

# AI Sensitivity
CONF_THRESH = 0.80      # Minimum confidence for YOLO detection
REQUIRED_FRAMES = 8     # Consecutive frames of compliance required to unlock (Stability check)
DOOR_OPEN_DURATION = 20 # How long the door stays unlocked (seconds)
