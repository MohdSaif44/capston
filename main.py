import streamlit as st
import cv2
import numpy as np
import requests
import threading
import time
from ultralytics import YOLO

ESP_URL = "http://192.168.1.200"
MODEL_PATH = 'runs/detect/train/weights/best.pt'
CONF_THRESH = 0.80 
REQUIRED_FRAMES = 8
DOOR_OPEN_DURATION = 20  

st.set_page_config(page_title="PPE Detection System", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM GUI CSS ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp { background-color: #0e1117; color: white; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .status-container { height: 600px; display: flex; flex-direction: column; gap: 20px; }
    .box-base { display: flex; flex-direction: column; justify-content: center; align-items: center; border-radius: 20px; text-align: center; padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); transition: all 0.3s ease; }
    .top-box { flex: 1; width: 100%; color: white; }
    .bottom-box { flex: 1; width: 100%; background-color: #FFC107; color: white; border: 5px solid #FF6F00; font-weight: 900; font-size: 28px; text-transform: uppercase; }
    .theme-denied { background-color: #D32F2F; border: 5px solid #ff5252; }
    .theme-granted { background-color: #2E7D32; border: 5px solid #66bb6a; }
    .theme-warning { background-color: #F57C00; border: 5px solid #ffb74d; }
    .big-icon { font-size: 80px; margin-bottom: 10px; }
    .main-text { font-size: 45px; font-weight: 900; text-transform: uppercase; letter-spacing: 2px;}
    .sub-text { font-size: 20px; font-weight: 700; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


def send_esp_command(command):
    def _req():
        try: requests.get(f"{ESP_URL}/{command}", timeout=0.1)
        except: pass
    threading.Thread(target=_req, daemon=True).start()

def render_status_card(placeholder, status_type, title, subtitle):
    icon_map = {'theme-denied': '🚫', 'theme-granted': '✅', 'theme-warning': '⚠️'}
    if status_type == 'theme-denied':
        html_code = f"""<div class="status-container"><div class="box-base top-box {status_type}"><div class="big-icon">{icon_map.get(status_type)}</div><div class="main-text">{title}</div></div><div class="box-base bottom-box"><div style="font-size: 60px; margin-bottom: 10px;">⚠️</div><div>{subtitle}</div></div></div>"""
    else:
        html_code = f"""<div class="status-container"><div class="box-base top-box {status_type}" style="height: 100%;"><div class="big-icon">{icon_map.get(status_type)}</div><div class="main-text">{title}</div><div class="sub-text" style="margin-top: 20px;">{subtitle}</div></div></div>""" 
    placeholder.markdown(html_code, unsafe_allow_html=True)

def check_overlap(box_inner, box_outer):
    ix = (box_inner[0] + box_inner[2]) / 2
    iy = (box_inner[1] + box_inner[3]) / 2
    px1, py1, px2, py2 = box_outer
    return (px1 < ix < px2) and (py1 < iy < py2)

def is_blue_glove(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return False
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_colored = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([180, 255, 255]))
    mask_blue = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([140, 255, 255]))
    
    count_valid = cv2.countNonZero(mask_colored)
    count_blue = cv2.countNonZero(mask_blue)
    if count_valid < 50: return False
    
    return (count_blue / count_valid) > 0.70

def is_yellow_helmet(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_colored = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([180, 255, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([35, 255, 255]))
    
    count_valid = cv2.countNonZero(mask_colored)
    count_yellow = cv2.countNonZero(mask_yellow)
    if count_valid < 50: return False
    
    return (count_yellow / count_valid) > 0.40

def check_helmet_on_head(h_box, p_box):
    """
    Validates if the helmet is:
    1. Located in the top 20% of the Person box (Head area).
    2. Sized correctly (not tiny, not huge) relative to the Person width.
    """
    hx1, hy1, hx2, hy2 = h_box
    px1, py1, px2, py2 = p_box
    
    h_center_y = (hy1 + hy2) / 2
    h_width = hx2 - hx1
    p_height = py2 - py1
    p_width = px2 - px1
    

    head_limit_y = py1 + (p_height * 0.20)
    
    if h_center_y > head_limit_y:
        return False 

    width_ratio = h_width / p_width
    
    if width_ratio < 0.35: 
        return False 
    if width_ratio > 0.45: 
        return False 
        
    return True

# --- MAIN APP ---
col1, col2 = st.columns([2.5, 1]) 
with col1:
    st.markdown("### 📷 Live Camera Feed")
    video_placeholder = st.empty()
with col2:
    st.markdown("### 🛡️ Status")
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

def main():
    if 'model' not in st.session_state: st.session_state.model = YOLO(MODEL_PATH)
    if 'door_unlock_time' not in st.session_state: st.session_state.door_unlock_time = None

    cap = cv2.VideoCapture(0) 
    
    while True:
        current_time = time.time()
        is_holding_open = False
        remaining_time = 0

        if st.session_state.door_unlock_time is not None:
            elapsed = current_time - st.session_state.door_unlock_time
            if elapsed < DOOR_OPEN_DURATION:
                is_holding_open = True
                remaining_time = DOOR_OPEN_DURATION - elapsed
            else:
                st.session_state.door_unlock_time = None
                send_esp_command("close")
                valid_frames = 0 

        ret, frame = cap.read()
        if not ret:
            st.error("Camera disconnected")
            time.sleep(1)
            continue

        if is_holding_open:
            annotated_frame = frame
            render_status_card(status_placeholder, 'theme-granted', "ACCESS GRANTED", f"DOOR OPEN: {int(remaining_time)}s")
            progress_bar.progress(remaining_time / DOOR_OPEN_DURATION)
        else:
            results = st.session_state.model.predict(frame, conf=CONF_THRESH, verbose=False)
            result = results[0]

            main_person_box = None
            max_area = 0
            
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = result.names[cls_id].lower()
                if 'person' in name:
                    coords = box.xyxy[0].tolist()
                    area = (coords[2]-coords[0]) * (coords[3]-coords[1])
                    if area > max_area:
                        max_area = area
                        main_person_box = coords

            valid_indices = []
            
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                name = result.names[cls_id].lower()
                coords = box.xyxy[0].tolist()
                
                if 'glove' in name:
                    if is_blue_glove(frame, coords):
                        valid_indices.append(i)
                
                elif 'helmet' in name:
                    if is_yellow_helmet(frame, coords):
                        if main_person_box:
                            if check_helmet_on_head(coords, main_person_box):
                                valid_indices.append(i)
                        else:
                            pass 
                            
                elif 'person' in name:
                    valid_indices.append(i)

            result.boxes = result.boxes[valid_indices]
            
            persons = []
            helmets = []
            gloves = []
            
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = result.names[cls_id].lower()
                coords = box.xyxy[0].tolist()
                
                if 'person' in name: persons.append(coords)
                elif 'helmet' in name: helmets.append(coords)
                elif 'glove' in name: gloves.append(coords)
            
            annotated_frame = result.plot()
            
            missing = []
            status_type = 'theme-denied'
            main_text = "ACCESS DENIED"
            sub_text = ""

            if len(persons) == 0:
                sub_text = "NO WORKER DETECTED"
                valid_frames = 0
            elif len(persons) > 1:
                status_type = 'theme-warning'
                main_text = "ONE AT A TIME"
                sub_text = "CROWD DETECTED"
                valid_frames = 0
            else:
                target_person = persons[0]
                valid_helmets = 0
                valid_gloves = 0
                
                for h in helmets:
                    if check_overlap(h, target_person): valid_helmets += 1
                for g in gloves:
                    if check_overlap(g, target_person): valid_gloves += 1
                
                if valid_helmets < 1: missing.append("HELMET")
                if valid_gloves < 2: missing.append("GLOVES")
                
                if not missing:
                    valid_frames += 1
                    status_type = 'theme-warning'
                    main_text = "VERIFYING..."
                    sub_text = "HOLD POSITION"
                else:
                    valid_frames = 0
                    status_type = 'theme-denied'
                    main_text = "ACCESS DENIED"
                    sub_text = f"{', '.join(missing)}"
                    
            if valid_frames >= REQUIRED_FRAMES:
                st.session_state.door_unlock_time = time.time()
                send_esp_command("open")
                render_status_card(status_placeholder, 'theme-granted', "ACCESS GRANTED", "DOOR UNLOCKED")
                progress_bar.progress(1.0)
            else:
                render_status_card(status_placeholder, status_type, main_text, sub_text)
                progress_bar.progress(valid_frames / REQUIRED_FRAMES)

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_frame, channels="RGB", width='stretch')

    cap.release()

if __name__ == "__main__":
    main()