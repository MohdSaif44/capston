import streamlit as st
import cv2
import numpy as np
import requests
import threading
import time
from ultralytics import YOLO

# --- CONFIGURATION ---
ESP_URL = "http://192.168.1.200"
MODEL_PATH = 'runs/detect/train/weights/best.pt'

# We lower global confidence slightly to ensuring we catch items, 
# then filter them strictly by class/location below.
CONF_THRESH = 0.80 

REQUIRED_FRAMES = 8
DOOR_OPEN_DURATION = 20  # Seconds to hold the door open

# --- SETUP UI ---
st.set_page_config(page_title="PPE Detection System", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Kiosk Mode & Split Layout
st.markdown("""
<style>
    /* 1. HIDE JUNK */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp { background-color: #0e1117; color: white; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }

    /* 2. LAYOUT CONTAINER (Holds the boxes) */
    .status-container {
        height: 600px; /* <--- MATCHES CAMERA HEIGHT */
        display: flex;
        flex-direction: column;
        gap: 20px; /* Space between Red and Yellow boxes */
    }

    /* 3. COMMON BOX STYLES */
    .box-base {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        border-radius: 20px;
        text-align: center;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
    }

    /* 4. TOP BLOCK (Status) */
    .top-box {
        flex: 1; /* Takes 50% height */
        width: 100%;
        color: white;
    }

    /* 5. BOTTOM BLOCK (Missing Items) */
    .bottom-box {
        flex: 1; /* Takes 50% height (equal to top) */
        width: 100%;
        background-color: #FFC107; /* WARNING YELLOW */
        color: white; /* White text for contrast */
        border: 5px solid #FF6F00;
        font-weight: 900;
        font-size: 28px;
        text-transform: uppercase;
    }

    /* 6. COLORS & THEMES */
    .theme-denied { background-color: #D32F2F; border: 5px solid #ff5252; }
    .theme-granted { background-color: #2E7D32; border: 5px solid #66bb6a; }
    .theme-warning { background-color: #F57C00; border: 5px solid #ffb74d; }

    /* 7. TEXT STYLES */
    .big-icon { font-size: 80px; margin-bottom: 10px; }
    .main-text { font-size: 45px; font-weight: 900; text-transform: uppercase; letter-spacing: 2px;}
    .sub-text { font-size: 20px; font-weight: 700; opacity: 0.9; }

</style>
""", unsafe_allow_html=True)

# --- ESP32 LOGIC ---
def send_esp_command(command):
    def _req():
        try: requests.get(f"{ESP_URL}/{command}", timeout=0.1)
        except: pass
    threading.Thread(target=_req, daemon=True).start()

# --- HELPER: DRAW HTML STATUS CARD ---
def render_status_card(placeholder, status_type, title, subtitle):
    icon_map = {
        'theme-denied': '🚫',
        'theme-granted': '✅',
        'theme-warning': '⚠️'
    }
    
    if status_type == 'theme-denied':
        html_code = f"""
<div class="status-container">
<div class="box-base top-box {status_type}">
<div class="big-icon">{icon_map.get(status_type)}</div>
<div class="main-text">{title}</div>
</div>
<div class="box-base bottom-box">
<div style="font-size: 60px; margin-bottom: 10px;">⚠️</div>
<div>{subtitle}</div>
</div>
</div>
"""
    else:
        html_code = f"""
<div class="status-container">
<div class="box-base top-box {status_type}" style="height: 100%;">
<div class="big-icon">{icon_map.get(status_type)}</div>
<div class="main-text">{title}</div>
<div class="sub-text" style="margin-top: 20px;">{subtitle}</div>
</div>
</div>
""" 
    placeholder.markdown(html_code, unsafe_allow_html=True)

# --- 🧠 LOGIC: SPATIAL OVERLAP CHECK ---
def check_overlap(box_inner, box_outer):
    """
    Returns True if the CENTER of the item (helmet/glove) is strictly inside the Person's box.
    """
    # Calculate Center of the Item
    ix = (box_inner[0] + box_inner[2]) / 2
    iy = (box_inner[1] + box_inner[3]) / 2
    
    # Get Person Boundaries
    px1, py1, px2, py2 = box_outer
    
    # Check if Item center is within Person boundaries
    return (px1 < ix < px2) and (py1 < iy < py2)

# --- MAIN APP LAYOUT ---
col1, col2 = st.columns([2.5, 1]) 

with col1:
    st.markdown("### 📷 Live Camera Feed")
    video_placeholder = st.empty()

with col2:
    st.markdown("### 🛡️ Status")
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

# --- LOGIC LOOP ---
def main():
    if 'model' not in st.session_state:
        st.session_state.model = YOLO(MODEL_PATH)
    
    if 'door_unlock_time' not in st.session_state:
        st.session_state.door_unlock_time = None

    cap = cv2.VideoCapture(0)
    
    while True:
        # 1. Timer Logic
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

        # 2. Capture Frame
        ret, frame = cap.read()
        if not ret:
            st.error("Camera disconnected")
            time.sleep(1)
            continue

        # 3. Decision Logic
        if is_holding_open:
            annotated_frame = frame
            render_status_card(status_placeholder, 'theme-granted', "ACCESS GRANTED", f"DOOR OPEN: {int(remaining_time)}s")
            progress_bar.progress(remaining_time / DOOR_OPEN_DURATION)

        else:
            # Run AI Inference
            results = st.session_state.model.predict(frame, conf=CONF_THRESH, verbose=False)
            result = results[0]
            
            # --- 🧠 SPATIAL LOGIC START ---
            persons = []
            helmets = []
            gloves = []
            
            # 1. Sort detections into lists
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = result.names[cls_id].lower()
                coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                
                if 'person' in name:
                    persons.append(coords)
                elif 'helmet' in name:
                    helmets.append(coords)
                elif 'glove' in name:
                    gloves.append(coords)
            
            annotated_frame = result.plot()
            
            # 2. Rules Engine
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
                # We have exactly 1 Person
                target_person = persons[0]
                
                valid_helmets = 0
                valid_gloves = 0
                
                # Check Overlap for Helmet
                for h in helmets:
                    if check_overlap(h, target_person):
                        valid_helmets += 1
                
                # Check Overlap for Gloves
                for g in gloves:
                    if check_overlap(g, target_person):
                        valid_gloves += 1
                
                # Validate Counts
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
                    # Note: Passing the missing list to the function for the Yellow Box
                    sub_text = f"{', '.join(missing)}"
            # --- 🧠 SPATIAL LOGIC END ---

            # Unlock Check
            if valid_frames >= REQUIRED_FRAMES:
                st.session_state.door_unlock_time = time.time()
                send_esp_command("open")
                render_status_card(status_placeholder, 'theme-granted', "ACCESS GRANTED", "DOOR UNLOCKED")
                progress_bar.progress(1.0)
            else:
                render_status_card(status_placeholder, status_type, main_text, sub_text)
                progress_bar.progress(valid_frames / REQUIRED_FRAMES)

        # 4. Display Video
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_frame, channels="RGB", width='stretch')

    cap.release()

if __name__ == "__main__":
    main()