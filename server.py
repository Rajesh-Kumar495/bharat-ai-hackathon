## This is the main server code for the YOLOv5 FPGA Inference API.
## It receives frames from the client, runs the C++ executable to perform inference on the FPGA

## This runs directly on the PYNQ-Z2 board. Make sure to set the WORK_DIR and EXECUTABLE variables correctly before running.

import subprocess
import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Lock

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
# PUT THE REAL PATH YOU FOUND HERE (e.g., "/home/root/yolo-pynqz2")
WORK_DIR = "/home/root/yolo_pynqz2"
EXECUTABLE = "./yolo_image"

dpu_lock = Lock()

# --- CORS FIX FOR NGROK & NEXT.JS ---
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,ngrok-skip-browser-warning,x-hackathon-token')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/process-frame', methods=['POST', 'OPTIONS'])
def process_frame_api():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    # 1. Check if the board is busy!
    if not dpu_lock.acquire(blocking=False):
        # The lock is already held by another frame. Drop this frame!
        return jsonify({"status": "skipped", "error": "FPGA busy, frame dropped"}), 429

    try:
        # 1. Save the incoming frame to a temporary file
        if 'frame' not in request.files:
            return jsonify({"error": "No frame received"}), 400

        frame = request.files['frame']
        input_path = os.path.join(WORK_DIR, "temp_input.jpg")
        frame.save(input_path)

        # 2. Run the C++ Executable (PYTHON 3.5 WAY)
        result = subprocess.run(
            [EXECUTABLE, "temp_input.jpg"],
            cwd=WORK_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        stdout_text = result.stdout
        detections = []

        # 3. Parse the Bounding Boxes
        box_pattern = r"xmin\s+([\d\.]+)\s+ymin\s+([\d\.]+)\s+xmax\s+([\d\.]+)\s+ymax\s+([\d\.]+)"
        raw_boxes = re.findall(box_pattern, stdout_text)

        class_pattern = r"Class name:\s+([a-zA-Z0-9_]+)"
        raw_classes = re.findall(class_pattern, stdout_text)

        for i in range(min(len(raw_classes), len(raw_boxes))):
            cls_name = raw_classes[i]
            xmin, ymin, xmax, ymax = map(float, raw_boxes[i])

            width = xmax - xmin
            height = ymax - ymin

            detections.append({
                "label": cls_name.capitalize(),
                "conf": 1.0,
                "box": [int(xmin), int(ymin), int(width), int(height)]
            })

        # 4. Parse the Hardware Latency
        dpu_time = 0.0
        time_match = re.search(r"___DPU task time:\s+([\d\.]+)", stdout_text)
        if time_match:
            dpu_time = float(time_match.group(1)) * 1000

        # Check for crash without f-strings
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else stdout_text
            raise Exception("C++ Execution Failed: " + str(error_msg))

        return jsonify({
            "status": "success",
            "latency_ms": round(dpu_time, 2) if dpu_time > 0 else 0,
            "detections": detections
        })

    except Exception as e:
        # Log error without f-strings
        print("CRITICAL ERROR: " + str(e))
        return jsonify({"error": str(e)}), 500

    finally:
        # ALWAYS release the lock when done, even if it crashes
        dpu_lock.release()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)