# PYNQ board details to run server code

ssh into the board: "ssh -X -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa root@192.168.1.224" password: xilinx

code sits in /board_ui/server.py

To run the code: "python3 server.py"

Any changes in the board server code will require a restart of the board server.

# For local machine to run the UI

npm.cmd run dev

First ssh into the board and run the board server code

In new terminal, run "ssh -L 8000:127.0.0.1:8000 -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa root@192.168.1.224" to tunnel the board to your local machine, password: xilinx

# To run the UI on internet

First ssh into the board and run the board server code

In new terminal, run "ssh -L 8000:127.0.0.1:8000 -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa root@192.168.1.224" to tunnel the board to your local machine, password: xilinx

In another new terminal, run "ngrok http 8000" to get the https:// link from ngrok and paste it in the UI board URL field.

# 🚀 Bharat-AI-Hackathon: Edge AI HW/SW Co-Design Accelerator

An end-to-end, real-time Edge AI inference pipeline built for the Bharat-AI Hackathon. This project demonstrates true hardware/software co-design by bridging a modern Next.js web frontend with a physical Xilinx PYNQ-Z2 FPGA board. 

It streams live video frames from a web browser over the internet (via Ngrok/SSH tunnels) directly into the FPGA's Deep Learning Processor Unit (DPU), parses the C++ hardware output, and renders high-speed bounding box overlays with real-time hardware telemetry.

[Image of Edge AI architecture bridging a React frontend to a Xilinx PYNQ FPGA backend via Flask and SSH tunnel]

## ✨ Key Features

* **Real-Time Video & Image Inference**: Seamlessly toggle between uploading static images or streaming live webcam feeds directly to the hardware.
* **FPGA DPU Acceleration**: Utilizes a custom C++ executable (`yolo_image`) running on the Xilinx PYNQ-Z2 board for hardware-accelerated object detection.
* **Zero-Overhead Data Transfer**: The frontend handles frame extraction and JPEG compression (0.4 quality) via HTML5 Canvas to ensure network throughput doesn't bottleneck the 450ms hardware latency.
* **Hardware Thread-Locking**: Implements Python `threading.Lock()` to manage DPU memory constraints, aggressively dropping frames if the board is busy to prevent `[DNNDK] Failure of DPU memory space used out` crashes.
* **Smart Telemetry & UI**: Features a professional "hacker-style" dashboard displaying live hardware latency, system throughput (FPS), smart edge-clipping prevention for bounding boxes, and an auto-scrolling system log.
* **Global Tunneling**: Integrated Ngrok and legacy RSA SSH tunneling support to expose the local edge device to the public internet for remote demos.

## 🛠️ Tech Stack

**Frontend (Software)**
* Next.js (App Router) & React
* Axios (HTTP Client)
* HTML5 Canvas API (Frame extraction & Box rendering)

**Backend & Hardware (Edge)**
* Xilinx PYNQ-Z2 Board (Zynq-7000 SoC)
* Python 3.5 (Flask, Flask-CORS)
* C++ (OpenCV, Xilinx DNNDK/Vitis AI)
* Ngrok / SSH (Port Forwarding)

---

## ⚙️ Architecture & Data Flow

1. **Capture**: React extracts a frame from the `<video>` stream using a hidden `<canvas>` and compresses it to a JPEG Blob.
2. **Transport**: Axios POSTs the blob to a dynamic Ngrok URL or local SSH tunnel.
3. **Bridge**: A Python 3.5 Flask server on the PYNQ board receives the frame, writes it to the SD card, and requests the DPU thread lock.
4. **Execution**: Flask triggers the compiled C++ executable (`./yolo_image`) via `subprocess`.
5. **Parse**: Python intercepts standard output (`stdout`), using Regular Expressions to extract `xmin, ymin, xmax, ymax`, class labels, and `DPU task time`.
6. **Render**: The frontend receives the JSON payload and draws smart, non-clipping bounding boxes on a transparent overlay canvas.

---

## 🚀 Installation & Setup

### 1. Board Setup (Xilinx PYNQ-Z2)
SSH into the board (note the legacy RSA key flags required for older PYNQ images):
```bash
ssh -L 8000:127.0.0.1:8000 -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa root@<BOARD_IP>

cd /home/root/yolo_pynqz2
sudo python3 -m pip install flask flask-cors --trusted-host pypi.org --trusted-host files.pythonhosted.org

## Run the backend server
python3 server.py

## Run the frontend
npm install
npm run dev

## Public access
ngrok http 8000