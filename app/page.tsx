"use client";

import { useState, useRef, useEffect, type ChangeEvent } from "react";
import axios from "axios";

// --- Types ---
interface Detection {
  label: string;
  conf: number;
  box: [number, number, number, number];
}

interface ApiResponse {
  status: string;
  latency_ms: number;
  detections: Detection[];
}

export default function Dashboard() {
  // --- Core State ---
  const [inputType, setInputType] = useState<"image" | "video">("video");
  const [mode, setMode] = useState<"cpu" | "fpga">("fpga");
  
  // DEFAULT URL: Change this in the UI to your Ngrok URL!
  const [backendUrl, setBackendUrl] = useState("http://localhost:8000"); 
  const [logs, setLogs] = useState<string[]>([]);
  
  // --- Image State ---
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  
  // --- Video State ---
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState(false);
  
  // --- Telemetry State ---
  const [metrics, setMetrics] = useState({ latency: 0, fps: 0.0 });

  // --- Refs ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const addLog = (msg: string) => {
    setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString().split(' ')[0]}] ${msg}`]);
  };

  // 1. Webcam Setup
  useEffect(() => {
    async function loadDevices() {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter((device) => device.kind === "videoinput");
      setVideoDevices(videoInputs);
      if (videoInputs.length > 0) setSelectedDeviceId(videoInputs[0].deviceId);
    }
    loadDevices();
  }, []);

  useEffect(() => {
    if (inputType !== "video") return;
    async function setupCamera() {
      if (!navigator.mediaDevices?.getUserMedia) return;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: selectedDeviceId ? { deviceId: { exact: selectedDeviceId }, width: 640, height: 480 } : { width: 640, height: 480 },
        });
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (err) {
        addLog("[Error] Failed to access webcam.");
      }
    }
    setupCamera();

    return () => {
      const currentStream = videoRef.current?.srcObject as MediaStream | null;
      currentStream?.getTracks().forEach((track) => track.stop());
    };
  }, [selectedDeviceId, inputType]);

  // 2. Image Selection
  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      addLog("New image loaded into buffer.");
    }
  };

  // 3. Inference Logic
  const runInference = async (blobData: Blob, sourceElement: CanvasImageSource, isVideo: boolean = false) => {
    const formData = new FormData();
    formData.append("frame", blobData, "frame.jpg");

    try {
      const startTime = performance.now();
      const res = await axios.post<ApiResponse>(`${backendUrl}/process-frame`, formData, {
        headers: { 
          "Content-Type": "multipart/form-data",
          "ngrok-skip-browser-warning": "true" 
        }
      });
      
      const totalTimeMs = performance.now() - startTime;
      const currentFps = 1000 / totalTimeMs;

      setMetrics({
        latency: res.data.latency_ms,
        fps: parseFloat(currentFps.toFixed(1))
      });

      if (res.data.status === "success") {
        drawOutput(sourceElement, res.data.detections);
        
        // Smart Logging
        if (!isVideo) {
          addLog(`[Success] Inference complete! Found ${res.data.detections.length} objects.`);
        } else if (res.data.detections.length > 0) {
          // Only log video frames if an object is actually detected to avoid spam
          const labels = res.data.detections.map(d => d.label).join(", ");
          addLog(`[Stream] Detected: ${labels}`);
        }
      }
    } catch (error) {
      addLog("[Error] Inference failed. Check Board URL/Tunnel.");
      setIsProcessing(false);
    }
  };

  // 4. Drawing Logic (with Erase Fix & Edge Clipping Fix)
  const drawOutput = (source: CanvasImageSource, detections: Detection[]) => {
    const canvas = outputCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // FIX 1: Completely erase the previous image/frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw the new raw image/frame
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

    // Draw the boxes
    detections.forEach((det) => {
      const [x, y, w, h] = det.box;
      const labelText = `${det.label} ${Math.round(det.conf * 100)}%`;
      
      ctx.strokeStyle = "#00FF00";
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      ctx.font = "bold 14px monospace";
      const textWidth = ctx.measureText(labelText).width + 10;
      const labelHeight = 22;

      // FIX 2: Smart positioning to prevent top-edge clipping
      let labelY = y - labelHeight;
      let textY = y - 6;

      if (labelY < 0) {
        labelY = y;
        textY = y + 15;
      }

      ctx.fillStyle = "#00FF00";
      ctx.fillRect(x, labelY, textWidth, labelHeight);
      
      ctx.fillStyle = "#000000";
      ctx.fillText(labelText, x + 5, textY);
    });
  };

  // 5. Video Loop Trigger (Optimized & Zombie-Proof)
  useEffect(() => {
    let isActive = true; // Instantly kills the loop when unmounted/stopped
    let timeoutId: NodeJS.Timeout;

    const captureLoop = async () => {
      // If user clicked stop, immediately halt execution
      if (!isActive || !videoRef.current || !hiddenCanvasRef.current) return;
      
      const video = videoRef.current;
      const canvas = hiddenCanvasRef.current;
      const ctx = canvas.getContext("2d");
      
      if (ctx && video.videoWidth > 0) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob(async (blob) => {
          // Double check isActive before making the heavy network call
          if (blob && isActive) {
            await runInference(blob, video, true);
          }
          // Only schedule the next frame if the user hasn't hit stop
          if (isActive) {
            timeoutId = setTimeout(captureLoop, 100); // Backed off to 100ms so board can breathe
          }
        }, "image/jpeg", 0.4); 
      } else {
        if (isActive) timeoutId = setTimeout(captureLoop, 100);
      }
    };

    if (isProcessing && inputType === "video") {
      addLog("[System] Started live video stream to FPGA...");
      captureLoop();
    }

    // Cleanup function: runs the millisecond you click "Stop"
    return () => {
      isActive = false; 
      clearTimeout(timeoutId);
    };
  }, [isProcessing, inputType, backendUrl]);

  // 6. Handle Single Image Processing
  const processSingleImage = async () => {
    if (!selectedImage || !imgRef.current) return;
    addLog(`[System] Sending static image to FPGA core...`);
    
    // Convert the image file to a blob for the API
    const response = await fetch(previewUrl!);
    const blob = await response.blob();
    
    runInference(blob, imgRef.current, false);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6 font-mono">
      <header className="mb-8 flex justify-between items-end border-b border-gray-800 pb-4">
        <div>
          <h1 className="text-3xl font-bold text-blue-500 tracking-tighter">Bharat-AI-Hackathon</h1>
          <p className="text-gray-400 text-xs mt-1 tracking-widest">HW/SW CO-DESIGN ACCELERATOR</p>
        </div>
        <div className="flex items-center gap-2">
           <div className="flex flex-col text-right mr-2">
              <span className="text-[10px] text-gray-500 uppercase mb-3">Status</span>
              <span className="text-xs font-bold text-green-400">BOARD ONLINE</span>
           </div>
           <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full mt-3 w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 mt-3 w-3 bg-green-500"></span>
            </span>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* LEFT COLUMN: Controls */}
        <div className="lg:col-span-3 space-y-6">
            <div className="bg-gray-800 p-5 rounded-lg border border-gray-700 shadow-xl">
                <h2 className="text-xs font-bold text-gray-500 mb-4 uppercase tracking-widest">Configuration</h2>
                
                {/* BACKEND URL INPUT RESTORED */}
                <div className="bg-gray-900 p-2 mb-4 rounded border border-gray-700">
                    <label className="text-[10px] text-gray-500 block mb-1">BOARD / NGROK URL:</label>
                    <input 
                        type="text" 
                        value={backendUrl} 
                        onChange={(e) => setBackendUrl(e.target.value)}
                        className="bg-black text-xs text-green-400 p-1.5 rounded w-full border border-gray-600 outline-none focus:border-blue-500"
                        placeholder="https://your-ngrok-url.app"
                    />
                </div>

                <div className="flex bg-gray-900 rounded p-1 mb-4 border border-gray-700">
                    <button onClick={() => setInputType("video")} className={`flex-1 py-2 text-xs font-bold rounded transition-colors ${inputType === "video" ? "bg-gray-700 text-white" : "text-gray-500 hover:text-gray-300"}`}>Live Video</button>
                    <button onClick={() => setInputType("image")} className={`flex-1 py-2 text-xs font-bold rounded transition-colors ${inputType === "image" ? "bg-gray-700 text-white" : "text-gray-500 hover:text-gray-300"}`}>Static Image</button>
                </div>

                {inputType === "video" ? (
                  <div className="mb-6">
                    <label className="text-[10px] text-gray-500 block mb-1">SELECT WEBCAM:</label>
                    <select value={selectedDeviceId} onChange={(e) => setSelectedDeviceId(e.target.value)} className="w-full bg-black text-xs text-white p-2 rounded border border-gray-600 outline-none">
                      {videoDevices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>)}
                    </select>
                  </div>
                ) : (
                  <div className="mb-6">
                     <label className="text-[10px] text-gray-500 block mb-1">UPLOAD FRAME:</label>
                     <input type="file" accept="image/*" onChange={handleFileChange} className="block w-full text-xs text-gray-400 file:mr-4 file:py-1.5 file:px-3 file:rounded file:border-0 file:bg-gray-700 file:text-white file:cursor-pointer hover:file:bg-gray-600" />
                  </div>
                )}

                <button
                  onClick={inputType === "video" ? () => setIsProcessing(!isProcessing) : processSingleImage}
                  disabled={inputType === "image" && !selectedImage}
                  className={`w-full py-3 rounded font-bold uppercase tracking-widest text-sm transition-all shadow-lg ${
                    inputType === "video" && isProcessing 
                      ? "bg-red-600 hover:bg-red-500 text-white" 
                      : "bg-blue-600 hover:bg-blue-500 text-white"
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  {inputType === "video" ? (isProcessing ? "Stop Stream" : "Start Stream") : "Execute Inference"}
                </button>
            </div>

            {/* Metrics */}
            <div className="bg-gray-800 p-5 rounded-lg border border-gray-700 shadow-xl">
                <h2 className="text-xs font-bold text-gray-500 mb-4 uppercase tracking-widest">Telemetry</h2>
                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-900 p-3 rounded border border-gray-700 flex flex-col justify-center items-center">
                        <p className="text-[10px] text-gray-500 uppercase mb-1">Hardware Latency</p>
                        <p className="text-2xl font-bold text-white">{metrics.latency}<span className="text-xs ml-1 text-gray-500">ms</span></p>
                    </div>
                    <div className="bg-gray-900 p-3 rounded border border-gray-700 flex flex-col justify-center items-center">
                        <p className="text-[10px] text-gray-500 uppercase mb-1">System Throughput</p>
                        <p className="text-2xl font-bold text-green-400">{metrics.fps}<span className="text-xs ml-1 text-gray-500">FPS</span></p>
                    </div>
                </div>
            </div>

            {/* Logs */}
            <div className="bg-black p-4 rounded-lg border border-gray-800 font-mono text-[10px] h-48 overflow-y-auto custom-scrollbar shadow-inner">
                <p className="text-gray-500 mb-2 border-b border-gray-800 pb-1 sticky top-0 bg-black">System Logs</p>
                {logs.length === 0 && <p className="text-gray-700 italic mt-2">Awaiting connection...</p>}
                {logs.map((log, i) => <p key={i} className="text-green-500 mb-1 leading-relaxed">{log}</p>)}
                <div ref={logsEndRef} />
            </div>
        </div>

        {/* RIGHT COLUMN: Side-by-Side Visualizers */}
        <div className="lg:col-span-9 grid grid-cols-2 gap-4">
          
          {/* Input View */}
          <div className="bg-gray-800 rounded-lg p-3 border border-gray-700 flex flex-col shadow-xl">
            <div className="flex justify-between items-center mb-2">
               <h3 className="text-xs text-gray-400 uppercase tracking-widest font-bold">Raw Input Buffer</h3>
               <span className="text-[10px] bg-gray-700 px-2 py-0.5 rounded text-gray-300">LAPTOP</span>
            </div>
            <div className="relative flex-1 bg-black rounded border border-gray-900 flex items-center justify-center overflow-hidden aspect-video shadow-inner">
              {inputType === "video" ? (
                <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-contain" />
              ) : previewUrl ? (
                <img ref={imgRef} src={previewUrl} alt="Upload" className="w-full h-full object-contain" />
              ) : (
                <div className="text-center">
                   <p className="text-gray-700 text-2xl mb-2">+</p>
                   <p className="text-gray-600 text-xs font-mono">No Source Detected</p>
                </div>
              )}
            </div>
          </div>

          {/* Output View */}
          <div className="bg-gray-800 rounded-lg p-3 border border-gray-700 flex flex-col shadow-xl">
             <div className="flex justify-between items-center mb-2">
               <h3 className="text-xs text-blue-400 uppercase tracking-widest font-bold">FPGA Inference Engine</h3>
               <span className="text-[10px] bg-blue-900/50 text-blue-300 border border-blue-800 px-2 py-0.5 rounded animate-pulse">PYNQ-Z2</span>
            </div>
            <div className="relative flex-1 bg-black rounded border border-gray-900 flex items-center justify-center overflow-hidden aspect-video shadow-inner">
              <canvas ref={outputCanvasRef} width={640} height={480} className="w-full h-full object-contain" />
            </div>
          </div>

        </div>
      </div>
      
      {/* Hidden canvas for extracting video frames */}
      <canvas ref={hiddenCanvasRef} width={640} height={480} className="hidden" />
    </div>
  );
}