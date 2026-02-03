"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";

// --- Types ---
interface Detection {
  label: string;
  conf: number;
  box: [number, number, number, number];
}

interface ApiResponse {
  status: string;
  mode: string; // Added mode to response
  device: string;
  metrics: {
    latency_ms: number;
    fps: number;
    power_usage: string;
  };
  detections: Detection[];
  logs?: string[]; // Added optional logs
}

export default function Home() {
  // --- State ---
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<ApiResponse | null>(null);
  const [backendUrl, setBackendUrl] = useState("http://localhost:8000"); // Default
  
  // New State for "Pro" Features
  const [mode, setMode] = useState<"cpu" | "fpga">("fpga"); // Default to FPGA
  const [logs, setLogs] = useState<string[]>([]); 

  // --- Refs ---
  const imgRef = useRef<HTMLImageElement>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs to bottom whenever they change
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // 1. Handle File Selection
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setData(null);
      setLogs([]); // Clear logs for new image
    }
  };

  // 2. Upload to Python Backend
  const handleUpload = async () => {
    if (!selectedImage) return;
    setLoading(true);
    
    // Add local log immediately
    setLogs(prev => [...prev, `[System] Initializing transfer to ${mode.toUpperCase()} core...`]);

    const formData = new FormData();
    formData.append("file", selectedImage);
    formData.append("mode", mode); // <--- SEND THE MODE (CPU/FPGA)

    try {
      const res = await axios.post<ApiResponse>(`${backendUrl}/process-image`, formData, {
        headers: { 
          "Content-Type": "multipart/form-data",
          "ngrok-skip-browser-warning": "true",
          "x-hackathon-token": "bharat-ai-soc@26*"
        },
      });
      
      setData(res.data);
      
      // Append backend logs if they exist
      if (res.data.logs) {
        setLogs(prev => [...prev, ...res.data.logs!]);
      }

    } catch (error) {
      console.error("Upload failed:", error);
      setLogs(prev => [...prev, "[Error] Connection to Zynq Board failed."]);
      alert("Failed to connect to the board server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6 font-mono">
      {/* Header */}
      <header className="mb-8 flex justify-between items-end border-b border-gray-800 pb-4">
        <div>
          <h1 className="text-3xl font-bold text-blue-500 tracking-tighter">
            Bharat-AI-Hackathon
          </h1>
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
        
        {/* LEFT COLUMN: Controls & Logs (4 Columns) */}
        <div className="lg:col-span-4 space-y-6">
            
            {/* 1. Configuration Card */}
            <div className="bg-gray-800 p-5 rounded-lg border border-gray-700 shadow-xl">
                <h2 className="text-xs font-bold text-gray-500 mb-4 uppercase tracking-widest">Configuration</h2>
                
                <div className="bg-gray-800 p-2 mb-4 rounded border border-gray-700 flex gap-2 items-center">
                    <span className="text-xs text-gray-400">BOARD URL:</span>
                    <input 
                        type="text" 
                        value={backendUrl} 
                        onChange={(e) => setBackendUrl(e.target.value)}
                        className="bg-gray-900 text-xs text-white p-1 rounded flex-1 border border-gray-600"
                    />
                </div>

                {/* Mode Switcher */}
                <div className="flex bg-gray-900 rounded p-1 mb-6 border border-gray-700">
                    <button 
                        onClick={() => setMode("cpu")}
                        className={`flex-1 py-2 text-xs font-bold rounded transition-colors ${mode === "cpu" ? "bg-green-700 text-white" : "text-gray-500 hover:text-gray-300"}`}
                    >
                        ARM Processor
                    </button>
                    <button 
                        onClick={() => setMode("fpga")}
                        className={`flex-1 py-2 text-xs font-bold rounded transition-all ${mode === "fpga" ? "bg-blue-600 text-white shadow-[0_0_15px_rgba(37,99,235,0.5)]" : "text-gray-500 hover:text-gray-300"}`}
                    >
                        FPGA FABRIC
                    </button>
                </div>

                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleFileChange}
                  className="block w-full text-xs text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-xs file:font-bold file:bg-gray-700 file:text-white hover:file:bg-gray-600 mb-4 file:cursor-pointer"
                />

                <button
                  onClick={handleUpload}
                  disabled={!selectedImage || loading}
                  className={`w-full py-3 rounded font-bold uppercase cursor-pointer tracking-widest text-sm transition-all ${
                    loading 
                      ? "bg-gray-700 text-gray-500 cursor-not-allowed" 
                      : "bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 shadow-lg text-white"
                  }`}
                >
                  {loading ? "Processing..." : "Execute Inference"}
                </button>
            </div>

            {/* 2. Metrics Card */}
            {data && (
                <div className="bg-gray-800 p-5 rounded-lg border border-gray-700 shadow-xl animate-fade-in">
                    <div className="flex justify-between items-center mb-4">
                        <h2 className="text-xs font-bold text-gray-500 uppercase tracking-widest">Telemetry</h2>
                        <span className="text-[10px] bg-gray-700 px-2 py-0.5 rounded text-gray-300">{data.mode}</span>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                        <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
                            <p className="text-[10px] text-gray-500 uppercase">Latency</p>
                            <p className="text-2xl font-bold text-white">{data.metrics.latency_ms}<span className="text-sm text-gray-500 ml-1">ms</span></p>
                        </div>
                        <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
                            <p className="text-[10px] text-gray-500 uppercase">Throughput</p>
                            <p className="text-2xl font-bold text-green-400">{data.metrics.fps}<span className="text-sm text-gray-500 ml-1">FPS</span></p>
                        </div>
                         <div className="bg-gray-900/50 p-3 rounded border border-gray-700 col-span-2 flex justify-between items-center">
                            <div>
                                <p className="text-[10px] text-gray-500 uppercase">Power Est.</p>
                                <p className="text-xl font-bold text-blue-300">{data.metrics.power_usage}</p>
                            </div>
                            <div className="text-right">
                                <p className="text-[10px] text-gray-500 uppercase">Efficiency</p>
                                <p className="text-xs text-gray-400">
                                    {mode === 'fpga' ? 'High Performance' : 'Standard'}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* 3. System Logs (The "Hacker" visual) */}
            <div className="bg-black p-4 rounded-lg border border-gray-800 font-mono text-[10px] h-48 overflow-y-auto custom-scrollbar shadow-inner">
                <p className="text-gray-500 mb-2 border-b border-gray-800 pb-1">System Logs</p>
                {logs.length === 0 && <p className="text-gray-700 italic">Waiting for input stream...</p>}
                {logs.map((log, i) => (
                    <p key={i} className="text-green-500 mb-1 font-mono">
                        <span className="text-gray-600 mr-2">[{new Date().toLocaleTimeString().split(' ')[0]}]</span>
                        {log}
                    </p>
                ))}
                <div ref={logsEndRef} />
            </div>
        </div>

        {/* RIGHT COLUMN: Visualization (8 Columns) */}
        <div className="lg:col-span-8 bg-gray-900 rounded-lg border border-gray-800 flex items-center justify-center relative overflow-hidden max-h-[540px] shadow-2xl">
          
          {!previewUrl && (
            <div className="text-center">
                <div className="w-16 h-16 border-2 border-gray-800 border-dashed rounded-lg mx-auto mb-4 flex items-center justify-center">
                    <span className="text-2xl text-gray-700">+</span>
                </div>
                <p className="text-gray-600 font-mono text-sm">Upload Image to Initialize Buffer</p>
            </div>
          )}

          {previewUrl && (
            <div className="relative inline-block shadow-2xl">
              {/* The Image */}
              <img 
                ref={imgRef}
                src={previewUrl} 
                alt="Input" 
                className="max-h-[600px] w-auto border border-gray-700 rounded shadow-lg"
              />

              {/* Bounding Box Overlay */}
              {data && data.detections.map((det, idx) => (
                  <div
                    key={idx}
                    className="absolute border-2 border-green-500 bg-green-500/10 hover:bg-green-500/20 transition-colors group z-10"
                    style={{
                      left: `${det.box[0]}px`,
                      top: `${det.box[1]}px`,
                      width: `${det.box[2]}px`,
                      height: `${det.box[3]}px`,
                    }}
                  >
                    <span className="absolute -top-5 left-0 bg-green-600 text-black text-[10px] px-1.5 py-0.5 font-bold uppercase tracking-wider shadow-sm">
                      {det.label} {Math.round(det.conf * 100)}%
                    </span>
                  </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}