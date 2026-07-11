import { useEffect, useState } from 'react';
import type { ServerStatus } from './types';
import { SplatViewer } from './components/SplatViewer';
import { TelemetryHUD } from './components/TelemetryHUD';
import { LogTerminal } from './components/LogTerminal';
import { SceneDescription } from './components/SceneDescription';

export default function App() {
  const [backendUrl, setBackendUrl] = useState<string>('http://127.0.0.1:5000');
  const [ipInput, setIpInput] = useState<string>('127.0.0.1');
  const [connected, setConnected] = useState<boolean>(false);
  const [status, setStatus] = useState<ServerStatus | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [plyUrl, setPlyUrl] = useState<string | null>(null);
  const [autoLoadCheckpoints, setAutoLoadCheckpoints] = useState<boolean>(true);

  // Sync IP input changes into the full HTTP backend URL
  const handleConnect = () => {
    const ip = ipInput.trim();
    if (ip) {
      setBackendUrl(`http://${ip}:5000`);
      setLogs(prev => [...prev, `[SYSTEM] Switched backend gateway to: http://${ip}:5000`]);
    }
  };

  // 1. Connection diagnostic check on backendUrl change
  useEffect(() => {
    let active = true;
    const checkConnection = async () => {
      try {
        const response = await fetch(backendUrl);
        if (response.ok) {
          const data = await response.json();
          if (active) {
            setConnected(true);
            setLogs(prev => [...prev, `[SYSTEM] Cloud server online. Mode: ${data.mock_mode ? 'MOCK' : 'REAL'}`]);
          }
        } else {
          if (active) setConnected(false);
        }
      } catch (err) {
        if (active) setConnected(false);
      }
    };

    checkConnection();
    return () => {
      active = false;
    };
  }, [backendUrl]);

  // 2. Continuous telemetry status & log polling loop (runs every 2 seconds)
  useEffect(() => {
    if (!connected) return;

    let active = true;
    const pollInterval = setInterval(async () => {
      try {
        // Fetch Telemetry Status
        const statusRes = await fetch(`${backendUrl}/api/status`);
        if (statusRes.ok) {
          const statusData: ServerStatus = await statusRes.json();
          if (active) {
            setStatus(statusData);

            // Progressive 3D model streaming resolver
            if (statusData.ply_available) {
              // Final PLY model is ready
              const finalUrl = `${backendUrl}/api/splat.ply`;
              if (plyUrl !== finalUrl) {
                setPlyUrl(finalUrl);
                setLogs(prev => [...prev, `[SYSTEM] Training completed. Loading final 3D model...`]);
              }
            } else if (autoLoadCheckpoints && statusData.checkpoint_ply_available) {
              // Intermediate checkpoint model is available
              const checkpointUrl = `${backendUrl}/api/splat/checkpoint?t=${Date.now()}`; // break browser cache
              if (plyUrl !== checkpointUrl) {
                setPlyUrl(checkpointUrl);
                setLogs(prev => [...prev, `[SYSTEM] New checkpoint dataset loaded. Progressive update triggered.`]);
              }
            } else {
              if (plyUrl !== null) setPlyUrl(null);
            }
          }
        }

        // Fetch logs
        const logsRes = await fetch(`${backendUrl}/api/logs?lines=60`);
        if (logsRes.ok) {
          const logsData = await logsRes.json();
          if (active) {
            setLogs(logsData.lines || []);
          }
        }
      } catch (err) {
        console.error('[Polling Error]', err);
      }
    }, 2000);

    return () => {
      active = false;
      clearInterval(pollInterval);
    };
  }, [connected, backendUrl, plyUrl, autoLoadCheckpoints]);

  // Action: POST /api/train/start
  const handleStartTraining = async () => {
    try {
      setLogs(prev => [...prev, `[SYSTEM] Starting training loop command sent...`]);
      const res = await fetch(`${backendUrl}/api/train/start`, { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setLogs(prev => [...prev, `[SYSTEM] Training started (PID: ${data.pid})`]);
      } else {
        setLogs(prev => [...prev, `[SYSTEM ERROR] Start failed: ${data.message || 'Unknown'}`]);
      }
    } catch (e: any) {
      setLogs(prev => [...prev, `[SYSTEM ERROR] Request error: ${e.message}`]);
    }
  };

  // Action: POST /api/train/stop
  const handleStopTraining = async () => {
    try {
      setLogs(prev => [...prev, `[SYSTEM] Stopping training loop command sent...`]);
      const res = await fetch(`${backendUrl}/api/train/stop`, { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setLogs(prev => [...prev, `[SYSTEM] Training aborted successfully.`]);
      } else {
        setLogs(prev => [...prev, `[SYSTEM ERROR] Abort failed: ${data.message || 'Unknown'}`]);
      }
    } catch (e: any) {
      setLogs(prev => [...prev, `[SYSTEM ERROR] Request error: ${e.message}`]);
    }
  };

  return (
    <div className="min-h-screen bg-[#07070B] text-slate-100 flex flex-col">
      {/* Top Header Navigation */}
      <header className="px-6 py-4 bg-slate-950/80 backdrop-blur-md border-b border-white/5 flex flex-wrap justify-between items-center gap-4 z-10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-400 flex items-center justify-center font-bold font-mono text-white shadow-lg shadow-indigo-500/20">
            SM
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight text-white">SplatMesh Core</h1>
            <p className="text-[10px] text-slate-500 font-mono tracking-wider">CLOUD MONITOR & VISUALIZER</p>
          </div>
        </div>

        {/* Cloud Link Config */}
        <div className="flex items-center gap-3 bg-slate-900/60 p-1.5 rounded-xl border border-white/5">
          <div className="flex items-center gap-2 px-2.5">
            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
            <span className="text-xs font-mono font-semibold text-slate-400">
              {connected ? 'Cloud Server Connected' : 'Cloud Server Disconnected'}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="text"
              value={ipInput}
              onChange={(e) => setIpInput(e.target.value)}
              placeholder="Cloud IP"
              className="px-3 py-1 bg-slate-950/80 text-xs text-white border border-white/10 rounded-lg focus:outline-none focus:border-indigo-500/40 w-32 font-mono"
            />
            <button
              onClick={handleConnect}
              className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-xs font-semibold text-white rounded-lg transition-all"
            >
              Link
            </button>
          </div>
        </div>
      </header>

      {/* Main Grid Content Dashboard */}
      <main className="flex-1 p-6 grid grid-cols-1 lg:grid-cols-4 gap-6 overflow-hidden">
        {/* Left Control & Telemetry Panel (1 Column) */}
        <div className="lg:col-span-1 flex flex-col gap-6 h-full justify-between">
          {/* Quick Actions Panel */}
          <div className="glass-panel p-5 rounded-2xl border border-white/10 flex flex-col gap-4">
            <h2 className="text-xs font-bold tracking-wider text-slate-400 uppercase">Training Controls</h2>
            
            <div className="flex flex-col gap-3">
              <button
                onClick={handleStartTraining}
                disabled={!connected || (status?.training_active ?? false)}
                className="py-2.5 bg-gradient-to-r from-indigo-500 to-indigo-600 hover:from-indigo-400 hover:to-indigo-500 disabled:from-slate-800 disabled:to-slate-800 disabled:opacity-50 text-xs font-bold text-white rounded-xl transition-all shadow-lg shadow-indigo-500/10 cursor-pointer disabled:cursor-not-allowed"
              >
                RUN NEURAL TRAINING
              </button>
              <button
                onClick={handleStopTraining}
                disabled={!connected || !(status?.training_active ?? false)}
                className="py-2.5 bg-slate-900/60 hover:bg-red-950/30 disabled:hover:bg-slate-900/60 border border-white/5 hover:border-red-500/30 text-xs font-bold text-slate-400 hover:text-red-400 rounded-xl transition-all cursor-pointer disabled:cursor-not-allowed"
              >
                ABORT RUN
              </button>
            </div>

            <div className="border-t border-white/5 pt-3 flex items-center justify-between">
              <span className="text-[10px] text-slate-500 font-mono">PROGRESSIVE CHECKPOINTS</span>
              <button
                onClick={() => setAutoLoadCheckpoints(prev => !prev)}
                className={`w-10 h-5 rounded-full p-0.5 transition-all duration-200 ${
                  autoLoadCheckpoints ? 'bg-indigo-500' : 'bg-slate-800'
                }`}
              >
                <div
                  className={`w-4 h-4 rounded-full bg-white transition-all duration-200 ${
                    autoLoadCheckpoints ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>
          </div>

          {/* Telemetry Curves HUD */}
          <div className="flex-1 mt-1">
            <TelemetryHUD status={status} />
          </div>
        </div>

        {/* Center / Right 3D Viewport & Metadata (3 Columns) */}
        <div className="lg:col-span-3 flex flex-col gap-6 h-full">
          {/* Top Row: Viewport & Description */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 flex-1 min-h-[350px]">
            {/* 3D Canvas Box (2 Columns) */}
            <div className="md:col-span-2 h-full">
              <SplatViewer plyUrl={plyUrl} />
            </div>
            
            {/* Llama Description card (1 Column) */}
            <div className="md:col-span-1 h-full">
              <SceneDescription backendUrl={backendUrl} />
            </div>
          </div>

          {/* Bottom Row: Scrolling logs (Fixed height) */}
          <div className="h-60">
            <LogTerminal logs={logs} />
          </div>
        </div>
      </main>
    </div>
  );
}
