import { useEffect, useState } from 'react';
import type { ServerStatus } from '../types';

interface TelemetryHUDProps {
  status: ServerStatus | null;
}

export const TelemetryHUD: React.FC<TelemetryHUDProps> = ({ status }) => {
  const [lossHistory, setLossHistory] = useState<number[]>([]);

  // Watch for changes in loss to update the custom history curve
  useEffect(() => {
    if (status && status.training_active && status.current_loss > 0) {
      setLossHistory(prev => {
        // Prevent adjacent duplicates
        if (prev.length > 0 && prev[prev.length - 1] === status.current_loss) {
          return prev;
        }
        const updated = [...prev, status.current_loss];
        if (updated.length > 30) updated.shift(); // Keep last 30 readings
        return updated;
      });
    } else if (status && !status.training_active && !status.training_done) {
      setLossHistory([]); // Reset when idle
    }
  }, [status?.current_loss, status?.training_active, status?.training_done]);

  if (!status) {
    return (
      <div className="glass-panel p-6 rounded-2xl h-full flex items-center justify-center text-slate-400">
        <span className="text-xs font-mono">No telemetry connection...</span>
      </div>
    );
  }

  const {
    training_active,
    training_done,
    progress,
    iteration,
    total,
    current_loss,
    eta_seconds,
  } = status;

  // Helper to format remaining time
  const formatTime = (seconds: number) => {
    if (seconds <= 0) return '00:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  // SVG dimensions for the loss chart
  const chartWidth = 300;
  const chartHeight = 80;

  // Calculate coordinates for the SVG path
  const getSvgPath = () => {
    if (lossHistory.length < 2) return '';
    const minVal = Math.min(...lossHistory) * 0.9;
    const maxVal = Math.max(...lossHistory) * 1.1;
    const valRange = maxVal - minVal || 1.0;

    return lossHistory
      .map((val, index) => {
        const x = (index / (lossHistory.length - 1)) * chartWidth;
        const y = chartHeight - ((val - minVal) / valRange) * chartHeight;
        return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
      })
      .join(' ');
  };

  return (
    <div className="glass-panel p-6 rounded-2xl h-full flex flex-col justify-between border border-white/10 animate-pulse-glow">
      {/* HUD Header */}
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-sm font-semibold tracking-wider text-slate-400 uppercase">System Telemetry</h2>
        <div className="flex items-center gap-2">
          <span className={`w-2.5 h-2.5 rounded-full ${
            training_active ? 'bg-indigo-500 animate-ping' : training_done ? 'bg-green-500' : 'bg-slate-600'
          }`} />
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-300">
            {training_active ? 'Training Active' : training_done ? 'Completed' : 'System Idle'}
          </span>
        </div>
      </div>

      {/* Primary Metrics Grid */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        {/* Step progress counter */}
        <div className="p-3 bg-slate-900/40 rounded-xl border border-white/5">
          <span className="text-[10px] text-slate-500 font-mono block">ITERATION STEP</span>
          <span className="text-lg font-mono font-bold text-white neon-text-blue">
            {iteration.toLocaleString()} <span className="text-xs opacity-50">/ {total.toLocaleString()}</span>
          </span>
        </div>
        {/* Current training loss */}
        <div className="p-3 bg-slate-900/40 rounded-xl border border-white/5">
          <span className="text-[10px] text-slate-500 font-mono block">CURRENT LOSS</span>
          <span className="text-lg font-mono font-bold text-cyan-400">
            {current_loss > 0 ? current_loss.toFixed(4) : '--'}
          </span>
        </div>
        {/* Training progress percent */}
        <div className="p-3 bg-slate-900/40 rounded-xl border border-white/5">
          <span className="text-[10px] text-slate-500 font-mono block">PROGRESS</span>
          <span className="text-lg font-mono font-bold text-indigo-400">
            {progress.toFixed(1)}%
          </span>
        </div>
        {/* ETA Remaining */}
        <div className="p-3 bg-slate-900/40 rounded-xl border border-white/5">
          <span className="text-[10px] text-slate-500 font-mono block">TIME REMAINING</span>
          <span className="text-lg font-mono font-bold text-amber-400">
            {training_active ? formatTime(eta_seconds) : '--'}
          </span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-4">
        <div className="w-full h-2 bg-slate-950/80 rounded-full overflow-hidden border border-white/5">
          <div
            className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-cyan-400 transition-all duration-500 rounded-full"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Custom Real-Time SVG Loss Chart */}
      <div className="flex-1 flex flex-col justify-end">
        <span className="text-[10px] text-slate-500 font-mono mb-1.5 uppercase">Loss Convergence Curve</span>
        <div className="w-full h-20 bg-slate-950/60 rounded-xl overflow-hidden border border-white/5 flex items-center justify-center relative p-1">
          {lossHistory.length >= 2 ? (
            <>
              <svg className="w-full h-full" viewBox={`0 0 ${chartWidth} ${chartHeight}`} preserveAspectRatio="none">
                {/* Gradient Fill under Path */}
                <defs>
                  <linearGradient id="chartGlow" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#818cf8" stopOpacity="0.4" />
                    <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.0" />
                  </linearGradient>
                </defs>
                {/* Curve Stroke */}
                <path
                  d={getSvgPath()}
                  fill="none"
                  stroke="url(#lineGlow)"
                  strokeWidth="2"
                  className="stroke-indigo-400"
                />
                <linearGradient id="lineGlow" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#6366f1" />
                  <stop offset="100%" stopColor="#22d3ee" />
                </linearGradient>
              </svg>
              {/* Overlay values */}
              <span className="absolute top-1 right-2 text-[9px] font-mono text-cyan-400/80">
                Min: {Math.min(...lossHistory).toFixed(4)}
              </span>
            </>
          ) : (
            <span className="text-[10px] font-mono text-white/20">Waiting for loss sequence data...</span>
          )}
        </div>
      </div>
    </div>
  );
};
