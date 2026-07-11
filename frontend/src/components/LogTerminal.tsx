import React, { useEffect, useRef, useState } from 'react';

interface LogTerminalProps {
  logs: string[];
}

export const LogTerminal: React.FC<LogTerminalProps> = ({ logs }) => {
  const terminalEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [filterText, setFilterText] = useState<string>('');
  const [autoScroll, setAutoScroll] = useState<boolean>(true);

  // Auto-scroll logic when new logs arrive
  useEffect(() => {
    if (autoScroll && terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  // Handle user scroll detection
  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    
    // If user scrolled up by more than 30px, disable auto scroll
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 30;
    setAutoScroll(isAtBottom);
  };

  const filteredLogs = logs.filter(line => 
    line.toLowerCase().includes(filterText.toLowerCase())
  );

  return (
    <div className="glass-panel rounded-2xl h-full flex flex-col overflow-hidden border border-white/10">
      {/* Terminal Header */}
      <div className="flex justify-between items-center px-5 py-3 bg-slate-900/60 border-b border-white/5">
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
          <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/80" />
          <div className="w-2.5 h-2.5 rounded-full bg-green-500/80" />
          <span className="text-xs font-mono font-bold text-slate-400 ml-2 tracking-wider">ns-train@cloud:~$ log --tail</span>
        </div>
        
        {/* Terminal Options */}
        <div className="flex items-center gap-3">
          {/* Keyword Search Input */}
          <input
            type="text"
            placeholder="Search logs..."
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
            className="px-2.5 py-1 bg-slate-950/80 text-[10px] text-emerald-400 border border-white/10 rounded-md focus:outline-none focus:border-emerald-500/40 w-36 font-mono"
          />
          {/* Auto Scroll Lock indicator */}
          <button 
            onClick={() => setAutoScroll(prev => !prev)}
            className={`px-2 py-0.5 rounded text-[9px] font-mono border transition-all ${
              autoScroll 
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' 
                : 'bg-slate-800/40 border-white/10 text-slate-400'
            }`}
          >
            {autoScroll ? 'LOCK SCROLL' : 'FREE SCROLL'}
          </button>
        </div>
      </div>

      {/* Terminal Body */}
      <div 
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 p-5 overflow-y-auto bg-slate-950/90 font-mono text-[11px] leading-relaxed select-text"
      >
        {filteredLogs.length > 0 ? (
          filteredLogs.map((line, idx) => {
            // Highlighting warnings or errors
            let colorClass = 'text-slate-300';
            if (line.includes('[SYSTEM ERROR]') || line.toLowerCase().includes('error') || line.toLowerCase().includes('fail')) {
              colorClass = 'text-red-400 font-semibold';
            } else if (line.includes('[SYSTEM]') || line.includes('Finished')) {
              colorClass = 'text-indigo-400 font-semibold';
            } else if (line.includes('loss=')) {
              colorClass = 'text-cyan-400/90';
            } else if (line.includes('[LLAMA]')) {
              colorClass = 'text-pink-400';
            }
            
            return (
              <div key={idx} className={`py-0.5 border-l-2 border-white/5 pl-2 hover:bg-white/5 transition-all ${colorClass}`}>
                {line}
              </div>
            );
          })
        ) : (
          <div className="text-white/20 text-center py-10">
            {filterText ? 'No logs match search filter.' : 'Terminal console is empty.'}
          </div>
        )}
        {/* Anchor point to scroll into view */}
        <div ref={terminalEndRef} />
      </div>
    </div>
  );
};
