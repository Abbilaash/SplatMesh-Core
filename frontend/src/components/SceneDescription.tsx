import React, { useState } from 'react';

interface SceneDescriptionProps {
  backendUrl: string;
}

export const SceneDescription: React.FC<SceneDescriptionProps> = ({ backendUrl }) => {
  const [description, setDescription] = useState<string>('');
  const [source, setSource] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDescription = async () => {
    setLoading(true);
    setError(null);
    try {
      console.log(`[Llama] Hitting describe endpoint...`);
      const response = await fetch(`${backendUrl}/api/describe`);
      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }
      const data = await response.json();
      if (data.status === 'success') {
        setDescription(data.text);
        setSource(data.source || 'Llama 3.1 NPU');
      } else {
        throw new Error(data.message || 'Unknown backend error');
      }
    } catch (err: any) {
      console.error('[Llama Fetch Error]', err);
      setError(err.message || 'Could not contact Llama service');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel p-6 rounded-2xl h-full flex flex-col justify-between border border-white/10">
      {/* Header */}
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-sm font-semibold tracking-wider text-slate-400 uppercase">Scene Annotation</h2>
        {source && (
          <span className="text-[10px] font-mono px-2 py-0.5 bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 rounded">
            {source.toUpperCase()}
          </span>
        )}
      </div>

      {/* Description Content */}
      <div className="flex-1 flex flex-col justify-center py-2">
        {loading ? (
          <div className="flex flex-col items-center justify-center space-y-2 py-4">
            <div className="flex space-x-1.5">
              <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
            <span className="text-[10px] font-mono text-slate-500">Llama 3.1 is analyzing keyframe...</span>
          </div>
        ) : error ? (
          <div className="text-xs text-red-400/90 font-mono text-center py-4 bg-red-950/20 border border-red-500/10 rounded-xl">
            {error}
          </div>
        ) : description ? (
          <p className="text-sm leading-relaxed text-slate-300 font-sans italic">
            "{description}"
          </p>
        ) : (
          <div className="text-xs text-slate-500 font-sans text-center py-6">
            Click analyze to request a natural language report of the scanned environment.
          </div>
        )}
      </div>

      {/* Trigger Button */}
      <div className="mt-3">
        <button
          onClick={fetchDescription}
          disabled={loading}
          className={`w-full py-2 bg-slate-900/80 hover:bg-indigo-600 border border-white/5 hover:border-indigo-500 text-xs font-semibold text-slate-300 hover:text-white rounded-xl transition-all duration-200 shadow-md ${
            loading ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          {loading ? 'ANALYZING...' : description ? 'RE-RUN SCENE NARRATOR' : 'DESCRIBE SCENE'}
        </button>
      </div>
    </div>
  );
};
