import { useEffect, useRef, useState } from 'react';
import * as GSPLAT from 'gsplat';

interface SplatViewerProps {
  plyUrl: string | null;
}

export const SplatViewer: React.FC<SplatViewerProps> = ({ plyUrl }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [downloadPct, setDownloadPct] = useState<number>(0);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [stats, setStats] = useState({ fps: 0, splats: 0 });

  useEffect(() => {
    if (!canvasRef.current || !plyUrl) return;

    let active = true;
    let renderer: any = null;
    let scene: any = null;
    let camera: any = null;
    let controls: any = null;
    let animationFrameId: number;

    const initViewer = async () => {
      setIsLoading(true);
      setErrorMsg(null);
      setDownloadPct(0);

      try {
        const canvas = canvasRef.current!;
        const container = containerRef.current!;
        
        // Initialize Scene
        scene = new GSPLAT.Scene();
        
        // Initialize Camera
        camera = new GSPLAT.Camera();
        
        // Initialize Renderer
        renderer = new GSPLAT.WebGLRenderer(canvas);
        
        // Initialize Controls
        controls = new GSPLAT.OrbitControls(camera, canvas);

        // Adjust camera aspect ratio
        const resize = () => {
          if (!container || !renderer) return;
          const width = container.clientWidth;
          const height = container.clientHeight;
          renderer.setSize(width, height);
          // camera aspect ratio is updated internally by gsplat WebGLRenderer resize
        };
        
        resize();
        window.addEventListener('resize', resize);

        // Load the 3D Gaussian Splat PLY model progressively
        console.log(`[Viewer] Loading PLY from: ${plyUrl}`);
        await GSPLAT.PLYLoader.LoadAsync(
          plyUrl,
          scene,
          (progress) => {
            if (active) {
              setDownloadPct(Math.round(progress * 100));
            }
          }
        );

        if (!active) return;
        setIsLoading(false);

        // Calculate total splat vertices in the scene
        let splatCount = 0;
        if (scene.objects && scene.objects.length > 0) {
          for (const obj of scene.objects) {
            if (obj.splat && obj.splat.count) {
              splatCount += obj.splat.count;
            }
          }
        }
        setStats(prev => ({ ...prev, splats: splatCount }));

        // Frame rate benchmarking variables
        let frameCount = 0;
        let fpsTimer = performance.now();

        // Render loop
        const tick = () => {
          if (!active) return;

          // Render
          if (controls) controls.update();
          if (renderer && scene && camera) renderer.render(scene, camera);

          // Calculate FPS
          frameCount++;
          const now = performance.now();
          if (now - fpsTimer >= 1000) {
            setStats(prev => ({
              ...prev,
              fps: Math.round((frameCount * 1000) / (now - fpsTimer))
            }));
            frameCount = 0;
            fpsTimer = now;
          }

          animationFrameId = requestAnimationFrame(tick);
        };

        animationFrameId = requestAnimationFrame(tick);

      } catch (err: any) {
        console.error('[Viewer Error]', err);
        setIsLoading(false);
        setErrorMsg(`Failed to initialize WebGL render context: ${err.message || err}`);
      }
    };

    initViewer();

    return () => {
      active = false;
      cancelAnimationFrame(animationFrameId);
      if (renderer) {
        try {
          renderer.dispose();
        } catch (e) {}
      }
      window.removeEventListener('resize', () => {});
    };
  }, [plyUrl]);

  return (
    <div ref={containerRef} className="relative w-full h-full rounded-2xl overflow-hidden border border-white/10 bg-slate-950/80 backdrop-blur-sm">
      {plyUrl ? (
        <canvas ref={canvasRef} className="w-full h-full block" />
      ) : (
        <div className="absolute inset-0 flex flex-col justify-center items-center text-slate-400">
          <svg className="w-12 h-12 text-indigo-500/50 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-sm font-semibold">No 3D Model Loaded</span>
          <span className="text-xs opacity-60 mt-1">Upload a dataset or start training to view 3D reconstruction</span>
        </div>
      )}

      {/* Loading Overlay */}
      {isLoading && (
        <div className="absolute inset-0 flex flex-col justify-center items-center bg-slate-950/90 z-20">
          <div className="relative w-20 h-20 mb-4 flex items-center justify-center">
            <div className="absolute inset-0 rounded-full border-2 border-indigo-500/10" />
            <div className="absolute inset-0 rounded-full border-2 border-t-indigo-400 animate-spin" />
            <span className="text-xs font-mono text-cyan-400">{downloadPct}%</span>
          </div>
          <span className="text-sm font-semibold text-indigo-300">Streaming 3D Gaussians...</span>
          <div className="w-64 h-1.5 bg-white/10 rounded-full overflow-hidden mt-3">
            <div className="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 transition-all duration-200" style={{ width: `${downloadPct}%` }} />
          </div>
          <span className="text-xs text-white/40 mt-1.5">progressive PLY parsing</span>
        </div>
      )}

      {/* Error Overlay */}
      {errorMsg && (
        <div className="absolute inset-0 flex flex-col justify-center items-center bg-slate-950/90 p-6 text-center z-20">
          <svg className="w-10 h-10 text-red-500 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="text-sm font-bold text-red-400">WebGL Load Failure</span>
          <span className="text-xs text-white/50 mt-1 max-w-xs">{errorMsg}</span>
        </div>
      )}

      {/* Render Telemetry Info Overlay */}
      {plyUrl && !isLoading && !errorMsg && (
        <div className="absolute bottom-4 left-4 flex gap-3 z-10 pointer-events-none">
          <div className="px-3 py-1.5 bg-slate-900/80 backdrop-blur-md rounded-lg border border-white/5 text-[11px] font-mono flex items-center gap-1.5 text-white/80">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span>WebGL Render: {stats.fps} FPS</span>
          </div>
          <div className="px-3 py-1.5 bg-slate-900/80 backdrop-blur-md rounded-lg border border-white/5 text-[11px] font-mono flex items-center gap-1.5 text-white/80">
            <span>Points: {(stats.splats / 1000).toFixed(0)}k Gaussians</span>
          </div>
        </div>
      )}
    </div>
  );
};
