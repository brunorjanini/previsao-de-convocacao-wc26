import { useRef, useState, useCallback, useEffect } from "react";

interface WebcamCaptureProps {
  onCapture: (dataUrl: string) => void;
  photo: string | null;
}

export function WebcamCapture({ onCapture, photo }: WebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [denied, setDenied] = useState(false);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      setStreaming(true);
      setDenied(false);
    } catch {
      setDenied(true);
    }
  }, []);

  // Anexa o stream ao elemento <video> depois que ele monta no DOM
  useEffect(() => {
    if (streaming && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [streaming]);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStreaming(false);
  }, []);

  const takePhoto = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    onCapture(canvas.toDataURL("image/jpeg", 0.9));
    stopCamera();
  }, [onCapture, stopCamera]);

  useEffect(() => () => { stopCamera(); }, [stopCamera]);

  if (photo) {
    return (
      <div className="flex items-center gap-3">
        <img src={photo} alt="Foto" className="w-16 h-16 rounded-full object-cover border-2 border-amber-400 shadow-md flex-shrink-0" />
        <button onClick={() => { onCapture(""); startCamera(); }} className="text-xs text-amber-300 underline hover:text-amber-100">
          Trocar foto
        </button>
      </div>
    );
  }

  if (denied) {
    return (
      <div className="flex items-center gap-3">
        <Silhouette />
        <p className="text-xs text-slate-400">Webcam indisponível</p>
      </div>
    );
  }

  if (streaming) {
    return (
      <div className="flex items-center gap-3">
        <video ref={videoRef} autoPlay playsInline className="w-24 h-20 rounded-lg object-cover border-2 border-amber-400" />
        <canvas ref={canvasRef} className="hidden" />
        <button onClick={takePhoto} className="px-3 py-1.5 bg-amber-500 hover:bg-amber-400 text-blue-950 font-bold rounded-lg text-sm">
          Capturar
        </button>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <Silhouette />
      <button onClick={startCamera} className="px-3 py-1.5 bg-amber-500 hover:bg-amber-400 text-blue-950 font-bold rounded-lg text-sm">
        Tirar Foto
      </button>
    </div>
  );
}

function Silhouette() {
  return (
    <div className="w-16 h-16 rounded-full bg-blue-800 border-2 border-amber-400 flex items-center justify-center overflow-hidden flex-shrink-0">
      <svg viewBox="0 0 100 100" className="w-12 h-12 fill-blue-600">
        <circle cx="50" cy="32" r="18" />
        <ellipse cx="50" cy="80" rx="28" ry="20" />
      </svg>
    </div>
  );
}
