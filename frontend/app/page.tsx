"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Mic, Square, Loader2 } from "lucide-react";
import { DottedSurface } from "@/components/ui/dotted-surface";
import { Badge } from "@/components/ui/badge";

function OrbitLoader() {
  return (
    <div className="relative h-8 w-8">
      {/* Static ring */}
      <span className="absolute inset-0 rounded-full border border-border/40" />
      {/* Orbiting dot */}
      <span
        className="absolute inset-0 rounded-full border-t-2 border-foreground/70 animate-spin"
        style={{ animationDuration: "900ms", animationTimingFunction: "linear" }}
      />
      {/* Trailing dot at 180° offset, slower */}
      <span
        className="absolute inset-[5px] rounded-full border-t border-foreground/25 animate-spin"
        style={{ animationDuration: "1400ms", animationTimingFunction: "linear" }}
      />
    </div>
  );
}
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type RecordingState = "idle" | "connecting" | "listening" | "processing";

const WS_URL = "ws://localhost:8000/ws/stream/";
const SAMPLE_RATE = 16000;
const BUFFER_SIZE = 4096;

export default function Home() {
  const [recordingState, setRecordingState] = useState<RecordingState>("idle");
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState("");
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const stopAudio = useCallback(() => {
    processorRef.current?.disconnect();
    processorRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    setAnalyser(null);
  }, []);

  const closeSession = useCallback(() => {
    stopAudio();
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send("stop");
    } else {
      wsRef.current?.close();
      wsRef.current = null;
      setRecordingState("idle");
    }
  }, [stopAudio]);

  const startRecording = useCallback(async () => {
    setError("");
    setTranscript("");
    setRecordingState("connecting");

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true },
      });
    } catch {
      setError("Microphone access denied.");
      setRecordingState("idle");
      return;
    }

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onmessage = (evt) => {
      const raw: string =
        evt.data instanceof ArrayBuffer
          ? new TextDecoder().decode(evt.data)
          : (evt.data as string);

      if (raw === "listening") {
        setRecordingState("listening");

        const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
        audioCtxRef.current = audioCtx;
        streamRef.current = stream;

        const source = audioCtx.createMediaStreamSource(stream);
        const an = audioCtx.createAnalyser();
        an.fftSize = 256;
        setAnalyser(an);

        const processor = audioCtx.createScriptProcessor(BUFFER_SIZE, 1, 1);
        processorRef.current = processor;
        source.connect(an);
        source.connect(processor);
        processor.connect(audioCtx.destination);

        processor.onaudioprocess = (e) => {
          if (ws.readyState !== WebSocket.OPEN) return;
          const float32 = e.inputBuffer.getChannelData(0);
          const int16 = new Int16Array(float32.length);
          for (let i = 0; i < float32.length; i++) {
            int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
          }
          ws.send(int16.buffer);
        };
        return;
      }

      if (raw === "silence") {
        stopAudio();
        setRecordingState("processing");
        return;
      }

      try {
        const data = JSON.parse(raw);
        if (data.transcription !== undefined) setTranscript(data.transcription);
        else if (data.error) setError(data.error);
      } catch { /* ignore */ }
      setRecordingState("idle");
      wsRef.current = null;
    };

    ws.onerror = () => {
      setError("Connection failed. Is the backend running?");
      stopAudio();
      setRecordingState("idle");
    };

    ws.onclose = () => {
      stopAudio();
      setRecordingState((s) => (s === "processing" ? s : "idle"));
    };
  }, [stopAudio]);

  useEffect(() => {
    return () => { stopAudio(); wsRef.current?.close(); };
  }, [stopAudio]);

  const isListening = recordingState === "listening";
  const isProcessing = recordingState === "processing";
  const isConnecting = recordingState === "connecting";
  const isIdle = recordingState === "idle";

  return (
    // No background color — lets DottedSurface show through
    <main className="relative min-h-screen w-full overflow-hidden">
      <DottedSurface analyser={analyser} amplitudeScale={2} speedScale={0.08} />

      <div className="relative z-10 flex min-h-screen flex-col items-center justify-center gap-10 px-6 py-16">

        {/* Status badge */}
        <Badge
          variant="outline"
          className={cn(
            "gap-1.5 border-border/50 bg-background/40 backdrop-blur-sm text-muted-foreground font-normal tracking-wide",
            isListening && "border-cyan-500/40 text-cyan-400",
            isProcessing && "border-violet-500/40 text-violet-400",
          )}
        >
          <span className={cn(
            "h-1.5 w-1.5 rounded-full",
            isIdle && "bg-muted-foreground",
            isConnecting && "bg-yellow-400 animate-pulse",
            isListening && "bg-cyan-400 animate-pulse",
            isProcessing && "bg-violet-400 animate-pulse",
          )} />
          {isIdle && "Conformer ASR · Ready"}
          {isConnecting && "Connecting…"}
          {isListening && "Listening"}
          {isProcessing && "Transcribing"}
        </Badge>

        {/* Hero */}
        <div className="text-center space-y-3 max-w-lg">
          <h1 className="text-5xl font-semibold tracking-tight text-foreground">
            Speech to Text
          </h1>
          <p className="text-muted-foreground text-base">
            On-device automatic speech recognition powered by a Conformer CTC model.
            No cloud, no data leaves your machine.
          </p>
        </div>

        {/* Mic button */}
        <div className="relative flex items-center justify-center">
          {isListening && (
            <>
              <span className="absolute h-32 w-32 rounded-full border border-foreground/10 animate-ping" style={{ animationDuration: "2s" }} />
              <span className="absolute h-24 w-24 rounded-full border border-foreground/15 animate-ping" style={{ animationDuration: "1.4s" }} />
            </>
          )}

          {(isIdle || isConnecting) && (
            <Button
              size="icon"
              variant="outline"
              onClick={isIdle ? startRecording : undefined}
              disabled={isConnecting}
              className="h-16 w-16 rounded-full border-border/60 bg-background/50 backdrop-blur-sm hover:bg-background/80 shadow-sm"
              aria-label="Start recording"
            >
              {isConnecting
                ? <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                : <Mic className="h-6 w-6" />
              }
            </Button>
          )}

          {isListening && (
            <Button
              size="icon"
              variant="destructive"
              onClick={closeSession}
              className="h-16 w-16 rounded-full shadow-sm"
              aria-label="Stop recording"
            >
              <Square className="h-5 w-5 fill-current" />
            </Button>
          )}

          {isProcessing && (
            <div className="h-16 w-16 rounded-full border border-border/40 bg-background/40 backdrop-blur-sm flex items-center justify-center">
              <OrbitLoader />
            </div>
          )}
        </div>

        {/* Helper text */}
        <p className="text-xs text-muted-foreground/60 tracking-wide">
          {isIdle && !transcript && "Click the microphone to start"}
          {isListening && "Speak clearly · stops automatically on silence"}
          {isProcessing && "Running beam-search decoder…"}
          {isIdle && transcript && (
            <button
              onClick={() => { setTranscript(""); setError(""); }}
              className="hover:text-muted-foreground transition-colors cursor-pointer"
            >
              Record again
            </button>
          )}
        </p>

        {/* Result card */}
        {(transcript || error) && (
          <Card className={cn(
            "w-full max-w-lg bg-background/60 backdrop-blur-md border-border/50 animate-slide-up",
            error && "border-destructive/30 bg-destructive/5"
          )}>
            <CardContent className="pt-6">
              <p className={cn(
                "text-base leading-relaxed",
                error ? "text-destructive" : "text-foreground"
              )}>
                {error || transcript}
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </main>
  );
}
