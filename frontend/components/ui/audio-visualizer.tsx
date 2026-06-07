"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

interface AudioVisualizerProps {
  analyser: AnalyserNode | null;
  className?: string;
  barCount?: number;
}

/**
 * Multiband bar visualizer inspired by the LiveKit default agent UI.
 * Reads frequency data from a Web Audio AnalyserNode each animation frame.
 */
export function AudioVisualizer({
  analyser,
  className,
  barCount = 20,
}: AudioVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const bufferLength = analyser ? analyser.frequencyBinCount : 128;
    const dataArray = new Uint8Array(bufferLength);

    // Idle animation values when no analyser is provided
    const idlePhases = Array.from(
      { length: barCount },
      (_, i) => (i / barCount) * Math.PI * 2
    );
    let idleT = 0;

    const draw = () => {
      frameRef.current = requestAnimationFrame(draw);
      ctx.clearRect(0, 0, W, H);

      const barWidth = (W / barCount) * 0.6;
      const gap = (W / barCount) * 0.4;

      for (let i = 0; i < barCount; i++) {
        let normalised: number;

        if (analyser) {
          analyser.getByteFrequencyData(dataArray);
          // Map bar index to a frequency bin (focus on speech range 0-4kHz)
          const binIndex = Math.floor(
            (i / barCount) * (bufferLength * 0.35)
          );
          normalised = dataArray[binIndex] / 255;
        } else {
          // Gentle idle wave
          normalised =
            0.08 + 0.06 * Math.sin(idleT * 1.5 + idlePhases[i]);
        }

        const barH = Math.max(4, normalised * H * 0.85);
        const x = i * (barWidth + gap) + gap / 2;
        const y = (H - barH) / 2;

        // Gradient: cyan → blue/purple (LiveKit palette)
        const grad = ctx.createLinearGradient(x, y, x, y + barH);
        grad.addColorStop(0, `rgba(120, 220, 255, ${0.4 + normalised * 0.6})`);
        grad.addColorStop(
          0.5,
          `rgba(80, 160, 255, ${0.5 + normalised * 0.5})`
        );
        grad.addColorStop(
          1,
          `rgba(140, 80, 255, ${0.4 + normalised * 0.6})`
        );

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barH, barWidth / 2);
        ctx.fill();
      }

      idleT += 0.04;
    };

    draw();
    return () => cancelAnimationFrame(frameRef.current);
  }, [analyser, barCount]);

  return (
    <canvas
      ref={canvasRef}
      width={320}
      height={120}
      className={cn("w-80 h-30", className)}
    />
  );
}
