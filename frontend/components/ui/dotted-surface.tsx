'use client';

import { cn } from '@/lib/utils';
import { useTheme } from 'next-themes';
import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

type DottedSurfaceProps = Omit<React.ComponentProps<'div'>, 'ref'> & {
  analyser?: AnalyserNode | null;
  /**
   * How much speech amplifies wave height above the idle baseline.
   * 0 = no reaction, 1 = subtle, 2 = moderate (default), 5 = dramatic.
   */
  amplitudeScale?: number;
  /**
   * How much speech speeds up the animation.
   * 0 = no change, 0.06 = slight (default), 0.3 = fast.
   */
  speedScale?: number;
};

export function DottedSurface({
  className,
  analyser,
  amplitudeScale = 2,
  speedScale = 0.06,
  ...props
}: DottedSurfaceProps) {
  const { theme } = useTheme();

  const containerRef = useRef<HTMLDivElement>(null);
  // Use refs for everything that changes without needing a remount
  const analyserRef = useRef<AnalyserNode | null>(null);
  const amplitudeScaleRef = useRef(amplitudeScale);
  const speedScaleRef = useRef(speedScale);

  const sceneRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    particles: THREE.Points[];
    animationId: number;
    count: number;
  } | null>(null);

  useEffect(() => { analyserRef.current = analyser ?? null; }, [analyser]);
  useEffect(() => { amplitudeScaleRef.current = amplitudeScale; }, [amplitudeScale]);
  useEffect(() => { speedScaleRef.current = speedScale; }, [speedScale]);

  useEffect(() => {
    if (!containerRef.current) return;

    const SEPARATION = 150;
    const AMOUNTX = 40;
    const AMOUNTY = 60;

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x000000, 2000, 10000);

    const camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      10000,
    );
    camera.position.set(0, 355, 1220);

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    containerRef.current.appendChild(renderer.domElement);

    const positions: number[] = [];
    const colors: number[] = [];
    const geometry = new THREE.BufferGeometry();

    for (let ix = 0; ix < AMOUNTX; ix++) {
      for (let iy = 0; iy < AMOUNTY; iy++) {
        positions.push(
          ix * SEPARATION - (AMOUNTX * SEPARATION) / 2,
          0,
          iy * SEPARATION - (AMOUNTY * SEPARATION) / 2,
        );
        if (theme === 'dark') {
          colors.push(200, 200, 200);
        } else {
          colors.push(0, 0, 0);
        }
      }
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 8,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      sizeAttenuation: true,
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    let count = 0;
    let animationId: number;
    const freqData = new Uint8Array(128);

    const animate = () => {
      animationId = requestAnimationFrame(animate);

      // Idle: barely perceptible drift (amplitude ≈ 6 units, very slow)
      // Speech: scales up smoothly based on energy
      let energy = 0;
      if (analyserRef.current) {
        analyserRef.current.getByteFrequencyData(freqData);
        energy = freqData.slice(0, 40).reduce((a: number, b: number) => a + b, 0) / 40 / 255;
      }

      const waveHeight = 18 + energy * amplitudeScaleRef.current * 50;  // 18 idle → up to ~118 on loud speech
      const speed = 0.04 + energy * speedScaleRef.current;             // 0.04 idle → slight increase on speech

      const posAttr = geometry.attributes.position;
      const pos = posAttr.array as Float32Array;

      let i = 0;
      for (let ix = 0; ix < AMOUNTX; ix++) {
        for (let iy = 0; iy < AMOUNTY; iy++) {
          pos[i * 3 + 1] =
            Math.sin((ix + count) * 0.3) * waveHeight +
            Math.sin((iy + count) * 0.5) * waveHeight;
          i++;
        }
      }

      posAttr.needsUpdate = true;
      renderer.render(scene, camera);
      count += speed;
    };

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);
    animate();

    sceneRef.current = {
      scene,
      camera,
      renderer,
      particles: [points],
      animationId,
      count,
    };

    return () => {
      window.removeEventListener('resize', handleResize);
      if (sceneRef.current) {
        cancelAnimationFrame(sceneRef.current.animationId);
        sceneRef.current.scene.traverse((object) => {
          if (object instanceof THREE.Points) {
            object.geometry.dispose();
            if (Array.isArray(object.material)) {
              object.material.forEach((m) => m.dispose());
            } else {
              object.material.dispose();
            }
          }
        });
        sceneRef.current.renderer.dispose();
        if (containerRef.current && sceneRef.current.renderer.domElement) {
          containerRef.current.removeChild(sceneRef.current.renderer.domElement);
        }
      }
    };
  }, [theme]);

  return (
    <div
      ref={containerRef}
      className={cn('pointer-events-none fixed inset-0 -z-10', className)}
      {...props}
    />
  );
}
