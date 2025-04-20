import React, { useRef, useEffect } from 'react';

// TypeScript interface for a body in the simulation
interface Body {
  x: number;
  y: number;
  vx: number;
  vy: number;
  mass: number;
  color: string;
}

const WIDTH = window.innerWidth;
const HEIGHT = window.innerHeight;

// Figure-eight periodic orbit initial conditions (G=1, m=1)
// Source: Chenciner & Montgomery (2000)
const initialBodies: Body[] = [
  // Body 1
  {
    x: 0.97000436,
    y: -0.24308753,
    vx: 0.4662036850,
    vy: 0.4323657300,
    mass: 1,
    color: '#FFD700',
  },
  // Body 2
  {
    x: -0.97000436,
    y: 0.24308753,
    vx: 0.4662036850,
    vy: 0.4323657300,
    mass: 1,
    color: '#FF8C00',
  },
  // Body 3
  {
    x: 0,
    y: 0,
    vx: -0.93240737,
    vy: -0.86473146,
    mass: 1,
    color: '#FF4500',
  },
];

// Use G=1 for figure-eight, and a smaller DT for stability
const G = 1;
const DT = 0.002;

const ThreeBodySim: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bodiesRef = useRef<Body[]>(JSON.parse(JSON.stringify(initialBodies)));

  // Make canvas truly full screen and responsive
  useEffect(() => {
    function handleResize() {
      if (canvasRef.current) {
        canvasRef.current.width = window.innerWidth;
        canvasRef.current.height = window.innerHeight;
      }
    }
    window.addEventListener('resize', handleResize);
    handleResize();
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Animation loop
  useEffect(() => {
    let animationId: number;
    // Camera state for fixed center and scale
    // Compute initial center and scale from initialBodies
    const margin = 0.6; // 60% margin for visible motion
    const xs = initialBodies.map(b => b.x);
    const ys = initialBodies.map(b => b.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const fixedCenter = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
    const spanX = (maxX - minX) * (1 + margin) || 1;
    const spanY = (maxY - minY) * (1 + margin) || 1;
    const fixedScale = Math.min(window.innerWidth / spanX, window.innerHeight / spanY);
    function step() {
      const ctx = canvasRef.current?.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      // Use fixed center and scale
      for (const body of bodiesRef.current) {
        ctx.beginPath();
        const drawX = window.innerWidth / 2 + (body.x - fixedCenter.x) * fixedScale;
        const drawY = window.innerHeight / 2 + (body.y - fixedCenter.y) * fixedScale;
        const radius = 14;
        ctx.arc(drawX, drawY, radius, 0, 2 * Math.PI);
        ctx.fillStyle = body.color;
        ctx.shadowColor = body.color;
        ctx.shadowBlur = 16;
        ctx.fill();
        ctx.shadowBlur = 0;
      }
      // Update positions and velocities
      for (let i = 0; i < bodiesRef.current.length; i++) {
        let ax = 0, ay = 0;
        for (let j = 0; j < bodiesRef.current.length; j++) {
          if (i === j) continue;
          const dx = bodiesRef.current[j].x - bodiesRef.current[i].x;
          const dy = bodiesRef.current[j].y - bodiesRef.current[i].y;
          const distSq = dx * dx + dy * dy;
          const dist = Math.sqrt(distSq) + 1e-6;
          const force = (G * bodiesRef.current[j].mass) / distSq;
          ax += force * dx / dist;
          ay += force * dy / dist;
        }
        bodiesRef.current[i].vx += ax * DT;
        bodiesRef.current[i].vy += ay * DT;
      }
      for (const body of bodiesRef.current) {
        body.x += body.vx * DT;
        body.y += body.vy * DT;
      }
      animationId = requestAnimationFrame(step);
    }
    step();
    return () => cancelAnimationFrame(animationId);
  }, []);

  // Reset simulation on click
  function handleReset() {
    bodiesRef.current = JSON.parse(JSON.stringify(initialBodies));
  }

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      background: 'black',
      margin: 0,
      padding: 0,
      overflow: 'hidden',
      zIndex: 1000,
    }}>
      <canvas
        ref={canvasRef}
        width={window.innerWidth}
        height={window.innerHeight}
        style={{
          display: 'block',
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'black',
        }}
      />
      <div style={{
        position: 'absolute',
        top: 24,
        left: 0,
        width: '100vw',
        color: 'white',
        textAlign: 'center',
        pointerEvents: 'none',
        fontFamily: 'sans-serif',
        textShadow: '0 0 8px #000',
      }}>
        <h1 style={{margin: 0}}>Three-Body Simulation (TypeScript + Canvas)</h1>
        <p style={{margin: 0}}>This animation simulates three equal-mass bodies in a figure-eight periodic orbit.</p>
      </div>
      <button
        onClick={handleReset}
        style={{
          position: 'absolute',
          top: 24,
          right: 32,
          zIndex: 1100,
          padding: '10px 28px',
          fontSize: 18,
          background: '#222',
          color: 'white',
          border: '1px solid #555',
          borderRadius: 8,
          cursor: 'pointer',
          opacity: 0.85,
        }}
      >
        Reset Simulation
      </button>
    </div>
  );
};

export default ThreeBodySim;