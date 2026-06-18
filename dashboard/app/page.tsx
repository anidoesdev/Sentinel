"use client";

import AlertFeed from "../components/AlertFeed";
import AnomalyChart from "../components/AnomalyChart";
import SensorGrid from "../components/SensorGrid";
import SystemStatus from "../components/SystemStatus";
import { useReplaySocket } from "../hooks/useReplaySocket";

export default function Dashboard() {
  const { latest, history, alerts, connected, error } = useReplaySocket();

  const threshold = latest?.threshold ?? 0.05;

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <header className="mb-6 flex items-stretch gap-4">
        <div className="w-1 bg-blue-500" />
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-[0.2em] font-mono text-white">
              SENTINEL
            </h1>
            <span className="font-mono text-xs text-blue-400 border border-blue-900 px-2 py-0.5">
              v0.1
            </span>
          </div>
          <p className="hmi-label mt-1">Real-Time Multimodal Anomaly Detection System</p>
        </div>
        <div className="text-right self-center">
          <div className="hmi-label">Unit ID</div>
          <div className="font-mono text-sm text-neutral-300 mt-0.5">SEN-001</div>
        </div>
      </header>

      {/* System status bar */}
      <div className="mb-6">
        <SystemStatus connected={connected} latest={latest} error={error} />
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <AnomalyChart history={history} threshold={threshold} />
        </div>
        <div>
          <AlertFeed alerts={alerts} />
        </div>
        <div className="lg:col-span-3">
          <SensorGrid latest={latest} />
        </div>
      </div>

      {/* Footer status line */}
      <footer className="mt-8 border-t border-neutral-800 pt-3 flex items-center justify-between">
        <span className="hmi-label">SENTINEL Industrial Monitor</span>
        <span className="hmi-label">VAE · GAUSSIAN ENSEMBLE · AST AUDIO · SHAP/IG</span>
      </footer>
    </div>
  );
}
