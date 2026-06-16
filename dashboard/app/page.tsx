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
    <div className="min-h-screen bg-slate-900 text-slate-100 p-6">
      {/* Header */}
      <header className="mb-6">
        <div className="flex items-center gap-3">
          <div className="w-2 h-8 bg-blue-500 rounded-full" />
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">SENTINEL</h1>
            <p className="text-slate-400 text-sm">Real-Time Multimodal Anomaly Detection</p>
          </div>
        </div>
      </header>

      {/* System status bar */}
      <div className="mb-6">
        <SystemStatus connected={connected} latest={latest} error={error} />
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Anomaly score chart — spans 2 cols on large screens */}
        <div className="lg:col-span-2">
          <AnomalyChart history={history} threshold={threshold} />
        </div>

        {/* Alert feed */}
        <div>
          <AlertFeed alerts={alerts} />
        </div>

        {/* Sensor health grid — full width */}
        <div className="lg:col-span-3">
          <SensorGrid latest={latest} />
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-8 text-center text-slate-600 text-xs">
        SENTINEL v0.1 · VAE + Gaussian Ensemble · AST Audio · SHAP/IG Explanations
      </footer>
    </div>
  );
}
