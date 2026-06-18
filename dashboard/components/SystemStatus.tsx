"use client";

import { useEffect, useState } from "react";
import type { HealthResponse, ReplayMessage } from "../types";

interface Props {
  connected: boolean;
  latest: ReplayMessage | null;
  error: string | null;
}

const phaseStyle: Record<string, string> = {
  healthy: "text-green-400",
  degrading: "text-yellow-400",
  anomalous: "text-red-400",
};

export default function SystemStatus({ connected, latest, error }: Props) {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    const apiHost = process.env.NEXT_PUBLIC_API_HOST ?? "localhost";
    const check = () =>
      fetch(`http://${apiHost}:8000/health`)
        .then((r) => r.json())
        .then(setHealth)
        .catch(() => setHealth(null));

    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, []);

  const phase = latest?.phase ?? "—";
  const phaseClass = phaseStyle[phase] ?? "text-slate-400";

  return (
    <div className="bg-slate-800 rounded-xl px-5 py-4 border border-slate-700 flex flex-wrap items-center gap-6">
      {/* Connection status */}
      <div className="flex items-center gap-2">
        <span
          className={`w-2.5 h-2.5 rounded-full ${connected ? "bg-green-400 animate-pulse" : "bg-red-500"}`}
        />
        <span className="text-slate-300 text-sm">
          {connected ? "Live" : "Disconnected"}
        </span>
      </div>

      {/* Current phase */}
      <div className="text-sm">
        <span className="text-slate-400">Phase: </span>
        <span className={`font-semibold uppercase ${phaseClass}`}>{phase}</span>
      </div>

      {/* Step counter */}
      {latest && (
        <div className="text-sm font-mono">
          <span className="text-slate-400">Step: </span>
          <span className="text-slate-200">{latest.step}</span>
        </div>
      )}

      {/* Model status from /health */}
      {health && (
        <div className="flex items-center gap-3 ml-auto">
          {["vae", "audio"].map((m) => (
            <span
              key={m}
              className={`text-xs px-2 py-0.5 rounded-full border font-mono ${
                health.models_loaded.includes(m)
                  ? "bg-green-900 text-green-300 border-green-700"
                  : "bg-slate-700 text-slate-400 border-slate-600"
              }`}
            >
              {m.toUpperCase()}
            </span>
          ))}
        </div>
      )}

      {error && (
        <p className="text-red-400 text-xs ml-auto">{error}</p>
      )}
    </div>
  );
}
