"use client";

import { useEffect, useState } from "react";
import type { HealthResponse, ReplayMessage } from "../types";

interface Props {
  connected: boolean;
  latest: ReplayMessage | null;
  error: string | null;
}

const phaseInfo: Record<string, { color: string; label: string }> = {
  healthy:   { color: "#16a34a", label: "HEALTHY" },
  degrading: { color: "#d97706", label: "DEGRADING" },
  anomalous: { color: "#dc2626", label: "ANOMALOUS" },
};

export default function SystemStatus({ connected, latest, error }: Props) {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
    const check = () =>
      fetch(`${apiBase}/health`)
        .then((r) => r.json())
        .then(setHealth)
        .catch(() => setHealth(null));

    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, []);

  const phase = latest?.phase ?? null;
  const info = phase ? (phaseInfo[phase] ?? { color: "#737373", label: phase.toUpperCase() }) : null;

  return (
    <div className="hmi-panel px-5 py-4 flex flex-wrap items-center gap-x-8 gap-y-3">
      {/* Link status */}
      <div>
        <div className="hmi-label mb-1">Link Status</div>
        <div className="flex items-center gap-2">
          <span
            className={`w-2.5 h-2.5 rounded-full ${connected ? "animate-pulse" : ""}`}
            style={{ background: connected ? "#16a34a" : "#dc2626" }}
          />
          <span
            className="led font-mono text-sm font-bold"
            style={{ color: connected ? "#16a34a" : "#dc2626" }}
          >
            {connected ? "LIVE" : "FAULT"}
          </span>
        </div>
      </div>

      <div className="w-px h-8 bg-neutral-800 hidden sm:block" />

      {/* Machine phase */}
      <div>
        <div className="hmi-label mb-1">Machine Phase</div>
        {info ? (
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full" style={{ background: info.color }} />
            <span className="font-mono text-sm font-bold" style={{ color: info.color }}>
              {info.label}
            </span>
          </div>
        ) : (
          <span className="hmi-label">—</span>
        )}
      </div>

      <div className="w-px h-8 bg-neutral-800 hidden sm:block" />

      {/* Step counter */}
      <div>
        <div className="hmi-label mb-1">Step Counter</div>
        <span className="led font-mono text-sm font-bold text-neutral-300">
          {latest ? String(latest.step).padStart(6, "0") : "------"}
        </span>
      </div>

      {/* Models */}
      {health && (
        <>
          <div className="w-px h-8 bg-neutral-800 hidden sm:block" />
          <div>
            <div className="hmi-label mb-1">Models Online</div>
            <div className="flex items-center gap-2">
              {["vae", "audio"].map((m) => (
                <span
                  key={m}
                  className="font-mono text-xs px-2 py-0.5 border"
                  style={
                    health.models_loaded.includes(m)
                      ? { color: "#16a34a", borderColor: "#16a34a55", background: "#0a1a0c" }
                      : { color: "#525252", borderColor: "#262626", background: "#141414" }
                  }
                >
                  {m.toUpperCase()}
                </span>
              ))}
            </div>
          </div>
        </>
      )}

      {error && (
        <div className="ml-auto">
          <div className="hmi-label mb-1">System Error</div>
          <p className="font-mono text-xs text-red-400">{error}</p>
        </div>
      )}
    </div>
  );
}
