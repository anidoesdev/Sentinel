"use client";

import type { Alert } from "../types";

interface Props {
  alerts: Alert[];
}

const phaseLabel: Record<string, { label: string; color: string }> = {
  degrading: { label: "DEGRADING", color: "#d97706" },
  anomalous: { label: "ANOMALOUS", color: "#dc2626" },
};

export default function AlertFeed({ alerts }: Props) {
  return (
    <div className="hmi-panel p-5 flex flex-col h-full">
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="hmi-label mb-0.5">Channel 02</div>
          <h2 className="font-mono font-semibold text-sm tracking-widest uppercase text-neutral-200">
            Alert Log
          </h2>
        </div>
        {alerts.length > 0 && (
          <span className="led font-mono text-sm font-bold text-red-500 border border-red-900 px-2 py-0.5">
            {String(alerts.length).padStart(2, "0")}
          </span>
        )}
      </div>

      {alerts.length === 0 ? (
        <div className="flex-1 flex items-center justify-center">
          <span className="hmi-label text-center leading-7">
            NO ANOMALIES<br />DETECTED
          </span>
        </div>
      ) : (
        <ul className="space-y-1.5 overflow-y-auto max-h-64">
          {alerts.map((a) => {
            const meta = phaseLabel[a.phase] ?? phaseLabel.anomalous;
            return (
              <li
                key={a.id}
                className="flex items-center justify-between px-3 py-2 border border-neutral-800"
                style={{ borderLeft: `3px solid ${meta.color}` }}
              >
                <div>
                  <div className="font-mono text-xs font-bold" style={{ color: meta.color }}>
                    {meta.label}
                  </div>
                  <div className="hmi-label mt-0.5">STEP {a.step}</div>
                </div>
                <div className="text-right">
                  <div className="led font-mono text-sm font-bold text-red-400">
                    {a.score.toFixed(4)}
                  </div>
                  <div className="hmi-label mt-0.5">{a.timestamp}</div>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
