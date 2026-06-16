"use client";

import type { Alert } from "../types";

interface Props {
  alerts: Alert[];
}

const phaseLabel: Record<string, { label: string; classes: string }> = {
  degrading: { label: "DEGRADING", classes: "bg-yellow-900 text-yellow-300 border-yellow-700" },
  anomalous: { label: "ANOMALOUS", classes: "bg-red-900 text-red-300 border-red-700" },
};

export default function AlertFeed({ alerts }: Props) {
  return (
    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 flex flex-col">
      <h2 className="text-slate-200 font-semibold text-sm tracking-wide uppercase mb-4">
        Anomaly Alerts
        {alerts.length > 0 && (
          <span className="ml-2 bg-red-600 text-white text-xs px-2 py-0.5 rounded-full">
            {alerts.length}
          </span>
        )}
      </h2>

      {alerts.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
          No anomalies detected
        </div>
      ) : (
        <ul className="space-y-2 overflow-y-auto max-h-64">
          {alerts.map((a) => {
            const meta = phaseLabel[a.phase] ?? phaseLabel.anomalous;
            return (
              <li
                key={a.id}
                className="flex items-center justify-between rounded-lg bg-slate-900 px-3 py-2 border border-slate-700"
              >
                <div className="flex items-center gap-3">
                  <span
                    className={`text-xs font-semibold px-2 py-0.5 rounded border ${meta.classes}`}
                  >
                    {meta.label}
                  </span>
                  <span className="text-slate-400 text-xs font-mono">step {a.step}</span>
                </div>
                <div className="text-right">
                  <span className="text-red-400 font-mono text-sm font-bold">
                    {a.score.toFixed(4)}
                  </span>
                  <div className="text-slate-500 text-xs">{a.timestamp}</div>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
