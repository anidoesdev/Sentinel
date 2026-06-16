"use client";

import type { ReplayMessage } from "../types";

interface Props {
  latest: ReplayMessage | null;
}

function sensorColor(value: number): string {
  const abs = Math.abs(value);
  if (abs < 1.5) return "bg-green-900 border-green-700 text-green-300";
  if (abs < 2.5) return "bg-yellow-900 border-yellow-700 text-yellow-300";
  return "bg-red-900 border-red-700 text-red-300";
}

export default function SensorGrid({ latest }: Props) {
  const sensors = latest?.sensors ?? {};
  const entries = Object.entries(sensors);

  return (
    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
      <h2 className="text-slate-200 font-semibold text-sm tracking-wide uppercase mb-4">
        Sensor Health
      </h2>

      {entries.length === 0 ? (
        <p className="text-slate-500 text-sm">Waiting for data…</p>
      ) : (
        <div className="grid grid-cols-2 gap-2">
          {entries.map(([name, value]) => (
            <div
              key={name}
              className={`rounded-lg border px-3 py-2 flex items-center justify-between ${sensorColor(value)}`}
            >
              <span className="text-xs font-mono opacity-80">{name}</span>
              <span className="text-xs font-mono font-bold">{value.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}

      <div className="flex gap-4 mt-4 text-xs text-slate-400">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-green-500" /> Normal (&lt;1.5σ)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-yellow-500" /> Caution (1.5–2.5σ)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-red-500" /> Critical (&gt;2.5σ)
        </span>
      </div>
    </div>
  );
}
