"use client";

import type { ReplayMessage } from "../types";

interface Props {
  latest: ReplayMessage | null;
}

function sensorStatus(value: number): { color: string; bg: string; label: string } {
  const abs = Math.abs(value);
  if (abs < 1.5) return { color: "#16a34a", bg: "#0a1a0c", label: "NOM" };
  if (abs < 2.5) return { color: "#d97706", bg: "#1a1100", label: "CAU" };
  return { color: "#dc2626", bg: "#1a0808", label: "CRT" };
}

const legend = [
  { color: "#16a34a", label: "NOM — Normal (<1.5σ)" },
  { color: "#d97706", label: "CAU — Caution (1.5–2.5σ)" },
  { color: "#dc2626", label: "CRT — Critical (>2.5σ)" },
];

export default function SensorGrid({ latest }: Props) {
  const sensors = latest?.sensors ?? {};
  const entries = Object.entries(sensors);

  return (
    <div className="hmi-panel p-5">
      <div className="mb-4">
        <div className="hmi-label mb-0.5">Channel 03</div>
        <h2 className="font-mono font-semibold text-sm tracking-widest uppercase text-neutral-200">
          Sensor Health
        </h2>
      </div>

      {entries.length === 0 ? (
        <p className="hmi-label">AWAITING SIGNAL...</p>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
          {entries.map(([name, value]) => {
            const { color, bg, label } = sensorStatus(value);
            return (
              <div
                key={name}
                className="px-3 py-2 border flex items-center justify-between"
                style={{
                  background: bg,
                  borderColor: color + "33",
                  borderLeft: `3px solid ${color}`,
                }}
              >
                <div>
                  <div className="font-mono text-xs text-neutral-600">{name}</div>
                  <div className="font-mono text-xs font-bold mt-0.5" style={{ color }}>
                    {label}
                  </div>
                </div>
                <div className="led font-mono text-sm font-bold" style={{ color }}>
                  {value.toFixed(2)}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="flex flex-wrap gap-6 mt-4">
        {legend.map(({ color, label }) => (
          <span key={label} className="hmi-label flex items-center gap-2">
            <span className="w-2 h-2 rounded-full" style={{ background: color }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
