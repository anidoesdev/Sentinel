"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ReplayMessage } from "../types";

interface Props {
  history: ReplayMessage[];
  threshold: number;
}

const phaseColor: Record<string, string> = {
  healthy: "#22c55e",
  degrading: "#f59e0b",
  anomalous: "#ef4444",
};

export default function AnomalyChart({ history, threshold }: Props) {
  const data = history.map((m) => ({
    step: m.step,
    score: m.anomaly_score,
    phase: m.phase,
  }));

  const latest = history[history.length - 1];
  const dotColor = latest ? (phaseColor[latest.phase] ?? "#94a3b8") : "#94a3b8";

  return (
    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-slate-200 font-semibold text-sm tracking-wide uppercase">
          Live Anomaly Score
        </h2>
        {latest && (
          <span
            className="text-2xl font-mono font-bold"
            style={{ color: dotColor }}
          >
            {latest.anomaly_score.toFixed(4)}
          </span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="step"
            tick={{ fill: "#64748b", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            label={{ value: "Step", position: "insideBottomRight", fill: "#64748b", fontSize: 11 }}
          />
          <YAxis
            tick={{ fill: "#64748b", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={50}
          />
          <Tooltip
            contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8 }}
            labelStyle={{ color: "#94a3b8" }}
            itemStyle={{ color: "#f1f5f9" }}
          />
          <ReferenceLine
            y={threshold}
            stroke="#ef4444"
            strokeDasharray="6 3"
            label={{ value: "threshold", fill: "#ef4444", fontSize: 10, position: "right" }}
          />
          <Area
            type="monotone"
            dataKey="score"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#scoreGrad)"
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
