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
  healthy: "#16a34a",
  degrading: "#d97706",
  anomalous: "#dc2626",
};

export default function AnomalyChart({ history, threshold }: Props) {
  const data = history.map((m) => ({
    step: m.step,
    score: m.anomaly_score,
    phase: m.phase,
  }));

  const latest = history[history.length - 1];
  const dotColor = latest ? (phaseColor[latest.phase] ?? "#737373") : "#737373";

  return (
    <div className="hmi-panel p-5">
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="hmi-label mb-0.5">Channel 01</div>
          <h2 className="font-mono font-semibold text-sm tracking-widest uppercase text-neutral-200">
            Anomaly Score
          </h2>
        </div>
        {latest && (
          <div className="text-right">
            <div className="hmi-label mb-0.5">Live Reading</div>
            <span className="led text-2xl font-bold" style={{ color: dotColor }}>
              {latest.anomaly_score.toFixed(4)}
            </span>
          </div>
        )}
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" />
          <XAxis
            dataKey="step"
            tick={{ fill: "#525252", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
            tickLine={false}
            axisLine={{ stroke: "#262626" }}
            label={{ value: "STEP", position: "insideBottomRight", fill: "#525252", fontSize: 10 }}
          />
          <YAxis
            tick={{ fill: "#525252", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
            tickLine={false}
            axisLine={{ stroke: "#262626" }}
            width={50}
          />
          <Tooltip
            contentStyle={{
              background: "#0d0d0d",
              border: "1px solid #3b82f6",
              borderRadius: 0,
            }}
            labelStyle={{ color: "#737373", fontFamily: "var(--font-geist-mono)", fontSize: 11 }}
            itemStyle={{ color: "#d4d4d4", fontFamily: "var(--font-geist-mono)" }}
          />
          <ReferenceLine
            y={threshold}
            stroke="#dc2626"
            strokeDasharray="6 3"
            label={{
              value: "THRESH",
              fill: "#dc2626",
              fontSize: 10,
              position: "right",
              fontFamily: "var(--font-geist-mono)",
            }}
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
