"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Alert, ReplayMessage } from "../types";

const API_HOST = process.env.NEXT_PUBLIC_API_HOST ?? "localhost";
const WS_URL = `ws://${API_HOST}:8000/ws/replay?speed_ms=500`;
const MAX_HISTORY = 100;
const MAX_ALERTS = 15;

export interface ReplayState {
  latest: ReplayMessage | null;
  history: ReplayMessage[];
  alerts: Alert[];
  connected: boolean;
  error: string | null;
}

export function useReplaySocket(): ReplayState {
  const [latest, setLatest] = useState<ReplayMessage | null>(null);
  const [history, setHistory] = useState<ReplayMessage[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const alertIdRef = useRef(0);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      const msg: ReplayMessage = JSON.parse(event.data);
      setLatest(msg);
      setHistory((prev) => [...prev.slice(-(MAX_HISTORY - 1)), msg]);

      if (msg.is_anomalous) {
        const alert: Alert = {
          id: alertIdRef.current++,
          step: msg.step,
          score: msg.anomaly_score,
          phase: msg.phase,
          timestamp: new Date().toLocaleTimeString(),
        };
        setAlerts((prev) => [alert, ...prev.slice(0, MAX_ALERTS - 1)]);
      }
    };

    ws.onerror = () => setError("WebSocket error — is the SENTINEL server running?");
    ws.onclose = () => {
      setConnected(false);
      // Auto-reconnect after 3 seconds
      setTimeout(connect, 3000);
    };
  }, []);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  return { latest, history, alerts, connected, error };
}
