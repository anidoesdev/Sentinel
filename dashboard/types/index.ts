export interface ReplayMessage {
  step: number;
  unit_id: number;
  anomaly_score: number;
  is_anomalous: boolean;
  threshold: number;
  sensors: Record<string, number>;
  phase: "healthy" | "degrading" | "anomalous";
}

export interface Alert {
  id: number;
  step: number;
  score: number;
  phase: string;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  models_loaded: string[];
  models_missing: string[];
}
