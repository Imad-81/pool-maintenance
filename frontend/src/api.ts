import type {
  FleetResponse,
  PoolDetail,
  StatusResponse,
  OptimiserResult,
  ModelRun,
  IngestLog,
  UploadPreview,
  HealthReadyResponse,
} from "./types";

const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${errorText}`);
  }
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${errorText}`);
  }
  return res.json();
}

export const api = {
  status: () => get<StatusResponse>("/status"),
  healthReady: () => get<HealthReadyResponse>("/../healthz/ready"),

  fleet: (params?: { date?: string; q?: string; urgency?: string; page?: number; page_size?: number }) => {
    const sp = new URLSearchParams();
    if (params?.date) sp.set("date", params.date);
    if (params?.q) sp.set("q", params.q);
    if (params?.urgency) sp.set("urgency", params.urgency);
    if (params?.page !== undefined) sp.set("page", String(params.page));
    if (params?.page_size !== undefined) sp.set("page_size", String(params.page_size));
    const qs = sp.toString();
    return get<FleetResponse>(`/fleet${qs ? "?" + qs : ""}`);
  },

  fleetSummary: (date?: string) => {
    const qs = date ? `?date=${encodeURIComponent(date)}` : "";
    return get<import("./types").FleetSummary>(`/fleet/summary${qs}`);
  },


  pool: (id: string, horizon?: number) => {
    const qs = horizon !== undefined ? `?horizon=${horizon}` : "";
    return get<PoolDetail>(`/pool/${encodeURIComponent(id)}${qs}`);
  },

  optimise: (poolId: string) => get<OptimiserResult>(`/optimise/${encodeURIComponent(poolId)}`),

  poolIds: () => get<string[]>("/fleet/pool-ids"),

  dates: () => get<{ min: string; max: string; count: number }>("/fleet/dates"),

  addReading: (data: {
    pool_id: string;
    reading_date: string;
    ph?: number | null;
    free_chlorine?: number | null;
    turbidity?: number | null;
    pool_volume_m3?: number | null;
    community_name?: string;
    water_temperature?: number | null;
    hypochlorite_dosing_pct?: number | null;
    hypochlorite_dosing_hours?: number | null;
  }) => post<{ success: boolean; pool_id: string; rows: number }>("/readings", data),

  uploadFile: async (file: File): Promise<UploadPreview> => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${BASE}/upload`, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Upload failed (${res.status}): ${err}`);
    }
    return res.json();
  },

  mapColumns: (uploadId: string, mapping: Record<string, string>) =>
    post<{ success: boolean; loaded_rows: number; pool_count: number; skipped_count: number; skipped: unknown[] }>(
      "/map-columns",
      { upload_id: uploadId, mapping }
    ),

  admin: {
    runs: () => get<ModelRun[]>("/admin/runs"),
    retrain: () => post<{ status: string; result?: Record<string, unknown> }>("/admin/retrain", {}),
    weather: () => post<{ status: string; rows_upserted: number }>("/admin/weather-refresh", {}),
    weatherStatus: () => get<{ latest_weather_date: string; timestamp: string }>("/admin/weather-status"),
    ingestLog: () => get<IngestLog[]>("/admin/ingest-log"),
  },
};
