import type { FleetResponse, PoolDetail, StatusResponse, OptimiserResult, ModelRun } from "./types";

const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  status: () => get<StatusResponse>("/status"),

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

  pool: (id: string, horizon?: number) => {
    const qs = horizon !== undefined ? `?horizon=${horizon}` : "";
    return get<PoolDetail>(`/pool/${encodeURIComponent(id)}${qs}`);
  },

  optimise: (poolId: string) => get<OptimiserResult>(`/optimise/${encodeURIComponent(poolId)}`),

  poolIds: () => get<string[]>("/fleet/pool-ids"),

  dates: () => get<{ min: string; max: string; count: number }>("/fleet/dates"),

  addReading: (data: Record<string, unknown>) => post<{ success: boolean }>("/readings", data),

  uploadFile: async (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${BASE}/upload`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(`${res.status}`);
    return res.json();
  },

  mapColumns: (mapping: Record<string, string>) =>
    post<{ success: boolean; loaded_rows: number; skipped_count: number; skipped: unknown[] }>("/map-columns", { mapping }),

  admin: {
    runs: () => get<ModelRun[]>("/admin/runs"),
    retrain: () => post<{ status: string; result?: Record<string, unknown> }>("/admin/retrain", {}),
    weather: () => post<{ status: string; rows_upserted: number }>("/admin/weather-refresh", {}),
  },
};
