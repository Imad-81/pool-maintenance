export interface FleetItem {
  pool_id: string;
  community_name: string;
  last_reading_date: string;
  ph: number | null;
  free_chlorine: number | null;
  turbidity: number | null;
  urgency: string;
  breach_proba: number;
  prediction_source: string;
  today_forecast: ForecastDay | null;
  tomorrow_forecast: ForecastDay | null;
}

export interface ForecastDay {
  date: string;
  day: string;
  days_from_visit: number;
  day_offset_from_today: number;
  predicted_cl: number;
  predicted_ph: number;
  predicted_turb: number;
  cl_breach: boolean;
  ph_breach: boolean;
  urgency: string;
  status: string;
  is_today: boolean;
  is_tomorrow: boolean;
  uncertainty_band?: {
    cl_low: number; cl_high: number;
    ph_low: number; ph_high: number;
    turb_low: number; turb_high: number;
  };
}

export interface PoolDetail {
  pool_id: string;
  community_name: string;
  latest: { reading_date: string; ph: number | null; free_chlorine: number | null; turbidity: number | null; water_temperature?: number | null };
  forecast: ForecastDay[];
  visit_needed: boolean;
  today_forecast: ForecastDay[];
  tomorrow_forecast: ForecastDay[];
  prediction: { source: string; error?: string };
  history: HistoryPoint[];
  pool_volume_m3: number | null;
  optimiser?: OptimiserResult;
}

export interface HistoryPoint {
  pool_id: string;
  reading_date: string;
  ph: number | null;
  free_chlorine: number | null;
  turbidity: number | null;
  water_temperature: number | null;
  hypochlorite_dosing_pct?: number | null;
  hypochlorite_dosing_hours?: number | null;
}

export interface OptimiserResult {
  recommended_dosing: { hypochlorite_dosing_pct: number; hypochlorite_dosing_hours: number };
  predicted_tomorrow: { free_chlorine: number; ph: number };
  feasible_configurations: number;
  top_3_configs: Record<string, number>[];
  urgency: string;
  reasons: string[];
}

export interface FleetResponse {
  items: FleetItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface FleetSummary {
  total: number;
  counts: {
    Immediate: number;
    Advised: number;
    Routine: number;
    Extended: number;
  };
  compliance_rate: number;
  as_of_date: string;
}


export interface StatusResponse {
  status: string;
  prediction: { loaded: boolean; run_id?: string; metrics?: Record<string, unknown> };
}

export interface HealthReadyResponse {
  status: string;
  database: string;
  model: { loaded: boolean; run_id?: string };
  timestamp: string;
}

export interface ModelRun {
  run_id: string;
  created_at: string;
  is_active: boolean;
  metrics: Record<string, Record<string, number>> | null;
  promoted_at: string | null;
  promote_reason: string | null;
}

export interface IngestLog {
  id: number;
  source: string;
  filename: string | null;
  pool_count: number;
  row_count: number;
  skipped_count: number;
  created_at: string;
  detail?: Record<string, unknown> | null;
}

export interface UploadPreview {
  upload_id: string;
  filename: string;
  total_rows: number;
  columns: string[];
  suggested_mapping: Record<string, string>;
  preview: Record<string, string>[];
}
