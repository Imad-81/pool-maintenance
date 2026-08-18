import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import type { ForecastDay } from "../types";
import IberHeader from "../components/IberHeader";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceArea } from "recharts";
import { useI18n } from "../i18n";

function valClass(v: number | null, low: number, high: number) {
  if (v == null) return "text-blue-300/50";
  if (v < low || v > high) return "text-red-400 font-bold";
  return "text-emerald-400 font-semibold";
}

export default function PoolDetailPage() {
  const { poolId } = useParams<{ poolId: string }>();
  const navigate = useNavigate();
  const { t } = useI18n();
  const [horizon, setHorizon] = useState(3);

  const { data, isLoading, error } = useQuery({
    queryKey: ["pool", poolId, horizon],
    queryFn: () => api.pool(poolId!, horizon),
    enabled: !!poolId,
  });

  if (isLoading) {
    return (
      <div className="min-h-screen bg-caustic text-white flex flex-col">
        <IberHeader subtitle={t("detail_subtitle")} />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center p-8 glass-panel rounded-2xl">
            <div className="text-3xl animate-bounce mb-3">🌊</div>
            <p className="text-blue-200">{t("detail_loading")} {poolId}...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-caustic text-white flex flex-col">
        <IberHeader subtitle={t("detail_subtitle")} />
        <div className="flex-1 flex items-center justify-center p-4">
          <div className="text-center p-8 glass-panel rounded-2xl max-w-md">
            <p className="text-red-400 font-semibold mb-4">Error: {error?.message}</p>
            <button
              onClick={() => navigate("/piscinas")}
              className="px-4 py-2 bg-blue-600 rounded-xl text-white font-medium"
            >
              {t("detail_back_to_pools")}
            </button>
          </div>
        </div>
      </div>
    );
  }

  const getUrgencyBadge = (urg: string) => {
    if (urg === "Immediate" || urg === "URGENT") {
      return { label: t("urg_immediate"), bg: "bg-red-500/20 border-red-500/40", text: "text-red-400" };
    }
    if (urg === "Advised" || urg === "Soon" || urg === "Monitor") {
      return { label: t("urg_advised"), bg: "bg-amber-500/20 border-amber-500/40", text: "text-amber-400" };
    }
    return { label: t("urg_routine"), bg: "bg-emerald-500/20 border-emerald-500/40", text: "text-emerald-400" };
  };

  const badgeInfo = getUrgencyBadge(data.optimiser?.urgency || (data.visit_needed ? "Immediate" : "Routine"));

  const chartData = [
    ...(data.history || []).slice(-10).map((h) => ({
      date: h.reading_date.slice(5, 10),
      cl: h.free_chlorine,
      ph: h.ph,
      turb: h.turbidity,
      type: "History",
    })),
    ...(data.forecast || []).map((f) => ({
      date: f.date.slice(5, 10),
      cl: f.predicted_cl,
      ph: f.predicted_ph,
      turb: f.predicted_turb,
      type: "Forecast",
    })),
  ];

  return (
    <div className="min-h-screen bg-caustic text-white flex flex-col">
      <IberHeader subtitle={`${t("detail_subtitle")} — ${data.community_name || data.pool_id}`} />

      <main className="flex-1 max-w-[1400px] w-full mx-auto px-4 md:px-8 pb-16">
        {/* Navigation Breadcrumb */}
        <div className="flex items-center justify-between gap-4 mb-6">
          <button
            onClick={() => navigate("/piscinas")}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl glass-card text-blue-200 hover:text-white text-xs font-semibold"
          >
            {t("detail_back_to_pools")}
          </button>
          <div className="flex items-center gap-2">
            <span className={`px-3 py-1.5 rounded-full text-xs font-semibold border ${badgeInfo.bg} ${badgeInfo.text}`}>
              {badgeInfo.label}
            </span>
            <span className="text-xs text-blue-300/60 hidden sm:inline">{t("detail_normative")}</span>
          </div>
        </div>

        {/* Pool Title & Summary Banner */}
        <div className="glass-panel rounded-2xl p-6 mb-6">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <span className="text-xs text-blue-300 font-mono">{data.pool_id}</span>
              <h2 className="text-2xl md:text-3xl font-extrabold text-white font-heading">
                {data.community_name || "Iberpiscinas"}
              </h2>
              <p className="text-xs text-blue-200/80 mt-1">
                {t("detail_volume")}: <strong className="text-white">{data.pool_volume_m3 || 150} m³</strong> | {t("detail_source")}: <strong className="text-cyan-300">{data.prediction.source}</strong>
              </p>
            </div>

            {/* Current Measurements */}
            <div className="flex items-center gap-3 bg-blue-950/60 p-3 rounded-xl border border-blue-800/40">
              <div className="text-center px-2">
                <div className="text-[10px] text-blue-300/70">{t("pools_cl")}</div>
                <div className={`text-base ${valClass(data.latest?.free_chlorine ?? null, 0.5, 2.0)}`}>
                  {data.latest?.free_chlorine != null ? `${data.latest.free_chlorine.toFixed(2)} mg/L` : "—"}
                </div>
              </div>
              <div className="h-8 w-px bg-blue-800/60" />
              <div className="text-center px-2">
                <div className="text-[10px] text-blue-300/70">{t("pools_ph")}</div>
                <div className={`text-base ${valClass(data.latest?.ph ?? null, 7.2, 8.0)}`}>
                  {data.latest?.ph != null ? data.latest.ph.toFixed(2) : "—"}
                </div>
              </div>
              <div className="h-8 w-px bg-blue-800/60" />
              <div className="text-center px-2">
                <div className="text-[10px] text-blue-300/70">{t("pools_turb")}</div>
                <div className={`text-base ${valClass(data.latest?.turbidity ?? null, 0, 5)}`}>
                  {data.latest?.turbidity != null ? `${data.latest.turbidity.toFixed(1)} NTU` : "—"}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Today & Tomorrow Forecast Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {data.forecast.slice(0, 2).map((day, idx) => (
            <ForecastTile key={day.date} day={day} label={idx === 0 ? t("detail_today_forecast") : t("detail_tomorrow_forecast")} isPrimary={idx === 0} />
          ))}
        </div>

        {/* Chemical Dosing Optimizer Recommendation */}
        {data.optimiser && (
          <div className="glass-panel rounded-2xl p-6 mb-6 border border-cyan-500/30">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-xl">🧪</span>
              <h3 className="text-lg font-bold text-white font-heading">
                {t("detail_optimizer_title")}
              </h3>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 bg-blue-950/50 p-4 rounded-xl border border-blue-800/30 mb-3">
              <div>
                <span className="text-xs text-blue-300/70 block">{t("detail_pump_power")}</span>
                <span className="text-xl font-bold text-cyan-300 font-heading">
                  {data.optimiser.recommended_dosing.hypochlorite_dosing_pct}%
                </span>
              </div>
              <div>
                <span className="text-xs text-blue-300/70 block">{t("detail_recirc_time")}</span>
                <span className="text-xl font-bold text-cyan-300 font-heading">
                  {data.optimiser.recommended_dosing.hypochlorite_dosing_hours} h
                </span>
              </div>
              <div>
                <span className="text-xs text-blue-300/70 block">{t("detail_projected_cl")}</span>
                <span className="text-xl font-bold text-emerald-400 font-heading">
                  {data.optimiser.predicted_tomorrow.free_chlorine.toFixed(2)} mg/L
                </span>
              </div>
            </div>
            {data.optimiser.reasons?.length > 0 && (
              <p className="text-xs text-blue-200/80">
                ℹ️ {data.optimiser.reasons.join(" • ")}
              </p>
            )}
          </div>
        )}

        {/* Prediction Chart (History + Horizon Forecast) */}
        <div className="glass-panel rounded-2xl p-6 mb-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
            <div>
              <h3 className="text-lg font-bold text-white font-heading">
                {t("detail_chart_title")}
              </h3>
              <p className="text-xs text-blue-300/70">
                {t("detail_chart_subtitle", { horizon })}
              </p>
            </div>
            <div className="flex items-center gap-1 bg-blue-950 p-1 rounded-xl border border-blue-800/40">
              {[2, 3, 5, 7].map((h) => (
                <button
                  key={h}
                  onClick={() => setHorizon(h)}
                  className={`px-3 py-1 text-xs rounded-lg font-medium transition ${
                    horizon === h ? "bg-blue-600 text-white shadow" : "text-blue-300 hover:text-white"
                  }`}
                >
                  {h} {t("detail_chart_days")}
                </button>
              ))}
            </div>
          </div>

          <div className="h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 20, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="date" stroke="#93c5fd" fontSize={11} />
                <YAxis stroke="#93c5fd" fontSize={11} domain={[0, "auto"]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#061e40",
                    borderColor: "#1877f2",
                    borderRadius: "12px",
                    color: "#fff",
                    fontSize: "12px",
                  }}
                />
                <ReferenceArea y1={0.5} y2={2.0} fill="#10b981" fillOpacity={0.07} label={t("detail_chart_optimal_range")} />
                <Line
                  type="monotone"
                  dataKey="cl"
                  name={t("detail_chart_cl_series")}
                  stroke="#38bdf8"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: "#38bdf8" }}
                />
                <Line
                  type="monotone"
                  dataKey="ph"
                  name={t("detail_chart_ph_series")}
                  stroke="#fbbf24"
                  strokeWidth={2}
                  dot={{ r: 3, fill: "#fbbf24" }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </main>
    </div>
  );
}

function ForecastTile({ day, label, isPrimary }: { day?: ForecastDay; label: string; isPrimary?: boolean }) {
  const { t } = useI18n();
  if (!day) return null;
  const isBreach = day.cl_breach || day.ph_breach;

  return (
    <div className={`glass-card rounded-2xl p-5 ${isPrimary ? "border-blue-400/50 bg-blue-900/40" : ""}`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-blue-200">{label}</span>
        <span className="text-xs text-blue-300/60 font-mono">{day.date}</span>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center bg-blue-950/60 p-3 rounded-xl border border-blue-800/30 mb-2">
        <div>
          <span className="text-[10px] text-blue-300/70 block">{t("detail_forecast_cl")}</span>
          <span className={`text-base ${day.cl_breach ? "text-red-400 font-bold" : "text-emerald-400 font-semibold"}`}>
            {day.predicted_cl.toFixed(2)} mg/L
          </span>
        </div>
        <div>
          <span className="text-[10px] text-blue-300/70 block">{t("detail_forecast_ph")}</span>
          <span className={`text-base ${day.ph_breach ? "text-red-400 font-bold" : "text-emerald-400 font-semibold"}`}>
            {day.predicted_ph.toFixed(2)}
          </span>
        </div>
        <div>
          <span className="text-[10px] text-blue-300/70 block">{t("detail_forecast_turb")}</span>
          <span className="text-base text-blue-200 font-semibold">
            {day.predicted_turb.toFixed(1)} NTU
          </span>
        </div>
      </div>

      <div className="flex items-center justify-between text-xs">
        <span className={isBreach ? "text-red-400 font-medium" : "text-emerald-400"}>
          {isBreach ? t("detail_out_range") : t("detail_in_range")}
        </span>
        <span className="text-blue-300/60 text-[11px]">{t("detail_day_offset")} +{day.day_offset_from_today ?? 0}</span>
      </div>
    </div>
  );
}
