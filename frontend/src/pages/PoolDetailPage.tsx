import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Calendar,
  Target,
  AlertTriangle,
  ArrowLeft,
  Loader2,
  CheckCircle2,
} from "lucide-react";
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
            <Loader2 size={36} className="animate-spin text-cyan-400 mx-auto mb-3" />
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
              onClick={() => navigate("/")}
              className="inline-flex items-center gap-1.5 px-4 py-2 bg-blue-600 rounded-xl text-white font-medium cursor-pointer"
            >
              <ArrowLeft size={14} />
              <span>{t("detail_back_to_pools")}</span>
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
            onClick={() => navigate("/")}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl glass-card text-blue-200 hover:text-white text-xs font-semibold cursor-pointer"
          >
            <ArrowLeft size={14} />
            <span>{t("detail_back_to_pools")}</span>
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

            {/* Pre-Treatment Measurements (Last Visit) */}
            <div className="flex flex-col items-start md:items-end gap-1.5">
              <div className="flex items-center gap-1.5 text-[11px] text-blue-300/80 font-medium">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                <span>{t("detail_last_pretreatment_title")}</span>
                {data.latest?.reading_date && (
                  <span className="text-blue-300/60 font-mono text-[10px]">
                    ({data.latest.reading_date.slice(0, 10)})
                  </span>
                )}
              </div>
              <div className="flex items-center gap-3 bg-blue-950/60 p-3 rounded-xl border border-blue-800/40 shadow-inner">
                <div className="text-center px-2">
                  <div className="text-[10px] text-blue-300/70">{t("detail_latest_cl")}</div>
                  <div className={`text-base ${valClass(data.latest?.free_chlorine ?? null, 0.5, 2.0)}`}>
                    {data.latest?.free_chlorine != null ? `${data.latest.free_chlorine.toFixed(2)} mg/L` : "—"}
                  </div>
                </div>
                <div className="h-8 w-px bg-blue-800/60" />
                <div className="text-center px-2">
                  <div className="text-[10px] text-blue-300/70">{t("detail_latest_ph")}</div>
                  <div className={`text-base ${valClass(data.latest?.ph ?? null, 7.2, 8.0)}`}>
                    {data.latest?.ph != null ? data.latest.ph.toFixed(2) : "—"}
                  </div>
                </div>
                <div className="h-8 w-px bg-blue-800/60" />
                <div className="text-center px-2">
                  <div className="text-[10px] text-blue-300/70">{t("detail_latest_turb")}</div>
                  <div className={`text-base ${valClass(data.latest?.turbidity ?? null, 0, 5)}`}>
                    {data.latest?.turbidity != null ? `${data.latest.turbidity.toFixed(1)} NTU` : "—"}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Recommended Next Technical Visit Hero Card */}
        {data.recommended_visit && (
          <div
            className={`glass-panel rounded-2xl p-6 mb-6 border transition-all ${
              data.recommended_visit.urgency === "Immediate" || data.recommended_visit.is_breach
                ? "border-red-500/50 bg-gradient-to-br from-red-950/40 via-blue-950/50 to-slate-900/70 shadow-lg shadow-red-950/20"
                : data.recommended_visit.urgency === "Advised"
                ? "border-amber-500/50 bg-gradient-to-br from-amber-950/40 via-blue-950/50 to-slate-900/70 shadow-lg shadow-amber-950/20"
                : "border-emerald-500/40 bg-gradient-to-br from-emerald-950/30 via-blue-950/50 to-slate-900/70"
            }`}
          >
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
              <div className="flex items-center gap-3.5">
                <div
                  className={`w-12 h-12 rounded-2xl flex items-center justify-center shadow-inner ${
                    data.recommended_visit.urgency === "Immediate"
                      ? "bg-red-600/30 border border-red-500/50 text-red-300"
                      : data.recommended_visit.urgency === "Advised"
                      ? "bg-amber-600/30 border border-amber-500/50 text-amber-300"
                      : "bg-emerald-600/30 border border-emerald-500/50 text-emerald-300"
                  }`}
                >
                  <Calendar size={24} />
                </div>
                <div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <h3 className="text-lg md:text-xl font-extrabold text-white font-heading">
                      {t("rec_visit_card_title")}
                    </h3>
                    <span className="text-[11px] px-2.5 py-0.5 rounded-full font-mono font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
                      {t("rec_visit_target_badge")}
                    </span>
                  </div>
                  <p className="text-xs text-blue-200/90 mt-1">
                    <span className="font-semibold text-blue-300">{t("rec_visit_reason_label")}:</span> {data.recommended_visit.reason}
                  </p>
                </div>
              </div>

              {/* Date & Urgency Countdown Pill */}
              <div className="flex items-center gap-2 self-start md:self-auto">
                <div className="bg-blue-950/80 px-4 py-2 rounded-xl border border-blue-800/50 text-right">
                  <span className="text-[10px] uppercase font-bold tracking-wider text-blue-300/70 block">
                    {t("rec_visit_date_label")}
                  </span>
                  <span className="text-sm md:text-base font-bold text-white font-mono">
                    {data.recommended_visit.date}
                  </span>
                </div>
                <div
                  className={`px-4 py-2 rounded-xl border text-center font-bold text-xs md:text-sm ${
                    data.recommended_visit.day_offset_from_today === 0
                      ? "bg-red-500/30 border-red-400 text-red-300 animate-pulse"
                      : data.recommended_visit.day_offset_from_today === 1
                      ? "bg-amber-500/30 border-amber-400 text-amber-300"
                      : "bg-blue-600/30 border-blue-400 text-blue-200"
                  }`}
                >
                  {data.recommended_visit.day_offset_from_today === 0
                    ? t("rec_visit_today")
                    : data.recommended_visit.day_offset_from_today === 1
                    ? t("rec_visit_tomorrow")
                    : t("rec_visit_in_days", { days: data.recommended_visit.day_offset_from_today })}
                </div>
              </div>
            </div>

            {/* Projected Chemistry Values on that Recommended Date */}
            <div className="mt-4 pt-4 border-t border-white/10">
              <div className="flex items-center justify-between mb-2.5">
                <span className="text-xs font-bold uppercase tracking-wider text-blue-200 flex items-center gap-1.5 font-heading">
                  <Target size={14} className="text-cyan-400" />
                  <span>{t("rec_visit_projected_title")}</span>
                </span>
                <span className="text-[11px] text-blue-300/70 font-mono">
                  {data.recommended_visit.day_label} (Día +{data.recommended_visit.day_offset_from_today})
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {/* Predicted Chlorine */}
                <div className="glass-card rounded-xl p-3.5 bg-blue-950/60 border border-blue-800/40">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-[11px] text-blue-300/80">{t("rec_visit_projected_cl")}</span>
                    <span className="text-[10px] text-cyan-300 font-mono font-bold">1.0–1.5 mg/L</span>
                  </div>
                  <div className="flex items-baseline gap-1.5">
                    <span
                      className={`text-2xl font-extrabold font-heading ${
                        data.recommended_visit.predicted_cl < 0.5 || data.recommended_visit.predicted_cl > 2.0
                          ? "text-red-400"
                          : data.recommended_visit.predicted_cl < 1.0 || data.recommended_visit.predicted_cl > 1.5
                          ? "text-amber-400"
                          : "text-emerald-400"
                      }`}
                    >
                      {data.recommended_visit.predicted_cl.toFixed(2)}
                    </span>
                    <span className="text-xs text-blue-300/60">mg/L</span>
                  </div>
                  {data.recommended_visit.uncertainty_band && (
                    <div className="text-[10px] text-blue-300/60 mt-1">
                      {t("rec_visit_confidence")}: {data.recommended_visit.uncertainty_band.cl_low.toFixed(2)} – {data.recommended_visit.uncertainty_band.cl_high.toFixed(2)} mg/L
                    </div>
                  )}
                </div>

                {/* Predicted pH */}
                <div className="glass-card rounded-xl p-3.5 bg-blue-950/60 border border-blue-800/40">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-[11px] text-blue-300/80">{t("rec_visit_projected_ph")}</span>
                    <span className="text-[10px] text-blue-300/60 font-mono">Norm: 7.2–8.0</span>
                  </div>
                  <div className="flex items-baseline gap-1.5">
                    <span
                      className={`text-2xl font-extrabold font-heading ${
                        data.recommended_visit.predicted_ph < 7.2 || data.recommended_visit.predicted_ph > 8.0
                          ? "text-red-400"
                          : "text-emerald-400"
                      }`}
                    >
                      {data.recommended_visit.predicted_ph.toFixed(2)}
                    </span>
                  </div>
                  {data.recommended_visit.uncertainty_band && (
                    <div className="text-[10px] text-blue-300/60 mt-1">
                      {t("rec_visit_confidence")}: {data.recommended_visit.uncertainty_band.ph_low.toFixed(2)} – {data.recommended_visit.uncertainty_band.ph_high.toFixed(2)}
                    </div>
                  )}
                </div>

                {/* Predicted Turbidity */}
                <div className="glass-card rounded-xl p-3.5 bg-blue-950/60 border border-blue-800/40">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-[11px] text-blue-300/80">{t("rec_visit_projected_turb")}</span>
                    <span className="text-[10px] text-blue-300/60 font-mono">Norm: ≤ 5.0</span>
                  </div>
                  <div className="flex items-baseline gap-1.5">
                    <span
                      className={`text-2xl font-extrabold font-heading ${
                        data.recommended_visit.predicted_turb > 5.0 ? "text-red-400" : "text-blue-200"
                      }`}
                    >
                      {data.recommended_visit.predicted_turb.toFixed(1)}
                    </span>
                    <span className="text-xs text-blue-300/60">NTU</span>
                  </div>
                  {data.recommended_visit.uncertainty_band && (
                    <div className="text-[10px] text-blue-300/60 mt-1">
                      {t("rec_visit_confidence")}: {data.recommended_visit.uncertainty_band.turb_low.toFixed(1)} – {data.recommended_visit.uncertainty_band.turb_high.toFixed(1)} NTU
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Today & Tomorrow Forecast Cards */}
        {(() => {
          const todayDay =
            data.forecast.find((d) => d.is_today) ??
            data.forecast.find((d) => d.day_offset_from_today === 0) ??
            (data.forecast.length > 0 ? data.forecast[data.forecast.length - 1] : undefined);
          const tomorrowDay =
            data.forecast.find((d) => d.is_tomorrow) ??
            data.forecast.find((d) => d.day_offset_from_today === 1);
          const futureDays = data.forecast.filter((d) => d.day_offset_from_today > 1);

          return (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                {todayDay && (
                  <ForecastTile day={todayDay} label={t("detail_today_forecast")} isPrimary />
                )}
                {tomorrowDay && (
                  <ForecastTile day={tomorrowDay} label={t("detail_tomorrow_forecast")} />
                )}
              </div>

              {futureDays.length > 0 && (
                <div className="glass-panel rounded-2xl p-5 mb-6">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-sm font-bold text-white font-heading">
                      {t("detail_chart_title")} (+{futureDays.length} {t("detail_chart_days")})
                    </h4>
                    <span className="text-[11px] text-amber-400 inline-flex items-center gap-1">
                      <AlertTriangle size={13} />
                      <span>{t("rec_visit_confidence")} (horizontes extendidos)</span>
                    </span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                    {futureDays.map((d) => (
                      <ForecastTile key={d.date} day={d} label={`Día +${d.day_offset_from_today}`} />
                    ))}
                  </div>
                </div>
              )}
            </>
          );
        })()}
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
                  className={`px-3 py-1 text-xs rounded-lg font-medium transition cursor-pointer ${
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
        <span className={`flex items-center gap-1 ${isBreach ? "text-red-400 font-medium" : "text-emerald-400 font-medium"}`}>
          {isBreach ? <AlertTriangle size={13} /> : <CheckCircle2 size={13} />}
          <span>{isBreach ? t("detail_out_range") : t("detail_in_range")}</span>
        </span>
        <span className="text-blue-300/60 text-[11px]">{t("detail_day_offset")} +{day.day_offset_from_today ?? 0}</span>
      </div>
    </div>
  );
}
