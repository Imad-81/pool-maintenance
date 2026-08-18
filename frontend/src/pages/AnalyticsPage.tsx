import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, CloudSun, RefreshCw, X } from "lucide-react";
import { api } from "../api";
import IberHeader from "../components/IberHeader";
import { AnalyticsFlaskIcon } from "../components/Icons";
import { useI18n } from "../i18n";

export default function AnalyticsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { t } = useI18n();
  const [msg, setMsg] = useState<{ type: "success" | "error" | "info"; text: string } | null>(null);

  const { data: runs, isLoading: runsLoading } = useQuery({
    queryKey: ["admin-runs"],
    queryFn: () => api.admin.runs(),
  });

  const { data: wxStatus } = useQuery({
    queryKey: ["admin-wx-status"],
    queryFn: () => api.admin.weatherStatus(),
  });

  const { data: summaryData } = useQuery({
    queryKey: ["fleet", "summary-kpi"],
    queryFn: () => api.fleetSummary(),
  });

  const retrainMut = useMutation({
    mutationFn: () => api.admin.retrain(),
    onSuccess: (d) => {
      setMsg({
        type: "success",
        text: `Retrain OK: Run ${d.result?.run_id || "New"} (${d.result?.status || d.status})`,
      });
      queryClient.invalidateQueries({ queryKey: ["admin-runs"] });
      queryClient.invalidateQueries({ queryKey: ["fleet"] });
      queryClient.invalidateQueries({ queryKey: ["fleet-summary"] });
    },
    onError: (e: Error) => setMsg({ type: "error", text: `Retrain Error: ${e.message}` }),
  });

  const weatherMut = useMutation({
    mutationFn: () => api.admin.weather(),
    onSuccess: (d) => {
      setMsg({
        type: "success",
        text: `Weather Sync OK: ${d.rows_upserted} records updated.`,
      });
      queryClient.invalidateQueries({ queryKey: ["admin-wx-status"] });
    },
    onError: (e: Error) => setMsg({ type: "error", text: `Weather Error: ${e.message}` }),
  });

  const activeRun = runs?.find((r) => r.is_active);
  const totalCount = summaryData?.total || 0;
  const complianceRate = summaryData?.compliance_rate ?? 100;
  const compliantCount = Math.round((complianceRate / 100) * totalCount);


  return (
    <div className="min-h-screen bg-caustic text-white flex flex-col">
      <IberHeader subtitle={t("analytics_subtitle")} />

      <main className="flex-1 max-w-[1400px] w-full mx-auto px-4 md:px-8 pb-16">
        {/* Navigation & Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate("/")}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl glass-card text-blue-200 hover:text-white text-xs font-semibold cursor-pointer"
            >
              <ArrowLeft size={14} />
              <span>{t("detail_back_to_pools")}</span>
            </button>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white">
                <AnalyticsFlaskIcon size={20} />
              </div>
              <h2 className="text-xl md:text-2xl font-bold tracking-tight font-heading">
                {t("analytics_title")}
              </h2>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => weatherMut.mutate()}
              disabled={weatherMut.isPending}
              className="px-3.5 py-2 rounded-xl glass-card text-xs font-semibold text-cyan-300 hover:text-white disabled:opacity-50 inline-flex items-center gap-1.5 cursor-pointer"
            >
              <CloudSun size={14} />
              <span>{weatherMut.isPending ? t("analytics_syncing") : t("analytics_sync_wx")}</span>
            </button>
            <button
              onClick={() => retrainMut.mutate()}
              disabled={retrainMut.isPending}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 active:scale-95 text-white text-xs font-semibold rounded-xl shadow transition disabled:opacity-50 inline-flex items-center gap-1.5 cursor-pointer"
            >
              <RefreshCw size={14} className={retrainMut.isPending ? "animate-spin" : ""} />
              <span>{retrainMut.isPending ? t("analytics_retraining") : t("analytics_retrain_ai")}</span>
            </button>
          </div>
        </div>

        {msg && (
          <div
            className={`mb-6 p-4 rounded-xl text-sm flex items-center justify-between ${
              msg.type === "success"
                ? "bg-emerald-500/20 border border-emerald-500/40 text-emerald-300"
                : "bg-red-500/20 border border-red-500/40 text-red-300"
            }`}
          >
            <span>{msg.text}</span>
            <button onClick={() => setMsg(null)} className="text-xs cursor-pointer">
              <X size={14} />
            </button>
          </div>
        )}

        {/* Global Analytics Overview Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
          <div className="glass-panel rounded-2xl p-5">
            <span className="text-xs text-blue-300/70 block mb-1">{t("analytics_compliance")}</span>
            <div className="text-3xl font-extrabold text-emerald-400 font-heading">
              {complianceRate}%
            </div>
            <span className="text-[11px] text-blue-200/80 mt-1 block">
              {t("analytics_compliant_pools", { count: compliantCount, total: totalCount })}
            </span>
          </div>


          <div className="glass-panel rounded-2xl p-5">
            <span className="text-xs text-blue-300/70 block mb-1">{t("analytics_model_name")}</span>
            <div className="text-2xl font-bold text-white font-heading truncate">
              {activeRun?.run_id ? activeRun.run_id.slice(0, 16) : "Physics-ML Alicante v2"}
            </div>
            <span className="text-[11px] text-cyan-300 mt-1 block">
              {t("analytics_promoted")}: {activeRun?.promoted_at ? activeRun.promoted_at.slice(0, 10) : "Active"}
            </span>
          </div>

          <div className="glass-panel rounded-2xl p-5">
            <span className="text-xs text-blue-300/70 block mb-1">{t("analytics_wx_data")}</span>
            <div className="text-2xl font-bold text-white font-heading">
              {wxStatus?.latest_weather_date || "Updated"}
            </div>
            <span className="text-[11px] text-blue-200/80 mt-1 block">
              {t("analytics_wx_details")}
            </span>
          </div>
        </div>

        {/* AI Performance Metrics */}
        {activeRun?.metrics && (
          <div className="glass-panel rounded-2xl p-6 mb-6">
            <h3 className="text-lg font-bold text-white font-heading mb-4">
              {t("analytics_model_metrics")}
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(activeRun.metrics).map(([key, val]) => (
                <div key={key} className="glass-card rounded-xl p-4">
                  <span className="text-xs text-blue-300/70 block uppercase font-mono">{key}</span>
                  <div className="mt-2 space-y-1 text-xs">
                    {Object.entries(val).map(([metricKey, metricVal]) => (
                      <div key={metricKey} className="flex justify-between text-blue-100">
                        <span className="text-blue-300/60 font-mono">{metricKey}:</span>
                        <strong className="text-white font-mono">{Number(metricVal).toFixed(4)}</strong>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Model Lifecycle Runs Table */}
        <div className="glass-panel rounded-2xl p-6">
          <h3 className="text-lg font-bold text-white font-heading mb-4">
            {t("analytics_runs_history")}
          </h3>

          {runsLoading ? (
            <p className="text-xs text-blue-300/70">...</p>
          ) : (runs?.length || 0) === 0 ? (
            <p className="text-xs text-blue-300/70">No runs.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-blue-800/40 text-blue-300/70">
                    <th className="pb-3 font-semibold">{t("analytics_run_id")}</th>
                    <th className="pb-3 font-semibold">{t("analytics_date")}</th>
                    <th className="pb-3 font-semibold">{t("analytics_status")}</th>
                    <th className="pb-3 font-semibold">{t("analytics_reason")}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-blue-900/30">
                  {runs?.map((r) => (
                    <tr key={r.run_id} className="hover:bg-blue-600/10">
                      <td className="py-3 font-mono text-cyan-300 font-medium">{r.run_id}</td>
                      <td className="py-3 text-blue-200/80">{r.created_at.slice(0, 19).replace("T", " ")}</td>
                      <td className="py-3">
                        {r.is_active ? (
                          <span className="px-2 py-0.5 rounded-full bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 text-[11px] font-semibold">
                            {t("analytics_active")}
                          </span>
                        ) : (
                          <span className="text-blue-300/50 text-[11px]">{t("analytics_archived")}</span>
                        )}
                      </td>
                      <td className="py-3 text-blue-200/80">{r.promote_reason || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
