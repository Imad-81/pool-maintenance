import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, RefreshCw, CloudSun, X } from "lucide-react";
import { api } from "../api";

export default function AdminPage() {
  const queryClient = useQueryClient();
  const [msg, setMsg] = useState<{ type: "success" | "error" | "info"; text: string } | null>(null);

  // Queries
  const { data: runs, isLoading: runsLoading } = useQuery({
    queryKey: ["admin-runs"],
    queryFn: () => api.admin.runs(),
  });

  const { data: wxStatus } = useQuery({
    queryKey: ["admin-wx-status"],
    queryFn: () => api.admin.weatherStatus(),
  });

  const { data: ingestLogs, isLoading: logsLoading } = useQuery({
    queryKey: ["admin-ingest-logs"],
    queryFn: () => api.admin.ingestLog(),
  });

  // Mutations
  const retrainMut = useMutation({
    mutationFn: () => api.admin.retrain(),
    onSuccess: (d) => {
      setMsg({
        type: "success",
        text: `Retrain completed: ${d.result?.run_id ? String(d.result.run_id) : "OK"} (Status: ${d.result?.status || d.status})`,
      });
      queryClient.invalidateQueries({ queryKey: ["admin-runs"] });
    },
    onError: (err: Error) => setMsg({ type: "error", text: `Retraining failed: ${err.message}` }),
  });

  const weatherMut = useMutation({
    mutationFn: () => api.admin.weather(),
    onSuccess: (d) => {
      setMsg({
        type: "success",
        text: `Weather synchronized: ${d.rows_upserted} days updated in database.`,
      });
      queryClient.invalidateQueries({ queryKey: ["admin-wx-status"] });
    },
    onError: (err: Error) => setMsg({ type: "error", text: `Weather sync failed: ${err.message}` }),
  });

  const activeRun = runs?.find((r) => r.is_active);

  return (
    <div className="min-h-screen bg-[#0f1117] text-[#e8eaed] p-8">
      {/* Header */}
      <div className="flex justify-between items-center mb-8 border-b border-[#2d3141] pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white font-heading">
            System Administration & MLOps Console
          </h1>
          <p className="text-xs text-[#9aa0a6] mt-1">
            Manage model lifecycle, promotion gates, weather pipelines, and raw ingest audit logs
          </p>
        </div>
      </div>

      {/* Message alert */}
      {msg && (
        <div
          className={`mb-6 p-4 rounded-xl text-sm border flex items-center justify-between ${
            msg.type === "success"
              ? "bg-green-500/10 border-green-500/30 text-green-400"
              : msg.type === "error"
              ? "bg-red-500/10 border-red-500/30 text-red-400"
              : "bg-[#4f8ff7]/10 border-[#4f8ff7]/30 text-[#4f8ff7]"
          }`}
        >
          <span>{msg.text}</span>
          <button onClick={() => setMsg(null)} className="text-xs opacity-70 hover:opacity-100 cursor-pointer">
            <X size={14} />
          </button>
        </div>
      )}

      {/* System Status Banner */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        <div className="bg-[#1a1d27] border border-[#2d3141] rounded-xl p-4">
          <div className="text-xs font-semibold text-[#6b7280] uppercase tracking-wider mb-1">Active Model Run</div>
          <div className="text-sm font-mono font-bold text-green-400">{activeRun ? activeRun.run_id : "None"}</div>
          <div className="text-[10px] text-[#9aa0a6] mt-1">
            {activeRun?.promoted_at ? `Promoted ${new Date(activeRun.promoted_at).toLocaleDateString()}` : "Active"}
          </div>
        </div>

        <div className="bg-[#1a1d27] border border-[#2d3141] rounded-xl p-4">
          <div className="text-xs font-semibold text-[#6b7280] uppercase tracking-wider mb-1">Weather Sync Status</div>
          <div className="text-sm font-bold text-[#4f8ff7]">
            {wxStatus?.latest_weather_date ? `Synced through ${wxStatus.latest_weather_date}` : "Loading..."}
          </div>
          <div className="text-[10px] text-[#9aa0a6] mt-1">Open-Meteo (Alicante station)</div>
        </div>

        <div className="bg-[#1a1d27] border border-[#2d3141] rounded-xl p-4">
          <div className="text-xs font-semibold text-[#6b7280] uppercase tracking-wider mb-1">Database Engine</div>
          <div className="text-sm font-bold text-[#e8eaed]">PostgreSQL 16</div>
          <div className="text-[10px] text-green-400 mt-1 flex items-center gap-1">
            <Check size={12} />
            <span>Prisma ORM Connected</span>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="grid grid-cols-2 gap-4 mb-8">
        <button
          onClick={() => retrainMut.mutate()}
          disabled={retrainMut.isPending}
          className="p-5 bg-[#1a1d27] border border-[#2d3141] hover:border-[#4f8ff7] rounded-xl text-left transition disabled:opacity-50 group cursor-pointer"
        >
          <div className="flex items-center justify-between mb-1">
            <span className="text-base font-semibold text-[#e8eaed] group-hover:text-[#4f8ff7] transition flex items-center gap-2">
              <RefreshCw size={16} className={retrainMut.isPending ? "animate-spin" : ""} />
              <span>Retrain & Evaluate Models</span>
            </span>
            {retrainMut.isPending && (
              <span className="text-xs text-[#4f8ff7] animate-pulse">Running training pipeline...</span>
            )}
          </div>
          <p className="text-xs text-[#6b7280]">
            Spawns full pipeline with promotion gate evaluation against active model run.
          </p>
        </button>

        <button
          onClick={() => weatherMut.mutate()}
          disabled={weatherMut.isPending}
          className="p-5 bg-[#1a1d27] border border-[#2d3141] hover:border-[#4f8ff7] rounded-xl text-left transition disabled:opacity-50 group cursor-pointer"
        >
          <div className="flex items-center justify-between mb-1">
            <span className="text-base font-semibold text-[#e8eaed] group-hover:text-[#4f8ff7] transition flex items-center gap-2">
              <CloudSun size={16} />
              <span>Synchronize Weather Cache</span>
            </span>
            {weatherMut.isPending && (
              <span className="text-xs text-[#4f8ff7] animate-pulse">Fetching Open-Meteo API...</span>
            )}
          </div>
          <p className="text-xs text-[#6b7280]">
            Fetches yesterday's historical archive + 7-day UV/solar/temp forecast for Alicante.
          </p>
        </button>
      </div>

      {/* Model Runs Table */}
      <div className="bg-[#1a1d27] border border-[#2d3141] rounded-xl overflow-hidden shadow-xl mb-8">
        <div className="p-4 border-b border-[#2d3141] bg-[#21242f] flex justify-between items-center">
          <h3 className="text-xs font-semibold text-[#9aa0a6] uppercase tracking-wider">Model Registry</h3>
          <span className="text-xs text-[#6b7280]">{runs?.length || 0} runs registered</span>
        </div>
        {runsLoading ? (
          <div className="p-8 text-center text-[#6b7280]">Loading model runs...</div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-[#141820] text-[#6b7280] text-xs font-semibold uppercase tracking-wider">
              <tr>
                <th className="text-left px-4 py-3">Run ID</th>
                <th className="text-left px-4 py-3">Created</th>
                <th className="text-left px-4 py-3">Status</th>
                <th className="text-left px-4 py-3">Cl MAE</th>
                <th className="text-left px-4 py-3">pH MAE</th>
                <th className="text-left px-4 py-3">Turb MAE</th>
                <th className="text-left px-4 py-3">Promotion Details</th>
              </tr>
            </thead>
            <tbody>
              {(runs || []).map((r) => (
                <tr key={r.run_id} className="border-b border-[#2d3141]/50 hover:bg-[#2a2e3b] transition">
                  <td className="px-4 py-3 font-mono text-xs text-[#e8eaed]">{r.run_id}</td>
                  <td className="px-4 py-3 text-xs text-[#9aa0a6]">{r.created_at?.slice(0, 19).replace("T", " ")}</td>
                  <td className="px-4 py-3">
                    {r.is_active ? (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-green-500/10 text-green-400 border border-green-500/30">
                        <span className="w-1.5 h-1.5 rounded-full bg-green-400" /> Active
                      </span>
                    ) : (
                      <span className="text-xs text-[#6b7280]">Archived</span>
                    )}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-[#e8eaed]">
                    {r.metrics?.chlorine_next?.mae?.toFixed(4) ?? "—"}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-[#e8eaed]">
                    {r.metrics?.ph_next?.mae?.toFixed(4) ?? "—"}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-[#e8eaed]">
                    {r.metrics?.turbidity_next?.mae?.toFixed(4) ?? "—"}
                  </td>
                  <td className="px-4 py-3 text-xs text-[#9aa0a6]">{r.promote_reason || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Ingestion Audit Log Table */}
      <div className="bg-[#1a1d27] border border-[#2d3141] rounded-xl overflow-hidden shadow-xl">
        <div className="p-4 border-b border-[#2d3141] bg-[#21242f] flex justify-between items-center">
          <h3 className="text-xs font-semibold text-[#9aa0a6] uppercase tracking-wider">Data Ingestion Audit Log</h3>
          <span className="text-xs text-[#6b7280]">{ingestLogs?.length || 0} batches recorded</span>
        </div>
        {logsLoading ? (
          <div className="p-8 text-center text-[#6b7280]">Loading audit logs...</div>
        ) : (ingestLogs || []).length === 0 ? (
          <div className="p-8 text-center text-[#6b7280]">No ingestion events recorded yet.</div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-[#141820] text-[#6b7280] text-xs font-semibold uppercase tracking-wider">
              <tr>
                <th className="text-left px-4 py-3">Timestamp</th>
                <th className="text-left px-4 py-3">Source</th>
                <th className="text-left px-4 py-3">Filename / Detail</th>
                <th className="text-left px-4 py-3">Pools</th>
                <th className="text-left px-4 py-3">Rows</th>
                <th className="text-left px-4 py-3">Skipped</th>
              </tr>
            </thead>
            <tbody>
              {(ingestLogs || []).map((l) => (
                <tr key={l.id} className="border-b border-[#2d3141]/50 hover:bg-[#2a2e3b] transition">
                  <td className="px-4 py-3 text-xs text-[#9aa0a6]">{l.created_at?.slice(0, 19).replace("T", " ")}</td>
                  <td className="px-4 py-3">
                    <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-[#2a2e3b] text-[#e8eaed]">
                      {l.source}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs font-mono text-[#9aa0a6]">{l.filename || "REST API payload"}</td>
                  <td className="px-4 py-3 text-xs text-[#e8eaed]">{l.pool_count}</td>
                  <td className="px-4 py-3 text-xs text-green-400 font-semibold">{l.row_count}</td>
                  <td className="px-4 py-3 text-xs text-amber-400">{l.skipped_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
