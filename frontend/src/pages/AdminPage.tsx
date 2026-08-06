import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { api } from "../api";

export default function AdminPage() {
  const [msg, setMsg] = useState("");

  const { data: runs, isLoading } = useQuery({ queryKey: ["admin-runs"], queryFn: () => api.admin.runs() });

  const retrainMut = useMutation({
    mutationFn: () => api.admin.retrain(),
    onSuccess: (d) => setMsg(`Retrain ${d.status}: ${d.result?.run_id || ""}`),
    onError: (e: Error) => setMsg(`Error: ${e.message}`),
  });

  const weatherMut = useMutation({
    mutationFn: () => api.admin.weather(),
    onSuccess: (d) => setMsg(`Weather refreshed: ${d.rows_upserted} rows`),
    onError: (e: Error) => setMsg(`Error: ${e.message}`),
  });

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Admin</h2>

      {msg && <div className="mb-4 p-3 rounded-lg bg-[#4f8ff7]/10 border border-[#4f8ff7]/30 text-sm">{msg}</div>}

      <div className="grid grid-cols-2 gap-4 mb-8">
        <button onClick={() => retrainMut.mutate()} disabled={retrainMut.isPending} className="p-4 bg-[#1a1d27] border border-[#2d3141] rounded-xl text-left hover:border-[#4f8ff7] transition disabled:opacity-50">
          <div className="text-lg font-semibold">🔄 Retrain Models</div>
          <div className="text-xs text-[#6b7280] mt-1">Run full pipeline (takes 3–5 minutes)</div>
        </button>
        <button onClick={() => weatherMut.mutate()} disabled={weatherMut.isPending} className="p-4 bg-[#1a1d27] border border-[#2d3141] rounded-xl text-left hover:border-[#4f8ff7] transition disabled:opacity-50">
          <div className="text-lg font-semibold">🌤 Refresh Weather</div>
          <div className="text-xs text-[#6b7280] mt-1">Fetch yesterday + 7-day forecast</div>
        </button>
      </div>

      <div className="bg-[#1a1d27] border border-[#2d3141] rounded-xl overflow-hidden">
        <h3 className="p-4 text-sm font-semibold text-[#9aa0a6] uppercase tracking-wider border-b border-[#2d3141]">Model Runs</h3>
        {isLoading ? (
          <div className="p-6 text-center text-[#6b7280]">Loading...</div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-[#21242f] text-[#6b7280] text-xs font-semibold uppercase tracking-wider">
              <tr>
                <th className="text-left px-4 py-2">Run ID</th>
                <th className="text-left px-4 py-2">Created</th>
                <th className="text-left px-4 py-2">Active</th>
                <th className="text-left px-4 py-2">Cl MAE</th>
                <th className="text-left px-4 py-2">pH MAE</th>
                <th className="text-left px-4 py-2">Turb MAE</th>
              </tr>
            </thead>
            <tbody>
              {(runs || []).map((r) => (
                <tr key={r.run_id} className="border-b border-[#2d3141]/50">
                  <td className="px-4 py-2 font-mono text-xs">{r.run_id}</td>
                  <td className="px-4 py-2 text-[#9aa0a6]">{r.created_at?.slice(0, 19)}</td>
                  <td className="px-4 py-2">{r.is_active ? <span className="text-green-400 font-semibold">✓ Active</span> : <span className="text-[#6b7280]">—</span>}</td>
                  <td className="px-4 py-2">{r.metrics?.chlorine_next?.mae?.toFixed(4) ?? "—"}</td>
                  <td className="px-4 py-2">{r.metrics?.ph_next?.mae?.toFixed(4) ?? "—"}</td>
                  <td className="px-4 py-2">{r.metrics?.turbidity_next?.mae?.toFixed(4) ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
