import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api";
import type { FleetItem } from "../types";

const URGENCY_COLORS: Record<string, string> = {
  Immediate: "text-red-500", URGENT: "text-red-500",
  Advised: "text-amber-400", Soon: "text-amber-400",
  Monitor: "text-amber-400", Routine: "text-green-400", Extended: "text-blue-400",
};
const URGENCY_BG: Record<string, string> = {
  Immediate: "bg-red-500/10", URGENT: "bg-red-500/10",
  Advised: "bg-amber-400/10", Soon: "bg-amber-400/10",
  Monitor: "bg-amber-400/10", Routine: "bg-green-400/10", Extended: "bg-blue-400/10",
};

function valClass(v: number | null, low: number, high: number) {
  if (v == null) return "text-[#9aa0a6]";
  if (v < low || v > high) return "text-red-400";
  return "text-green-400";
}

export default function FleetPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const pageSize = 50;

  const date = searchParams.get("date") || undefined;
  const urgency = searchParams.get("urgency") || undefined;

  const { data, isLoading, error } = useQuery({
    queryKey: ["fleet", date, urgency, page],
    queryFn: () => api.fleet({ date, urgency, page, page_size: pageSize }),
  });

  const items = data?.items ?? [];
  const total = data?.total ?? 0;

  const filtered = useMemo(() => {
    if (!search) return items;
    const q = search.toLowerCase();
    return items.filter((p) => p.pool_id.toLowerCase().includes(q) || (p.community_name || "").toLowerCase().includes(q));
  }, [items, search]);

  const stats = useMemo(() => {
    const counts: Record<string, number> = { Immediate: 0, Advised: 0, Monitor: 0, Routine: 0, Extended: 0 };
    items.forEach((p) => { counts[p.urgency] = (counts[p.urgency] || 0) + 1; });
    return counts;
  }, [items]);

  return (
    <div>
      <div className="grid grid-cols-4 gap-4 mb-6">
        <StatCard label="Immediate Action" value={stats.Immediate} color="text-red-400" />
        <StatCard label="Needs Attention" value={stats.Advised + (stats.Monitor || 0)} color="text-amber-400" />
        <StatCard label="Routine" value={stats.Routine} color="text-green-400" />
        <StatCard label="Extended" value={stats.Extended} color="text-blue-400" />
      </div>

      <div className="flex items-center gap-4 mb-4">
        <input
          type="text"
          placeholder="Search pools by name or community..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="flex-1 max-w-md bg-[#1a1d27] border border-[#2d3141] rounded-lg text-[#e8eaed] px-4 py-2.5 text-sm outline-none focus:border-[#4f8ff7] placeholder:text-[#6b7280]"
        />
        {urgency && (
          <button onClick={() => navigate("/")} className="px-3 py-2 text-sm text-[#4f8ff7] border border-[#4f8ff7]/30 rounded-lg hover:bg-[#4f8ff7]/10">
            Clear filter
          </button>
        )}
      </div>

      <div className="bg-[#1a1d27] border border-[#2d3141] rounded-xl overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-[#6b7280]">Loading fleet data...</div>
        ) : error ? (
          <div className="p-12 text-center text-red-400">Failed to load fleet: {error.message}</div>
        ) : filtered.length === 0 ? (
          <div className="p-12 text-center text-[#6b7280]">No pools match the current filters.</div>
        ) : (
          <table className="w-full">
            <thead className="bg-[#21242f] text-[#6b7280] text-xs font-semibold uppercase tracking-wider">
              <tr>
                <th className="text-left px-4 py-3">Pool</th>
                <th className="text-left px-4 py-3">Last Reading</th>
                <th className="text-left px-4 py-3">pH</th>
                <th className="text-left px-4 py-3">Cl mg/L</th>
                <th className="text-left px-4 py-3">Turb NTU</th>
                <th className="text-left px-4 py-3">Status</th>
                <th className="text-left px-4 py-3">Today</th>
                <th className="text-left px-4 py-3">Tomorrow</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((p) => (
                <FleetRow key={p.pool_id} pool={p} onClick={() => navigate(`/pool/${encodeURIComponent(p.pool_id)}`)} />
              ))}
            </tbody>
          </table>
        )}
        <div className="flex justify-between items-center px-4 py-3 text-sm text-[#6b7280] border-t border-[#2d3141]">
          <span>{total > 0 ? `Showing ${page * pageSize + 1}–${Math.min((page + 1) * pageSize, total)} of ${total}` : "No pools"}</span>
          <div className="flex gap-2">
            <button disabled={page === 0} onClick={() => setPage(page - 1)} className="px-3 py-1.5 rounded border border-[#2d3141] text-sm disabled:opacity-30 hover:border-[#4f8ff7]">
              Prev
            </button>
            <button disabled={(page + 1) * pageSize >= total} onClick={() => setPage(page + 1)} className="px-3 py-1.5 rounded border border-[#2d3141] text-sm disabled:opacity-30 hover:border-[#4f8ff7]">
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="bg-[#1a1d27] border border-[#2d3141] rounded-xl p-5 transition hover:border-[#4f8ff7]">
      <div className="text-xs text-[#6b7280] uppercase tracking-wider mb-2">{label}</div>
      <div className={`text-3xl font-bold tracking-tight ${color}`}>{value}</div>
    </div>
  );
}

function FleetRow({ pool, onClick }: { pool: FleetItem; onClick: () => void }) {
  const today = pool.today_forecast;
  const tomorrow = pool.tomorrow_forecast;
  return (
    <tr onClick={onClick} className="border-b border-[#2d3141]/50 cursor-pointer hover:bg-[#2a2e3b] transition text-sm">
      <td className="px-4 py-3"><div className="font-medium">{pool.pool_id}</div><div className="text-xs text-[#6b7280]">{pool.community_name}</div></td>
      <td className="px-4 py-3 text-[#9aa0a6]">{pool.last_reading_date?.slice(0, 10)}</td>
      <td className={`px-4 py-3 ${valClass(pool.ph, 7.2, 8.0)}`}>{pool.ph?.toFixed(1) ?? "—"}</td>
      <td className={`px-4 py-3 ${valClass(pool.free_chlorine, 0.5, 5.0)}`}>{pool.free_chlorine?.toFixed(2) ?? "—"}</td>
      <td className={`px-4 py-3 ${valClass(pool.turbidity, 0, 5.0)}`}>{pool.turbidity?.toFixed(1) ?? "—"}</td>
      <td className="px-4 py-3">
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${URGENCY_BG[pool.urgency]} ${URGENCY_COLORS[pool.urgency]}`}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: "currentColor" }} />
          {pool.urgency}
        </span>
      </td>
      <td className="px-4 py-3 text-xs">
        {today ? <span className={today.cl_breach || today.ph_breach ? "text-red-400" : "text-green-400"}>Cl {today.predicted_cl.toFixed(1)}</span> : "—"}
      </td>
      <td className="px-4 py-3 text-xs">
        {tomorrow ? <span className={tomorrow.cl_breach || tomorrow.ph_breach ? "text-red-400" : "text-green-400"}>Cl {tomorrow.predicted_cl.toFixed(1)}</span> : "—"}
      </td>
    </tr>
  );
}
