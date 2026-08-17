import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api";
import type { FleetItem, ForecastDay } from "../types";
import IngestModal from "../components/IngestModal";

const URGENCY_COLORS: Record<string, string> = {
  Immediate: "text-red-400", URGENT: "text-red-400",
  Advised: "text-amber-400", Soon: "text-amber-400",
  Monitor: "text-amber-400", Routine: "text-green-400", Extended: "text-blue-400",
};
const URGENCY_BG: Record<string, string> = {
  Immediate: "bg-red-500/10 border-red-500/30", URGENT: "bg-red-500/10 border-red-500/30",
  Advised: "bg-amber-400/10 border-amber-400/30", Soon: "bg-amber-400/10 border-amber-400/30",
  Monitor: "bg-amber-400/10 border-amber-400/30", Routine: "bg-green-400/10 border-green-400/30", Extended: "bg-blue-400/10 border-blue-400/30",
};

function valClass(v: number | null, low: number, high: number) {
  if (v == null) return "text-[#9aa0a6]";
  if (v < low || v > high) return "text-red-400";
  return "text-green-400";
}

export default function FleetPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [isIngestOpen, setIsIngestOpen] = useState(false);
  const pageSize = 50;

  const date = searchParams.get("date") || undefined;
  const urgencyFilter = searchParams.get("urgency") || undefined;

  const { data, isLoading, error } = useQuery({
    queryKey: ["fleet", date, urgencyFilter, page],
    queryFn: () => api.fleet({ date, urgency: urgencyFilter, page, page_size: pageSize }),
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

  const sampleToday = items.find((p) => p.today_forecast?.date)?.today_forecast?.date;
  const sampleTomorrow = items.find((p) => p.tomorrow_forecast?.date)?.tomorrow_forecast?.date;

  const todayLabel = formatDateLabel(sampleToday, "Today");
  const tomorrowLabel = formatDateLabel(sampleTomorrow, "Tomorrow");

  const handleUrgencyClick = (urg: string) => {
    if (urgencyFilter === urg) {
      searchParams.delete("urgency");
    } else {
      searchParams.set("urgency", urg);
    }
    setSearchParams(searchParams);
    setPage(0);
  };

  return (
    <div>
      {/* Top Action Bar */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-[#e8eaed]">Fleet Water Quality & Prediction</h2>
          <p className="text-xs text-[#9aa0a6]">Chained physics-ML forecasts for Alicante collective-use pools</p>
        </div>
        <button
          onClick={() => setIsIngestOpen(true)}
          className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-[#4f8ff7] to-[#7c3aed] hover:opacity-90 text-white text-xs font-semibold rounded-xl shadow-lg transition"
        >
          <span>➕</span> Add Reading / Import Data
        </button>
      </div>

      {/* KPI Stats Cards */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <StatCard
          label="Immediate Action"
          value={stats.Immediate}
          color="text-red-400"
          active={urgencyFilter === "Immediate"}
          onClick={() => handleUrgencyClick("Immediate")}
        />
        <StatCard
          label="Needs Attention"
          value={stats.Advised + (stats.Monitor || 0)}
          color="text-amber-400"
          active={urgencyFilter === "Advised"}
          onClick={() => handleUrgencyClick("Advised")}
        />
        <StatCard
          label="Routine"
          value={stats.Routine}
          color="text-green-400"
          active={urgencyFilter === "Routine"}
          onClick={() => handleUrgencyClick("Routine")}
        />
        <StatCard
          label="Extended"
          value={stats.Extended}
          color="text-blue-400"
          active={urgencyFilter === "Extended"}
          onClick={() => handleUrgencyClick("Extended")}
        />
      </div>

      {/* Search & Filter Toolbar */}
      <div className="flex items-center justify-between gap-4 mb-4">
        <div className="flex items-center gap-3 flex-1">
          <input
            type="text"
            placeholder="Search pools by name or community..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="flex-1 max-w-md bg-[#1a1d27] border border-[#2d3141] rounded-lg text-[#e8eaed] px-4 py-2.5 text-sm outline-none focus:border-[#4f8ff7] placeholder:text-[#6b7280]"
          />
          {urgencyFilter && (
            <button
              onClick={() => {
                searchParams.delete("urgency");
                setSearchParams(searchParams);
              }}
              className="px-3 py-2 text-xs font-medium text-[#4f8ff7] border border-[#4f8ff7]/30 rounded-lg hover:bg-[#4f8ff7]/10"
            >
              Filter: {urgencyFilter} ✕
            </button>
          )}
        </div>
      </div>

      {/* Fleet Table */}
      <div className="bg-[#1a1d27] border border-[#2d3141] rounded-xl overflow-hidden shadow-xl">
        {isLoading ? (
          <div className="p-12 text-center text-[#6b7280]">
            <div className="inline-block w-8 h-8 border-2 border-[#4f8ff7] border-t-transparent rounded-full animate-spin mb-3"></div>
            <div>Loading fleet forecasts...</div>
          </div>
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
                <th className="text-left px-4 py-3">{todayLabel}</th>
                <th className="text-left px-4 py-3">{tomorrowLabel}</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((p) => (
                <FleetRow key={p.pool_id} pool={p} onClick={() => navigate(`/pool/${encodeURIComponent(p.pool_id)}`)} />
              ))}
            </tbody>
          </table>
        )}

        {/* Pagination */}
        <div className="flex justify-between items-center px-4 py-3 text-sm text-[#6b7280] border-t border-[#2d3141] bg-[#141820]">
          <span>{total > 0 ? `Showing ${page * pageSize + 1}–${Math.min((page + 1) * pageSize, total)} of ${total}` : "No pools"}</span>
          <div className="flex gap-2">
            <button
              disabled={page === 0}
              onClick={() => setPage(page - 1)}
              className="px-3 py-1.5 rounded border border-[#2d3141] text-xs disabled:opacity-30 hover:border-[#4f8ff7] text-[#9aa0a6]"
            >
              Prev
            </button>
            <button
              disabled={(page + 1) * pageSize >= total}
              onClick={() => setPage(page + 1)}
              className="px-3 py-1.5 rounded border border-[#2d3141] text-xs disabled:opacity-30 hover:border-[#4f8ff7] text-[#9aa0a6]"
            >
              Next
            </button>
          </div>
        </div>
      </div>

      {/* Data Ingest Studio Modal */}
      <IngestModal isOpen={isIngestOpen} onClose={() => setIsIngestOpen(false)} />
    </div>
  );
}

function formatDateLabel(dateStr?: string, prefix: string = ""): string {
  if (!dateStr) return prefix;
  const parts = dateStr.slice(0, 10).split("-").map(Number);
  if (parts.length < 3) return prefix ? `${prefix} (${dateStr})` : dateStr;
  const [y, m, d] = parts;
  const dateObj = new Date(y, m - 1, d);
  const formatted = dateObj.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  return prefix ? `${prefix} (${formatted})` : formatted;
}

function StatCard({
  label,
  value,
  color,
  active,
  onClick,
}: {
  label: string;
  value: number;
  color: string;
  active?: boolean;
  onClick?: () => void;
}) {
  return (
    <div
      onClick={onClick}
      className={`bg-[#1a1d27] border rounded-xl p-5 transition cursor-pointer ${
        active ? "border-[#4f8ff7] ring-1 ring-[#4f8ff7]" : "border-[#2d3141] hover:border-[#4f8ff7]"
      }`}
    >
      <div className="text-xs text-[#6b7280] uppercase tracking-wider mb-2">{label}</div>
      <div className={`text-3xl font-bold tracking-tight ${color}`}>{value}</div>
    </div>
  );
}

function ForecastValues({ day }: { day: ForecastDay | null }) {
  if (!day) return <span className="text-[#6b7280]">—</span>;
  return (
    <div className="flex flex-col gap-0.5 text-xs font-mono">
      <div className="flex items-center gap-1.5">
        <span className="text-[#6b7280] text-[10px] uppercase w-7 font-sans">Cl</span>
        <span className={`font-semibold ${valClass(day.predicted_cl, 0.5, 5.0)}`}>
          {day.predicted_cl.toFixed(2)}
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-[#6b7280] text-[10px] uppercase w-7 font-sans">pH</span>
        <span className={`font-semibold ${valClass(day.predicted_ph, 7.2, 8.0)}`}>
          {day.predicted_ph.toFixed(1)}
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-[#6b7280] text-[10px] uppercase w-7 font-sans">Turb</span>
        <span className={`font-semibold ${valClass(day.predicted_turb, 0, 5.0)}`}>
          {day.predicted_turb.toFixed(1)}
        </span>
      </div>
    </div>
  );
}

function FleetRow({ pool, onClick }: { pool: FleetItem; onClick: () => void }) {
  const today = pool.today_forecast;
  const tomorrow = pool.tomorrow_forecast;
  return (
    <tr onClick={onClick} className="border-b border-[#2d3141]/50 cursor-pointer hover:bg-[#2a2e3b] transition text-sm">
      <td className="px-4 py-3">
        <div className="font-medium text-[#e8eaed]">{pool.pool_id}</div>
        <div className="text-xs text-[#6b7280]">{pool.community_name}</div>
      </td>
      <td className="px-4 py-3 text-[#9aa0a6]">{pool.last_reading_date?.slice(0, 10)}</td>
      <td className={`px-4 py-3 ${valClass(pool.ph, 7.2, 8.0)}`}>{pool.ph?.toFixed(1) ?? "—"}</td>
      <td className={`px-4 py-3 ${valClass(pool.free_chlorine, 0.5, 5.0)}`}>{pool.free_chlorine?.toFixed(2) ?? "—"}</td>
      <td className={`px-4 py-3 ${valClass(pool.turbidity, 0, 5.0)}`}>{pool.turbidity?.toFixed(1) ?? "—"}</td>
      <td className="px-4 py-3">
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${URGENCY_BG[pool.urgency]} ${URGENCY_COLORS[pool.urgency]}`}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: "currentColor" }} />
          {pool.urgency}
        </span>
      </td>
      <td className="px-4 py-3">
        <ForecastValues day={today} />
      </td>
      <td className="px-4 py-3">
        <ForecastValues day={tomorrow} />
      </td>
    </tr>
  );
}
