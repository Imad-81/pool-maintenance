import { useState, useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import type { ForecastDay, PoolDetail } from "../types";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceArea } from "recharts";

export default function PoolDetailPage() {
  const { poolId } = useParams<{ poolId: string }>();
  const navigate = useNavigate();
  const [horizon, setHorizon] = useState(2);

  const { data, isLoading, error } = useQuery({
    queryKey: ["pool", poolId, horizon],
    queryFn: () => api.pool(poolId!, horizon),
    enabled: !!poolId,
  });

  if (isLoading) return <div className="p-12 text-center text-[#6b7280]">Loading {poolId}...</div>;
  if (error || !data) return <div className="p-12 text-center text-red-400">Failed to load pool: {error?.message}</div>;

  return (
    <div>
      <button onClick={() => navigate("/")} className="inline-flex items-center gap-2 mb-6 px-4 py-2 bg-[#1a1d27] border border-[#2d3141] rounded-lg text-[#9aa0a6] text-sm hover:text-white hover:border-[#4f8ff7] transition">
        ← Back to Fleet
      </button>

      <div className="flex justify-between items-start mb-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">{data.pool_id}</h2>
          <p className="text-sm text-[#9aa0a6]">{data.community_name || "Unknown community"}</p>
        </div>
        <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-semibold ${urgencyBadge(data).bg} ${urgencyBadge(data).color}`}>
          {urgencyBadge(data).label}
          <span className="text-xs text-[#6b7280] ml-2">RD 742/2013</span>
        </span>
      </div>

      {/* Today & Tomorrow cards */}
      {(() => {
        const todayDay = data.forecast.find((d) => d.is_today);
        const tomorrowDay = data.forecast.find((d) => d.is_tomorrow);
        return (
          <div className="grid grid-cols-2 gap-4 mb-6">
            <ForecastCard day={todayDay} label={formatDateLabel(todayDay?.date, "Today")} highlight />
            <ForecastCard day={tomorrowDay} label={formatDateLabel(tomorrowDay?.date, "Tomorrow")} />
          </div>
        );
      })()}

      {/* Horizon toggle & future days */}
      {data.forecast.filter((d) => !d.is_today && !d.is_tomorrow).length > 0 && (
        <details className="mb-6 bg-[#1a1d27] border border-[#2d3141] rounded-xl overflow-hidden">
          <summary className="p-4 cursor-pointer text-sm font-semibold text-[#9aa0a6] uppercase tracking-wider">
            Extended forecast — {data.forecast.filter((d) => !d.is_today && !d.is_tomorrow).length} more days
            <span className="ml-2 text-xs font-normal text-amber-400">⚠ Higher uncertainty past tomorrow</span>
          </summary>
          <div className="grid grid-cols-3 gap-4 p-4 pt-0">
            {data.forecast.filter((d) => !d.is_today && !d.is_tomorrow).map((d) => (
              <ForecastCard key={d.date} day={d} label={formatDateLabel(d.date, "Forecast")} muted />
            ))}
          </div>
        </details>
      )}

      <div className="grid grid-cols-3 gap-2 mb-6">
        {[2, 3, 5, 7].map((h) => (
          <button
            key={h}
            onClick={() => setHorizon(h)}
            className={`py-1.5 text-xs rounded-lg border transition ${horizon === h ? "border-[#4f8ff7] bg-[#4f8ff7]/10 text-[#4f8ff7]" : "border-[#2d3141] text-[#6b7280] hover:border-[#6b7280]"}`}
          >
            {h} days
          </button>
        ))}
      </div>

      {/* Optimiser */}
      {data.optimiser && (
        <div className="bg-[#1a1d27] border border-[#2d3141] rounded-xl p-5 mb-6">
          <h3 className="text-sm font-semibold text-[#9aa0a6] uppercase tracking-wider mb-4">Dosing Recommendation</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-xs text-[#6b7280] mb-1">Recommended</div>
              <div className="text-xl font-bold text-[#4f8ff7]">{data.optimiser.recommended_dosing.hypochlorite_dosing_pct}% × {data.optimiser.recommended_dosing.hypochlorite_dosing_hours}h</div>
            </div>
            <div>
              <div className="text-xs text-[#6b7280] mb-1">Predicted outcome</div>
              <div className="text-lg">Cl {data.optimiser.predicted_tomorrow.free_chlorine} mg/L · pH {data.optimiser.predicted_tomorrow.ph}</div>
            </div>
            <div>
              <div className="text-xs text-[#6b7280] mb-1">Feasible configs</div>
              <div className="text-lg font-semibold text-green-400">{data.optimiser.feasible_configurations}</div>
            </div>
          </div>
          <div className="mt-3 text-xs text-[#9aa0a6]">{data.optimiser.reasons.join(" · ")}</div>
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 gap-6">
        <ChartCard title="pH" data={data.history as unknown as { reading_date: string; [k: string]: unknown }[]} field="ph" yDomain={[6, 9]} low={7.2} high={8.0} unit="" />
        <ChartCard title="Free Chlorine" data={data.history as unknown as { reading_date: string; [k: string]: unknown }[]} field="free_chlorine" yDomain={[0, 8]} low={0.5} high={5.0} unit=" mg/L" />
        <ChartCard title="Turbidity" data={data.history as unknown as { reading_date: string; [k: string]: unknown }[]} field="turbidity" yDomain={[0, 6]} low={0} high={5.0} unit=" NTU" />
      </div>
    </div>
  );
}

function ForecastCard({ day, label, highlight, muted }: { day?: ForecastDay; label: string; highlight?: boolean; muted?: boolean }) {
  if (!day) return null;
  const isBreach = day.cl_breach || day.ph_breach;
  return (
    <div className={`rounded-xl p-4 border ${highlight ? "border-[#4f8ff7] bg-[#4f8ff7]/5" : muted ? "border-[#2d3141]/50 opacity-70" : "border-[#2d3141] bg-[#1a1d27]"}`}>
      <div className="flex justify-between items-start mb-2">
        <span className="text-xs font-semibold text-[#6b7280] uppercase">{label}</span>
        <span className={`text-xs font-bold ${isBreach ? "text-red-400" : "text-green-400"}`}>{day.status}</span>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <div><div className="text-[10px] text-[#6b7280]">Cl</div><div className={`text-lg font-bold ${valCl(day.predicted_cl)}`}>{day.predicted_cl.toFixed(1)}</div></div>
        <div><div className="text-[10px] text-[#6b7280]">pH</div><div className={`text-lg font-bold ${valPh(day.predicted_ph)}`}>{day.predicted_ph.toFixed(2)}</div></div>
        <div><div className="text-[10px] text-[#6b7280]">Turb</div><div className="text-lg font-bold text-[#9aa0a6]">{day.predicted_turb.toFixed(1)}</div></div>
      </div>
      {day.uncertainty_band && day.day_offset_from_today > 1 && (
        <div className="mt-2 text-[10px] text-amber-400">±{(day.uncertainty_band.cl_high - day.predicted_cl).toFixed(2)} mg/L uncertainty</div>
      )}
    </div>
  );
}

function ChartCard({ title, data, field, yDomain, low, high, unit }: { title: string; data: { reading_date: string; [k: string]: unknown }[]; field: string; yDomain: [number, number]; low: number; high: number; unit: string }) {
  const chartData = useMemo(() => data.map((d) => ({
    ...d,
    date: new Date(d.reading_date).getTime(),
    v: d[field] as number | null,
  })).filter((d) => d.v != null), [data, field]);

  return (
    <div className="bg-[#1a1d27] border border-[#2d3141] rounded-xl p-5">
      <h3 className="text-sm font-semibold text-[#9aa0a6] uppercase tracking-wider mb-3">{title}{unit}</h3>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={chartData}>
          <CartesianGrid stroke="#2d3141" strokeDasharray="4 4" />
          <XAxis dataKey="date" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(t) => new Date(t).toLocaleDateString("es-ES", { month: "short", year: "2-digit" })} stroke="#6b7280" fontSize={10} />
          <YAxis domain={yDomain} stroke="#6b7280" fontSize={10} />
          <Tooltip labelFormatter={(t) => new Date(t as number).toLocaleDateString()} />
          <ReferenceArea y1={low} y2={high} fill="#22c55e" fillOpacity={0.06} />
          <Line type="monotone" dataKey="v" stroke="#4f8ff7" dot={false} strokeWidth={1.5} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function valCl(v: number) { if (v < 0.5 || v > 5.0) return "text-red-400"; if (v < 1.0) return "text-amber-400"; return "text-green-400"; }
function valPh(v: number) { if (v < 7.2 || v > 8.0) return "text-red-400"; if (v < 7.3 || v > 7.9) return "text-amber-400"; return "text-green-400"; }

function urgencyBadge(d: PoolDetail) {
  const isBreach = d.forecast.some((f) => f.cl_breach || f.ph_breach);
  if (isBreach) return { label: "🚨 URGENT", bg: "bg-red-500/10", color: "text-red-400" };
  const advised = d.forecast.some((f) => f.urgency === "Advised");
  if (advised) return { label: "⚠ Advised", bg: "bg-amber-400/10", color: "text-amber-400" };
  return { label: "✅ Routine", bg: "bg-green-400/10", color: "text-green-400" };
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
