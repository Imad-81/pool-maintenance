import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api";
import type { FleetItem } from "../types";
import IberHeader from "../components/IberHeader";
import IngestModal from "../components/IngestModal";
import { PoolLadderIcon } from "../components/Icons";
import { useI18n } from "../i18n";

function valClass(v: number | null, low: number, high: number) {
  if (v == null) return "text-blue-200/60";
  if (v < low || v > high) return "text-red-400 font-bold";
  return "text-emerald-400 font-medium";
}

export default function FleetPage() {
  const { t } = useI18n();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [isIngestOpen, setIsIngestOpen] = useState(false);
  const [msg, setMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const pageSize = 50;

  const date = searchParams.get("date") || undefined;
  const urgencyFilter = searchParams.get("urgency") || undefined;

  const { data, isLoading, error } = useQuery({
    queryKey: ["fleet", date, urgencyFilter, page],
    queryFn: () => api.fleet({ date, urgency: urgencyFilter, page, page_size: pageSize }),
  });

  const runInferenceMut = useMutation({
    mutationFn: () => api.runInference(date),
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["fleet"] });
      setMsg({
        type: "success",
        text: t("pools_inference_success", { count: res.predictions_generated }),
      });
      setTimeout(() => setMsg(null), 5000);
    },
    onError: (err: Error) => {
      setMsg({
        type: "error",
        text: `${t("pools_inference_error")}: ${err.message}`,
      });
    },
  });

  const items = useMemo(() => data?.items ?? [], [data?.items]);
  const total = data?.total ?? 0;

  const filtered = useMemo(() => {
    if (!search) return items;
    const q = search.toLowerCase();
    return items.filter(
      (p) =>
        p.pool_id.toLowerCase().includes(q) ||
        (p.community_name || "").toLowerCase().includes(q)
    );
  }, [items, search]);

  const stats = useMemo(() => {
    const counts: Record<string, number> = { Immediate: 0, Advised: 0, Routine: 0, Extended: 0 };
    items.forEach((p) => {
      if (p.urgency === "Immediate" || p.urgency === "URGENT") counts.Immediate++;
      else if (p.urgency === "Advised" || p.urgency === "Soon" || p.urgency === "Monitor") counts.Advised++;
      else counts.Routine++;
    });
    return counts;
  }, [items]);

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
    <div className="min-h-screen bg-caustic text-white flex flex-col">
      <IberHeader subtitle={t("pools_subtitle")} />

      <main className="flex-1 max-w-[1400px] w-full mx-auto px-4 md:px-8 pb-16">
        {/* Navigation & Action bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate("/")}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl glass-card text-blue-200 hover:text-white text-xs font-semibold"
            >
              {t("backToMenu")}
            </button>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white">
                <PoolLadderIcon size={20} />
              </div>
              <h2 className="text-xl md:text-2xl font-bold tracking-tight font-heading">
                {t("pools_title")} ({total})
              </h2>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => runInferenceMut.mutate()}
              disabled={runInferenceMut.isPending}
              className="px-3.5 py-2 glass-card hover:bg-blue-600/30 active:scale-95 text-cyan-300 hover:text-white text-xs font-semibold rounded-xl shadow transition flex items-center gap-1.5 disabled:opacity-50 cursor-pointer"
              title={t("pools_run_inference")}
            >
              <span className={runInferenceMut.isPending ? "animate-spin" : ""}>⚡</span>
              {runInferenceMut.isPending ? t("pools_running_inference") : t("pools_run_inference")}
            </button>
            <button
              onClick={() => setIsIngestOpen(true)}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 active:scale-95 text-white text-xs font-semibold rounded-xl shadow-lg transition flex items-center gap-1.5 cursor-pointer"
            >
              <span>➕</span> {t("pools_add_reading")}
            </button>
          </div>
        </div>

        {/* Action feedback message */}
        {msg && (
          <div
            className={`mb-6 p-4 rounded-xl text-sm flex items-center justify-between transition-all ${
              msg.type === "success"
                ? "bg-emerald-500/20 border border-emerald-500/40 text-emerald-300"
                : "bg-red-500/20 border border-red-500/40 text-red-300"
            }`}
          >
            <div className="flex items-center gap-2">
              <span>{msg.type === "success" ? "✓" : "⚠️"}</span>
              <span>{msg.text}</span>
            </div>
            <button onClick={() => setMsg(null)} className="text-xs hover:text-white">✕</button>
          </div>
        )}


        {/* Quick Filter KPI Badges */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
          <StatBadge
            label={t("pools_stat_immediate")}
            count={stats.Immediate}
            color="text-red-400"
            active={urgencyFilter === "Immediate"}
            onClick={() => handleUrgencyClick("Immediate")}
          />
          <StatBadge
            label={t("pools_stat_advised")}
            count={stats.Advised}
            color="text-amber-400"
            active={urgencyFilter === "Advised"}
            onClick={() => handleUrgencyClick("Advised")}
          />
          <StatBadge
            label={t("pools_stat_routine")}
            count={stats.Routine}
            color="text-emerald-400"
            active={urgencyFilter === "Routine"}
            onClick={() => handleUrgencyClick("Routine")}
          />
          <StatBadge
            label={t("pools_stat_total")}
            count={items.length}
            color="text-blue-300"
            active={!urgencyFilter}
            onClick={() => {
              searchParams.delete("urgency");
              setSearchParams(searchParams);
            }}
          />
        </div>

        {/* Search Bar */}
        <div className="mb-6 flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <input
              type="text"
              placeholder={t("pools_search_placeholder")}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 rounded-xl glass-card text-sm text-white placeholder-blue-300/50 focus:outline-none focus:ring-2 focus:ring-blue-400"
            />
            <span className="absolute left-3.5 top-3 text-blue-300/60 text-sm">🔍</span>
            {search && (
              <button
                onClick={() => setSearch("")}
                className="absolute right-3 top-2.5 text-xs text-blue-300/70 hover:text-white"
              >
                ✕
              </button>
            )}
          </div>
        </div>

        {/* Pool Fleet Grid */}
        {isLoading ? (
          <div className="glass-panel rounded-2xl p-12 text-center text-blue-200/70">
            <div className="inline-block animate-spin text-2xl mb-3">🌊</div>
            <p>{t("pools_loading")}</p>
          </div>
        ) : error ? (
          <div className="glass-panel rounded-2xl p-8 text-center text-red-400">
            Error: {(error as Error).message}
          </div>
        ) : filtered.length === 0 ? (
          <div className="glass-panel rounded-2xl p-12 text-center text-blue-200/70">
            {t("pools_no_results")}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filtered.map((pool) => (
              <PoolCard key={pool.pool_id} pool={pool} onSelect={() => navigate(`/piscinas/${pool.pool_id}`)} />
            ))}
          </div>
        )}
      </main>

      <IngestModal isOpen={isIngestOpen} onClose={() => setIsIngestOpen(false)} />
    </div>
  );
}

function StatBadge({
  label,
  count,
  color,
  active,
  onClick,
}: {
  label: string;
  count: number;
  color: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`glass-card rounded-xl p-3 text-left transition-all cursor-pointer ${
        active ? "ring-2 ring-blue-400 bg-blue-600/30" : "hover:bg-blue-600/15"
      }`}
    >
      <div className="text-[11px] font-medium text-blue-200/70">{label}</div>
      <div className={`text-xl font-bold font-heading ${color}`}>{count}</div>
    </button>
  );
}

function PoolCard({ pool, onSelect }: { pool: FleetItem; onSelect: () => void }) {
  const { t } = useI18n();

  const getUrgencyBadge = (urg: string) => {
    if (urg === "Immediate" || urg === "URGENT") {
      return { label: t("urg_immediate"), bg: "bg-red-500/20 border-red-500/40", text: "text-red-400" };
    }
    if (urg === "Advised" || urg === "Soon" || urg === "Monitor") {
      return { label: t("urg_advised"), bg: "bg-amber-500/20 border-amber-500/40", text: "text-amber-400" };
    }
    if (urg === "Routine") {
      return { label: t("urg_routine"), bg: "bg-emerald-500/20 border-emerald-500/40", text: "text-emerald-400" };
    }
    return { label: t("urg_extended"), bg: "bg-blue-500/20 border-blue-500/40", text: "text-blue-300" };
  };

  const badgeInfo = getUrgencyBadge(pool.urgency);
  const proba = Math.round((pool.breach_proba || 0) * 100);

  // Today's predicted values (the core predictive maintenance forecast)
  const displayCl = pool.today_forecast?.predicted_cl ?? pool.free_chlorine;
  const displayPh = pool.today_forecast?.predicted_ph ?? pool.ph;
  const displayTurb = pool.today_forecast?.predicted_turb ?? pool.turbidity;
  const forecastDate = pool.today_forecast?.date;

  return (
    <div
      onClick={onSelect}
      className="glass-card rounded-2xl p-5 cursor-pointer hover:border-blue-400/60 hover:-translate-y-1 transition-all flex flex-col justify-between"
    >
      <div>
        {/* Card Header */}
        <div className="flex items-start justify-between gap-2 mb-3">
          <div>
            <h3 className="font-bold text-base text-white group-hover:text-blue-200 font-heading">
              {pool.community_name || pool.pool_id}
            </h3>
            <span className="text-[11px] text-blue-300/60 font-mono">{pool.pool_id}</span>
          </div>
          <span
            className={`px-2.5 py-1 rounded-full text-xs font-semibold border ${badgeInfo.bg} ${badgeInfo.text}`}
          >
            {badgeInfo.label}
          </span>
        </div>

        {/* Today's Predicted Chemistry Status */}
        <div className="grid grid-cols-3 gap-2 bg-blue-950/40 rounded-xl p-2.5 mb-3 border border-blue-800/30 text-center">
          <div>
            <div className="text-[10px] text-blue-300/70">{t("pools_cl")}</div>
            <div className={`text-sm ${valClass(displayCl, 0.5, 2.0)}`}>
              {displayCl != null ? `${displayCl.toFixed(2)}` : "—"}
              <span className="text-[9px] text-blue-300/50 block">mg/L</span>
            </div>
          </div>
          <div>
            <div className="text-[10px] text-blue-300/70">{t("pools_ph")}</div>
            <div className={`text-sm ${valClass(displayPh, 7.2, 8.0)}`}>
              {displayPh != null ? `${displayPh.toFixed(2)}` : "—"}
              <span className="text-[9px] text-blue-300/50 block">7.2 - 8.0</span>
            </div>
          </div>
          <div>
            <div className="text-[10px] text-blue-300/70">{t("pools_turb")}</div>
            <div className={`text-sm ${valClass(displayTurb, 0, 5)}`}>
              {displayTurb != null ? `${displayTurb.toFixed(1)}` : "—"}
              <span className="text-[9px] text-blue-300/50 block">NTU</span>
            </div>
          </div>
        </div>

        {/* Risk Probability Meter */}
        <div className="mb-4">
          <div className="flex justify-between text-[11px] text-blue-200/80 mb-1">
            <span>{t("pools_risk_24_48h")}</span>
            <span className={proba > 50 ? "text-red-400 font-bold" : "text-emerald-400"}>{proba}%</span>
          </div>
          <div className="w-full h-1.5 bg-blue-950 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${
                proba > 60 ? "bg-red-500" : proba > 30 ? "bg-amber-400" : "bg-emerald-400"
              }`}
              style={{ width: `${Math.min(100, Math.max(5, proba))}%` }}
            />
          </div>
        </div>
      </div>

      <div className="pt-2 border-t border-blue-800/30 flex items-center justify-between text-xs text-blue-300">
        <span className="text-[11px] text-blue-300/60">
          {forecastDate
            ? `${t("pools_forecast_date")}: ${forecastDate}`
            : `${t("pools_last_reading")}: ${pool.last_reading_date ? pool.last_reading_date.slice(0, 10) : "N/D"}`}
        </span>
        <span className="text-blue-400 hover:text-white font-medium inline-flex items-center gap-1">
          {t("pools_diagnosis")}
        </span>
      </div>
    </div>
  );
}

