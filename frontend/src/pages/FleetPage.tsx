import { useState, useMemo, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  Zap,
  Plus,
  Search,
  X,
  Calendar,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  Filter,
} from "lucide-react";
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

  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [page, setPage] = useState(0);
  const [isIngestOpen, setIsIngestOpen] = useState(false);
  const [msg, setMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const pageSize = 24; // 24 items per page for clean 1/2/3-column grid

  const date = searchParams.get("date") || undefined;
  const urgencyFilter = searchParams.get("urgency") || undefined;

  // Debounce search input by 300ms
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchInput.trim());
      setPage(0);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchInput]);

  // 1. Fetch Fleet Summary (Global KPIs)
  const { data: summaryData, isLoading: summaryLoading } = useQuery({
    queryKey: ["fleet-summary", date],
    queryFn: () => api.fleetSummary(date),
  });

  // 2. Fetch Paginated & Filtered Fleet Items
  const { data: fleetData, isLoading: fleetLoading, error } = useQuery({
    queryKey: ["fleet", date, urgencyFilter, debouncedSearch, page, pageSize],
    queryFn: () =>
      api.fleet({
        date,
        q: debouncedSearch || undefined,
        urgency: urgencyFilter,
        page,
        page_size: pageSize,
      }),
  });

  const runInferenceMut = useMutation({
    mutationFn: () => api.runInference(date),
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["fleet"] });
      queryClient.invalidateQueries({ queryKey: ["fleet-summary"] });
      queryClient.invalidateQueries({ queryKey: ["admin-runs"] });
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

  const items = useMemo(() => fleetData?.items ?? [], [fleetData?.items]);
  const totalFiltered = fleetData?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(totalFiltered / pageSize));

  // Global KPI stats from summary endpoint
  const stats = useMemo(() => {
    if (!summaryData) {
      return { Immediate: 0, Advised: 0, Routine: 0, Total: 0 };
    }
    const counts = summaryData.counts || {};
    return {
      Immediate: counts.Immediate ?? 0,
      Advised: counts.Advised ?? 0,
      Routine: (counts.Routine ?? 0) + (counts.Extended ?? 0),
      Total: summaryData.total ?? 0,
    };
  }, [summaryData]);

  const handleUrgencyClick = (urg: string) => {
    setPage(0);
    if (urgencyFilter === urg) {
      searchParams.delete("urgency");
    } else {
      searchParams.set("urgency", urg);
    }
    setSearchParams(searchParams);
  };

  const handleClearAllFilters = () => {
    setPage(0);
    setSearchInput("");
    setDebouncedSearch("");
    searchParams.delete("urgency");
    setSearchParams(searchParams);
  };

  const startItem = totalFiltered > 0 ? page * pageSize + 1 : 0;
  const endItem = Math.min((page + 1) * pageSize, totalFiltered);

  return (
    <div className="min-h-screen bg-caustic text-white flex flex-col">
      <IberHeader subtitle={t("pools_subtitle")} />

      <main className="flex-1 max-w-[1400px] w-full mx-auto px-4 md:px-8 pb-16">
        {/* Navigation & Action bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-md">
              <PoolLadderIcon size={22} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-xl md:text-2xl font-bold tracking-tight font-heading">
                  {t("pools_title")} ({totalFiltered})
                </h2>
                {summaryData?.as_of_date && (
                  <span className="hidden sm:inline-block px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-blue-950/70 border border-blue-800/40 text-blue-300">
                    {summaryData.as_of_date}
                  </span>
                )}
              </div>
              <p className="text-xs text-blue-200/70 mt-0.5">
                {urgencyFilter
                  ? `${t("pools_title")} — ${
                      urgencyFilter === "Immediate"
                        ? t("pools_stat_immediate")
                        : urgencyFilter === "Advised"
                        ? t("pools_stat_advised")
                        : t("pools_stat_routine")
                    }`
                  : `${stats.Total} ${t("pools_stat_total").toLowerCase()}`}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => runInferenceMut.mutate()}
              disabled={runInferenceMut.isPending}
              className="px-3.5 py-2 glass-card hover:bg-blue-600/30 active:scale-95 text-cyan-300 hover:text-white text-xs font-semibold rounded-xl shadow transition flex items-center gap-1.5 disabled:opacity-50 cursor-pointer"
              title={t("pools_run_inference")}
            >
              <Zap size={14} className={runInferenceMut.isPending ? "animate-spin" : ""} />
              <span>{runInferenceMut.isPending ? t("pools_running_inference") : t("pools_run_inference")}</span>
            </button>
            <button
              onClick={() => setIsIngestOpen(true)}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 active:scale-95 text-white text-xs font-semibold rounded-xl shadow-lg transition flex items-center gap-1.5 cursor-pointer"
            >
              <Plus size={14} />
              <span>{t("pools_add_reading")}</span>
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
              {msg.type === "success" ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
              <span>{msg.text}</span>
            </div>
            <button onClick={() => setMsg(null)} className="text-xs hover:text-white cursor-pointer">
              <X size={14} />
            </button>
          </div>
        )}

        {/* Quick Filter KPI Badges (Always showing global fleet summary stats) */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
          <StatBadge
            label={t("pools_stat_immediate")}
            count={stats.Immediate}
            color="text-red-400"
            activeColor="ring-2 ring-red-400 bg-red-600/20 border-red-500/50"
            active={urgencyFilter === "Immediate"}
            isLoading={summaryLoading}
            onClick={() => handleUrgencyClick("Immediate")}
          />
          <StatBadge
            label={t("pools_stat_advised")}
            count={stats.Advised}
            color="text-amber-400"
            activeColor="ring-2 ring-amber-400 bg-amber-600/20 border-amber-500/50"
            active={urgencyFilter === "Advised"}
            isLoading={summaryLoading}
            onClick={() => handleUrgencyClick("Advised")}
          />
          <StatBadge
            label={t("pools_stat_routine")}
            count={stats.Routine}
            color="text-emerald-400"
            activeColor="ring-2 ring-emerald-400 bg-emerald-600/20 border-emerald-500/50"
            active={urgencyFilter === "Routine"}
            isLoading={summaryLoading}
            onClick={() => handleUrgencyClick("Routine")}
          />
          <StatBadge
            label={t("pools_stat_total")}
            count={stats.Total}
            color="text-blue-300"
            activeColor="ring-2 ring-blue-400 bg-blue-600/30 border-blue-500/50"
            active={!urgencyFilter}
            isLoading={summaryLoading}
            onClick={() => {
              setPage(0);
              searchParams.delete("urgency");
              setSearchParams(searchParams);
            }}
          />
        </div>

        {/* Search Bar & Active Filter Bar */}
        <div className="mb-6 flex flex-col sm:flex-row items-center gap-3">
          <div className="relative flex-1 w-full">
            <input
              type="text"
              placeholder={t("pools_search_placeholder")}
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              className="w-full pl-10 pr-10 py-2.5 rounded-xl glass-card text-sm text-white placeholder-blue-300/50 focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
            />
            <span className="absolute left-3.5 top-3 text-blue-300/60">
              <Search size={15} />
            </span>
            {searchInput && (
              <button
                onClick={() => setSearchInput("")}
                className="absolute right-3 top-2.5 text-xs text-blue-300/70 hover:text-white p-1 cursor-pointer"
              >
                <X size={14} />
              </button>
            )}
          </div>

          {(urgencyFilter || debouncedSearch) && (
            <div className="flex items-center gap-2 self-start sm:self-auto">
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-blue-950/70 border border-blue-700/50 text-xs text-blue-200">
                <Filter size={12} className="text-cyan-400" />
                <span>
                  {urgencyFilter
                    ? urgencyFilter === "Immediate"
                      ? t("pools_stat_immediate")
                      : urgencyFilter === "Advised"
                      ? t("pools_stat_advised")
                      : t("pools_stat_routine")
                    : ""}
                  {urgencyFilter && debouncedSearch ? " + " : ""}
                  {debouncedSearch ? `"${debouncedSearch}"` : ""}
                </span>
                <button
                  onClick={handleClearAllFilters}
                  className="ml-1 text-blue-300 hover:text-white cursor-pointer"
                  title="Clear filter"
                >
                  <X size={13} />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Pool Fleet Grid */}
        {fleetLoading ? (
          <div className="glass-panel rounded-2xl p-16 text-center text-blue-200/70">
            <Loader2 size={32} className="animate-spin text-cyan-400 mx-auto mb-3" />
            <p className="text-sm">{t("pools_loading")}</p>
          </div>
        ) : error ? (
          <div className="glass-panel rounded-2xl p-8 text-center text-red-400">
            <AlertTriangle size={28} className="mx-auto mb-2 text-red-400" />
            <p className="font-semibold mb-1">Error loading pools</p>
            <p className="text-xs text-red-300/80">{(error as Error).message}</p>
          </div>
        ) : items.length === 0 ? (
          <div className="glass-panel rounded-2xl p-16 text-center text-blue-200/70">
            <PoolLadderIcon size={40} className="mx-auto mb-3 text-blue-400/40" />
            <p className="text-base font-semibold text-white mb-1">{t("pools_no_results")}</p>
            {(urgencyFilter || debouncedSearch) && (
              <button
                onClick={handleClearAllFilters}
                className="mt-3 inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-blue-600/30 hover:bg-blue-600/50 text-xs font-semibold text-cyan-300 cursor-pointer"
              >
                <span>Clear filters</span>
              </button>
            )}
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {items.map((pool) => (
                <PoolCard key={pool.pool_id} pool={pool} onSelect={() => navigate(`/piscinas/${pool.pool_id}`)} />
              ))}
            </div>

            {/* Modern Pagination Toolbar */}
            {totalFiltered > 0 && (
              <div className="mt-8 pt-4 border-t border-blue-900/40 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-blue-200/80">
                <div>
                  <span>
                    Showing <strong className="text-white font-mono">{startItem}</strong> –{" "}
                    <strong className="text-white font-mono">{endItem}</strong> of{" "}
                    <strong className="text-white font-mono">{totalFiltered}</strong> facilities
                  </span>
                </div>

                {totalPages > 1 && (
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => setPage((p) => Math.max(0, p - 1))}
                      disabled={page === 0}
                      className="px-3 py-1.5 rounded-lg glass-card hover:bg-blue-600/20 disabled:opacity-30 disabled:cursor-not-allowed text-xs font-medium flex items-center gap-1 cursor-pointer transition"
                    >
                      <ChevronLeft size={14} />
                      <span className="hidden sm:inline">Previous</span>
                    </button>

                    <div className="flex items-center gap-1 px-1">
                      {Array.from({ length: totalPages }, (_, idx) => {
                        // Display sliding window around active page if many pages
                        if (
                          totalPages <= 7 ||
                          idx === 0 ||
                          idx === totalPages - 1 ||
                          (idx >= page - 1 && idx <= page + 1)
                        ) {
                          return (
                            <button
                              key={idx}
                              onClick={() => setPage(idx)}
                              className={`w-8 h-8 rounded-lg text-xs font-bold transition cursor-pointer ${
                                page === idx
                                  ? "bg-blue-600 text-white shadow-md"
                                  : "glass-card hover:bg-blue-600/20 text-blue-300 hover:text-white"
                              }`}
                            >
                              {idx + 1}
                            </button>
                          );
                        }
                        if (idx === page - 2 || idx === page + 2) {
                          return (
                            <span key={idx} className="px-1 text-blue-400/50">
                              ...
                            </span>
                          );
                        }
                        return null;
                      })}
                    </div>

                    <button
                      onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                      disabled={page >= totalPages - 1}
                      className="px-3 py-1.5 rounded-lg glass-card hover:bg-blue-600/20 disabled:opacity-30 disabled:cursor-not-allowed text-xs font-medium flex items-center gap-1 cursor-pointer transition"
                    >
                      <span className="hidden sm:inline">Next</span>
                      <ChevronRight size={14} />
                    </button>
                  </div>
                )}
              </div>
            )}
          </>
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
  activeColor,
  active,
  isLoading,
  onClick,
}: {
  label: string;
  count: number;
  color: string;
  activeColor?: string;
  active: boolean;
  isLoading?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`glass-card rounded-xl p-3.5 text-left transition-all cursor-pointer ${
        active
          ? activeColor || "ring-2 ring-blue-400 bg-blue-600/30"
          : "hover:bg-blue-600/15 border border-blue-900/30"
      }`}
    >
      <div className="text-[11px] font-medium text-blue-200/70 truncate">{label}</div>
      <div className={`text-xl md:text-2xl font-bold font-heading mt-0.5 ${color}`}>
        {isLoading ? "—" : count}
      </div>
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

  // Visit recommendation with fallback guarantee
  const recVisit = pool.recommended_visit || {
    date:
      pool.urgency === "Immediate" || pool.urgency === "URGENT"
        ? forecastDate || "Hoy"
        : pool.urgency === "Advised"
        ? "Mañana"
        : "Próxima pauta",
    day_offset_from_today:
      pool.urgency === "Immediate" || pool.urgency === "URGENT" ? 0 : pool.urgency === "Advised" ? 1 : 2,
    predicted_cl: displayCl ?? 1.2,
    urgency: pool.urgency,
  };

  return (
    <div
      onClick={onSelect}
      className="glass-card rounded-2xl p-5 cursor-pointer hover:border-blue-400/60 hover:-translate-y-1 transition-all flex flex-col justify-between"
    >
      <div>
        {/* Card Header */}
        <div className="flex items-start justify-between gap-2 mb-3">
          <div className="overflow-hidden">
            <h3 className="font-bold text-base text-white group-hover:text-blue-200 font-heading truncate">
              {pool.community_name || pool.pool_id}
            </h3>
            <span className="text-[11px] text-blue-300/60 font-mono truncate block">{pool.pool_id}</span>
          </div>
          <span
            className={`px-2.5 py-1 rounded-full text-xs font-semibold border shrink-0 ${badgeInfo.bg} ${badgeInfo.text}`}
          >
            {badgeInfo.label}
          </span>
        </div>

        {/* Today's Predicted Chemistry Status */}
        <div className="grid grid-cols-3 gap-2 bg-blue-950/40 rounded-xl p-2.5 mb-3 border border-blue-800/30 text-center">
          <div>
            <div className="text-[10px] text-blue-300/70">{t("pools_pred_cl_today")}</div>
            <div className={`text-sm ${valClass(displayCl, 0.5, 2.0)}`}>
              {displayCl != null ? `${displayCl.toFixed(2)}` : "—"}
              <span className="text-[9px] text-blue-300/50 block">mg/L</span>
            </div>
          </div>
          <div>
            <div className="text-[10px] text-blue-300/70">{t("pools_pred_ph_today")}</div>
            <div className={`text-sm ${valClass(displayPh, 7.2, 8.0)}`}>
              {displayPh != null ? `${displayPh.toFixed(2)}` : "—"}
              <span className="text-[9px] text-blue-300/50 block">7.2 - 8.0</span>
            </div>
          </div>
          <div>
            <div className="text-[10px] text-blue-300/70">{t("pools_pred_turb_today")}</div>
            <div className={`text-sm ${valClass(displayTurb, 0, 5)}`}>
              {displayTurb != null ? `${displayTurb.toFixed(1)}` : "—"}
              <span className="text-[9px] text-blue-300/50 block">NTU</span>
            </div>
          </div>
        </div>

        {/* Next Recommended Visit Pill */}
        <div
          className={`mb-3.5 px-3 py-2 rounded-xl border flex items-center justify-between text-xs shadow-inner ${
            recVisit.day_offset_from_today === 0
              ? "bg-red-950/50 border-red-500/40 text-red-200"
              : recVisit.day_offset_from_today === 1
              ? "bg-amber-950/50 border-amber-500/40 text-amber-200"
              : "bg-blue-950/70 border-blue-800/50 text-blue-100"
          }`}
        >
          <div className="flex items-center gap-1.5 overflow-hidden">
            <Calendar
              size={13}
              className={recVisit.day_offset_from_today === 0 ? "text-red-400" : "text-cyan-400"}
            />
            <span className="font-semibold truncate">
              {recVisit.day_offset_from_today === 0
                ? t("rec_visit_today")
                : recVisit.day_offset_from_today === 1
                ? t("rec_visit_tomorrow")
                : `${recVisit.date} (${t("rec_visit_in_days", { days: recVisit.day_offset_from_today })})`}
            </span>
          </div>
          <div className="text-[11px] font-mono font-bold text-cyan-300 ml-2 whitespace-nowrap">
            Cl: {recVisit.predicted_cl.toFixed(2)} mg/L
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
        <span className="text-[11px] text-blue-300/60 truncate mr-2">
          {forecastDate
            ? `${t("pools_forecast_date")}: ${forecastDate}`
            : `${t("pools_last_reading")}: ${
                pool.last_reading_date ? pool.last_reading_date.slice(0, 10) : "N/D"
              }`}
        </span>
        <span className="text-blue-400 hover:text-white font-medium inline-flex items-center gap-1 shrink-0">
          <span>{t("pools_diagnosis")}</span>
          <ArrowRight size={13} />
        </span>
      </div>
    </div>
  );
}
