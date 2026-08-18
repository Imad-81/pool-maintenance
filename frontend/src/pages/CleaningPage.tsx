import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, FlaskConical, ClipboardList } from "lucide-react";
import { api } from "../api";
import IberHeader from "../components/IberHeader";
import { SkimmerNetIcon } from "../components/Icons";
import { useI18n } from "../i18n";

export default function CleaningPage() {
  const navigate = useNavigate();
  const { t } = useI18n();

  // Interactive Dosing Simulator State
  const [volume, setVolume] = useState<number>(150);
  const [currentCl, setCurrentCl] = useState<number>(0.8);
  const [targetCl, setTargetCl] = useState<number>(1.5);
  const [pumpCapacityLh, setPumpCapacityLh] = useState<number>(5.0);

  // Checklists
  const [checkedTasks, setCheckedTasks] = useState<Record<string, boolean>>({
    task1: true,
    task2: true,
    task3: false,
    task4: false,
    task5: false,
  });

  const { data: fleetData, isLoading } = useQuery({
    queryKey: ["fleet", "cleaning-list"],
    queryFn: () => api.fleet({ page_size: 50 }),
  });

  const toggleTask = (key: string) => {
    setCheckedTasks((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const deltaCl = Math.max(0, targetCl - currentCl);
  const requiredGramsPureCl = deltaCl * volume;
  const hypochloriteConcentration = 0.15;
  const hypochloriteDensity = 1.2;
  const requiredGramsSolution = requiredGramsPureCl / hypochloriteConcentration;
  const requiredLitersProduct = requiredGramsSolution / (hypochloriteDensity * 1000);
  const pumpHours = Number((requiredLitersProduct / (pumpCapacityLh * 0.7)).toFixed(1));
  const recommendedPumpPct = 70;

  return (
    <div className="min-h-screen bg-caustic text-white flex flex-col">
      <IberHeader subtitle={t("cleaning_subtitle")} />

      <main className="flex-1 max-w-[1400px] w-full mx-auto px-4 md:px-8 pb-16">
        {/* Navigation & Header */}
        <div className="flex items-center gap-3 mb-6">
          <button
            onClick={() => navigate("/")}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl glass-card text-blue-200 hover:text-white text-xs font-semibold cursor-pointer"
          >
            <ArrowLeft size={14} />
            <span>{t("backToMenu")}</span>
          </button>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white">
              <SkimmerNetIcon size={20} />
            </div>
            <h2 className="text-xl md:text-2xl font-bold tracking-tight font-heading">
              {t("cleaning_title")}
            </h2>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Chemical Dosing Optimizer Simulator */}
          <div className="lg:col-span-2 glass-panel rounded-2xl p-6">
            <h3 className="text-lg font-bold text-white font-heading mb-2 flex items-center gap-2">
              <FlaskConical size={18} className="text-cyan-300" />
              <span>{t("cleaning_sim_title")}</span>
            </h3>
            <p className="text-xs text-blue-200/80 mb-6">
              {t("cleaning_sim_desc")}
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
              <div>
                <label className="text-xs text-blue-300/80 block mb-1">{t("cleaning_volume")}</label>
                <input
                  type="number"
                  value={volume}
                  onChange={(e) => setVolume(Number(e.target.value) || 0)}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">{t("cleaning_pump_flow")}</label>
                <input
                  type="number"
                  step="0.5"
                  value={pumpCapacityLh}
                  onChange={(e) => setPumpCapacityLh(Number(e.target.value) || 1)}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">{t("cleaning_current_cl")}</label>
                <input
                  type="number"
                  step="0.1"
                  value={currentCl}
                  onChange={(e) => setCurrentCl(Number(e.target.value) || 0)}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">{t("cleaning_target_cl")}</label>
                <input
                  type="number"
                  step="0.1"
                  value={targetCl}
                  onChange={(e) => setTargetCl(Number(e.target.value) || 0)}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>
            </div>

            {/* Calculated Recommendation Output */}
            <div className="bg-blue-950/60 p-5 rounded-2xl border border-blue-800/40">
              <span className="text-xs text-cyan-300 font-semibold uppercase tracking-wider block mb-3">
                {t("cleaning_rec_title")}
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
                <div className="glass-card rounded-xl p-3">
                  <span className="text-[11px] text-blue-300/70 block">{t("cleaning_pump_pct")}</span>
                  <span className="text-2xl font-bold text-white font-heading">{recommendedPumpPct}%</span>
                </div>
                <div className="glass-card rounded-xl p-3">
                  <span className="text-[11px] text-blue-300/70 block">{t("cleaning_injection_time")}</span>
                  <span className="text-2xl font-bold text-cyan-300 font-heading">{pumpHours} h</span>
                </div>
                <div className="glass-card rounded-xl p-3">
                  <span className="text-[11px] text-blue-300/70 block">{t("cleaning_est_product")}</span>
                  <span className="text-2xl font-bold text-emerald-400 font-heading">
                    {requiredLitersProduct.toFixed(2)} L
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Maintenance Checklist */}
          <div className="glass-panel rounded-2xl p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-lg font-bold text-white font-heading mb-2 flex items-center gap-2">
                <ClipboardList size={18} className="text-cyan-300" />
                <span>{t("cleaning_checklist_title")}</span>
              </h3>
              <p className="text-xs text-blue-200/80 mb-4">
                {t("cleaning_checklist_desc")}
              </p>

              <div className="space-y-3 text-xs">
                {[
                  { id: "task1", label: t("cleaning_task1") },
                  { id: "task2", label: t("cleaning_task2") },
                  { id: "task3", label: t("cleaning_task3") },
                  { id: "task4", label: t("cleaning_task4") },
                  { id: "task5", label: t("cleaning_task5") },
                ].map((item) => (
                  <label
                    key={item.id}
                    className="flex items-center gap-3 p-2.5 rounded-xl glass-card cursor-pointer hover:bg-blue-600/20 transition"
                  >
                    <input
                      type="checkbox"
                      checked={!!checkedTasks[item.id]}
                      onChange={() => toggleTask(item.id)}
                      className="w-4 h-4 accent-blue-500 rounded"
                    />
                    <span className={checkedTasks[item.id] ? "line-through text-blue-300/50" : "text-white"}>
                      {item.label}
                    </span>
                  </label>
                ))}
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-blue-800/30 text-right">
              <span className="text-[11px] text-emerald-400 font-medium">
                {t("cleaning_tasks_completed", {
                  done: Object.values(checkedTasks).filter(Boolean).length,
                  total: 5,
                })}
              </span>
            </div>
          </div>
        </div>

        {/* Action Priority Queue of Pools */}
        <div className="glass-panel rounded-2xl p-6">
          <h3 className="text-lg font-bold text-white font-heading mb-4">
            {t("cleaning_routes_title")}
          </h3>

          {isLoading ? (
            <p className="text-xs text-blue-200">{t("hub_loading")}</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {fleetData?.items?.slice(0, 6).map((pool) => (
                <div
                  key={pool.pool_id}
                  onClick={() => navigate(`/piscinas/${pool.pool_id}`)}
                  className="glass-card rounded-xl p-4 cursor-pointer hover:border-blue-400 flex items-center justify-between"
                >
                  <div>
                    <h4 className="font-bold text-white text-sm font-heading">{pool.community_name || pool.pool_id}</h4>
                    <span className="text-[11px] text-blue-300/60 font-mono">{pool.pool_id}</span>
                  </div>
                  <button className="px-3 py-1 bg-blue-600/30 hover:bg-blue-600 rounded-lg text-xs font-semibold text-white">
                    {t("cleaning_view_plan")}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
