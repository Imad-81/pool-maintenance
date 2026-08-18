import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, ClipboardList, CheckCircle2, Sparkles } from "lucide-react";
import { api } from "../api";
import IberHeader from "../components/IberHeader";
import { SkimmerNetIcon } from "../components/Icons";
import { useI18n } from "../i18n";

export default function CleaningPage() {
  const navigate = useNavigate();
  const { t } = useI18n();

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

  const completedCount = Object.values(checkedTasks).filter(Boolean).length;
  const totalTasks = 5;
  const progressPercent = Math.round((completedCount / totalTasks) * 100);

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

        {/* Maintenance Protocol & Checklist Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Checklist Card */}
          <div className="lg:col-span-2 glass-panel rounded-2xl p-6">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-bold text-white font-heading flex items-center gap-2">
                <ClipboardList size={20} className="text-cyan-300" />
                <span>{t("cleaning_checklist_title")}</span>
              </h3>
              <span className="text-xs px-2.5 py-1 rounded-full bg-blue-500/20 text-cyan-300 font-semibold border border-blue-400/30">
                {completedCount}/{totalTasks} completadas
              </span>
            </div>
            <p className="text-xs text-blue-200/80 mb-6">
              {t("cleaning_checklist_desc")}
            </p>

            {/* Progress Bar */}
            <div className="w-full bg-blue-950/80 rounded-full h-2.5 mb-6 overflow-hidden border border-blue-800/40">
              <div
                className="bg-gradient-to-r from-blue-500 to-cyan-400 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${progressPercent}%` }}
              />
            </div>

            <div className="space-y-3 text-sm">
              {[
                { id: "task1", label: t("cleaning_task1") },
                { id: "task2", label: t("cleaning_task2") },
                { id: "task3", label: t("cleaning_task3") },
                { id: "task4", label: t("cleaning_task4") },
                { id: "task5", label: t("cleaning_task5") },
              ].map((item) => (
                <label
                  key={item.id}
                  className={`flex items-center gap-3 p-3.5 rounded-xl glass-card cursor-pointer hover:bg-blue-600/20 transition border ${
                    checkedTasks[item.id] ? "border-emerald-500/30 bg-emerald-950/10" : "border-blue-800/30"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={!!checkedTasks[item.id]}
                    onChange={() => toggleTask(item.id)}
                    className="w-4 h-4 accent-blue-500 rounded cursor-pointer"
                  />
                  <span className={checkedTasks[item.id] ? "line-through text-blue-300/50" : "text-white font-medium"}>
                    {item.label}
                  </span>
                  {checkedTasks[item.id] && (
                    <CheckCircle2 size={16} className="text-emerald-400 ml-auto shrink-0" />
                  )}
                </label>
              ))}
            </div>
          </div>

          {/* Quick Summary & Best Practices */}
          <div className="glass-panel rounded-2xl p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-lg font-bold text-white font-heading mb-3 flex items-center gap-2">
                <Sparkles size={18} className="text-cyan-300" />
                <span>Protocolo de Higiene</span>
              </h3>
              <p className="text-xs text-blue-200/80 mb-4 leading-relaxed">
                Cumplimiento normativo del Real Decreto 742/2013 para instalaciones de uso colectivo.
              </p>

              <div className="space-y-3">
                <div className="glass-card rounded-xl p-3 border border-blue-800/30">
                  <span className="text-[11px] text-cyan-300 font-semibold block uppercase tracking-wider mb-1">
                    Frecuencia de Skimmer
                  </span>
                  <span className="text-xs text-blue-100">
                    Limpieza obligatoria diaria antes del inicio de la jornada de baño.
                  </span>
                </div>

                <div className="glass-card rounded-xl p-3 border border-blue-800/30">
                  <span className="text-[11px] text-cyan-300 font-semibold block uppercase tracking-wider mb-1">
                    Lavado de Filtro
                  </span>
                  <span className="text-xs text-blue-100">
                    Contralavado periódico al detectar aumento de 0.3 bar en manómetro.
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-blue-800/30">
              <div className="flex items-center justify-between">
                <span className="text-xs text-blue-300/70">Estado del Turno</span>
                <span className="text-xs font-bold text-emerald-400">
                  {completedCount === totalTasks ? "100% Completado" : `${progressPercent}% En Progreso`}
                </span>
              </div>
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
                  className="glass-card rounded-xl p-4 cursor-pointer hover:border-blue-400 flex items-center justify-between transition group"
                >
                  <div>
                    <h4 className="font-bold text-white text-sm font-heading group-hover:text-cyan-300 transition-colors">
                      {pool.community_name || pool.pool_id}
                    </h4>
                    <span className="text-[11px] text-blue-300/60 font-mono">{pool.pool_id}</span>
                  </div>
                  <button className="px-3 py-1 bg-blue-600/30 hover:bg-blue-600 rounded-lg text-xs font-semibold text-white transition cursor-pointer">
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

