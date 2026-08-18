import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  User,
  Globe,
  ShieldCheck,
  CheckCircle2,
  X,
} from "lucide-react";
import { api } from "../api";
import IberHeader from "../components/IberHeader";
import { AccountLockIcon } from "../components/Icons";
import LanguageSwitcher from "../components/LanguageSwitcher";
import { useI18n } from "../i18n";

export default function AccountPage() {
  const { t } = useI18n();
  const navigate = useNavigate();
  const [toast, setToast] = useState<string | null>(null);

  // System Health queries
  const { data: healthData, isLoading: healthLoading } = useQuery({
    queryKey: ["healthReady"],
    queryFn: () => api.healthReady(),
    refetchInterval: 30000,
  });

  const { data: statusData } = useQuery({
    queryKey: ["status"],
    queryFn: () => api.status(),
  });

  const triggerToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  };

  return (
    <div className="min-h-screen bg-caustic text-white flex flex-col">
      <IberHeader subtitle={t("account_subtitle")} />

      <main className="flex-1 max-w-[1400px] w-full mx-auto px-4 md:px-8 pb-16">
        {/* Header & Back button */}
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
              <AccountLockIcon size={20} />
            </div>
            <h2 className="text-xl md:text-2xl font-bold tracking-tight font-heading">
              {t("account_title")}
            </h2>
          </div>
        </div>

        {toast && (
          <div className="mb-6 p-4 rounded-xl bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-sm flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle2 size={16} />
              <span>{toast}</span>
            </div>
            <button onClick={() => setToast(null)} className="text-xs text-emerald-400 cursor-pointer">
              <X size={14} />
            </button>
          </div>
        )}

        {/* Profile Card */}
        <div className="glass-panel rounded-2xl p-6 mb-6">
          <div className="flex flex-col sm:flex-row items-center sm:items-start gap-5">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center shadow-lg border-2 border-white/20">
              <User size={36} className="text-white" />
            </div>
            <div className="flex-1 text-center sm:text-left">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <h3 className="text-2xl font-bold text-white font-heading">
                    {t("account_tech_title")}
                  </h3>
                  <p className="text-sm text-blue-300">
                    {t("account_tech_role")}
                  </p>
                </div>
                <span className="inline-block px-3 py-1 bg-blue-500/20 border border-blue-400/30 text-blue-200 rounded-full text-xs font-medium self-center sm:self-auto">
                  {t("account_zone")}
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mt-4 pt-4 border-t border-blue-800/30 text-xs">
                <div>
                  <span className="text-blue-300/70 block">{t("account_license")}</span>
                  <strong className="text-white">IBER-POOL-ES-4829</strong>
                </div>
                <div>
                  <span className="text-blue-300/70 block">{t("account_assigned_pools")}</span>
                  <strong className="text-cyan-300">100</strong>
                </div>
                <div>
                  <span className="text-blue-300/70 block">{t("account_shift_status")}</span>
                  <strong className="text-emerald-400">{t("account_shift_active")}</strong>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Language Selection Card */}
        <div className="glass-panel rounded-2xl p-6 mb-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-bold text-white font-heading mb-1 flex items-center gap-2">
              <Globe size={18} className="text-cyan-300" />
              <span>Idioma / Language</span>
            </h3>
            <p className="text-xs text-blue-300/70">
              Selecciona el idioma de la interfaz / Choose interface language
            </p>
          </div>
          <LanguageSwitcher variant="inline" />
        </div>

        {/* System & AI Engine Connectivity */}
        <div className="glass-panel rounded-2xl p-6 mb-6">
          <h3 className="text-lg font-bold text-white font-heading mb-4 flex items-center gap-2">
            <ShieldCheck size={20} className="text-cyan-300" />
            <span>{t("account_sys_health")}</span>
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="glass-card rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-blue-300/70">{t("account_db")}</span>
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
              </div>
              <div className="text-base font-bold text-white">
                {healthLoading ? "..." : healthData?.database || "PostgreSQL"}
              </div>
              <span className="text-[11px] text-emerald-400/90 mt-1 block">{t("account_db_synced")}</span>
            </div>

            <div className="glass-card rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-blue-300/70">{t("account_ai_engine")}</span>
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
              </div>
              <div className="text-base font-bold text-white">
                {statusData?.prediction?.loaded ? "Chained Physics-ML" : "Active"}
              </div>
              <span className="text-[11px] text-cyan-300 mt-1 block">
                {statusData?.prediction?.run_id ? `Run: ${statusData.prediction.run_id.slice(0, 12)}` : "Model ready"}
              </span>
            </div>

            <div className="glass-card rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-blue-300/70">{t("account_wx_service")}</span>
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
              </div>
              <div className="text-base font-bold text-white">Open-Meteo API</div>
              <span className="text-[11px] text-blue-200/80 mt-1 block">{t("account_wx_desc")}</span>
            </div>
          </div>
        </div>

        {/* App Configuration Preferences */}
        <div className="glass-panel rounded-2xl p-6">
          <h3 className="text-lg font-bold text-white font-heading mb-4">
            {t("account_prefs_title")}
          </h3>

          <div className="space-y-4">
            <div className="flex items-center justify-between py-2 border-b border-blue-800/30">
              <div>
                <strong className="text-sm text-white block">{t("account_pref_alerts")}</strong>
                <span className="text-xs text-blue-300/70">
                  {t("account_pref_alerts_desc")}
                </span>
              </div>
              <input type="checkbox" defaultChecked className="w-5 h-5 accent-blue-500 cursor-pointer" />
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => triggerToast(t("account_prefs_saved"))}
                className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs rounded-xl shadow transition cursor-pointer"
              >
                {t("account_save_prefs")}
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
