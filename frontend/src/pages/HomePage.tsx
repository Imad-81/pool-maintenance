import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";
import IberHeader from "../components/IberHeader";
import HubMenu from "../components/HubMenu";
import { useI18n } from "../i18n";

export default function HomePage() {
  const { t } = useI18n();

  const { data: summaryData, isLoading } = useQuery({
    queryKey: ["fleet", "summary-kpi"],
    queryFn: () => api.fleetSummary(),
  });

  const { data: statusData } = useQuery({
    queryKey: ["status-summary"],
    queryFn: () => api.status(),
  });

  const totalPools = summaryData?.total || 0;
  const urgentCount = summaryData?.counts?.Immediate || 0;


  return (
    <div className="min-h-screen flex flex-col justify-between bg-caustic relative overflow-hidden">
      {/* Decorative ambient water glow */}
      <div className="absolute top-[-10%] left-1/2 -translate-x-1/2 w-[600px] h-[350px] bg-blue-500/20 blur-[100px] rounded-full pointer-events-none" />

      {/* Main Brand Header */}
      <IberHeader />

      {/* Center 6-Module Hub Grid */}
      <main className="flex-1 flex flex-col justify-center items-center py-4 z-10">
        <HubMenu urgentCount={urgentCount} totalPools={totalPools} />
      </main>

      {/* Bottom Quick Status Strip */}
      <footer className="w-full max-w-[540px] mx-auto px-4 pb-6 z-10">
        <div className="glass-panel rounded-2xl p-4 flex items-center justify-between text-xs text-blue-100">
          <div className="flex items-center gap-3">
            <span className="flex h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse" />
            <div>
              <span className="font-semibold text-white">Iberpiscinas AI</span>
              <span className="text-blue-200/80 ml-2">
                {statusData?.prediction?.loaded ? t("hub_ai_active") : t("hub_ai_ready")}
              </span>
            </div>
          </div>
          <Link
            to="/piscinas"
            className="px-3 py-1.5 rounded-xl bg-blue-500/20 hover:bg-blue-500/40 border border-blue-400/30 text-white font-medium text-xs transition"
          >
            {isLoading ? t("hub_loading") : `${totalPools} ${t("hub_pools")}`}
          </Link>
        </div>
      </footer>
    </div>
  );
}
