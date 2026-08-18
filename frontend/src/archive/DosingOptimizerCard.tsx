import { FlaskConical, Info } from "lucide-react";
import type { OptimiserResult } from "../types";
import { useI18n } from "../i18n";

interface DosingOptimizerCardProps {
  optimiser?: OptimiserResult | null;
}

/**
 * Archived Chemical Dosing Optimizer Recommendation Card.
 * Originally displayed in PoolDetailPage.tsx.
 */
export default function DosingOptimizerCard({ optimiser }: DosingOptimizerCardProps) {
  const { t } = useI18n();

  if (!optimiser) return null;

  return (
    <div className="glass-panel rounded-2xl p-6 mb-6 border border-cyan-500/30">
      <div className="flex items-center gap-2 mb-3">
        <FlaskConical size={20} className="text-cyan-300" />
        <h3 className="text-lg font-bold text-white font-heading">
          {t("detail_optimizer_title")}
        </h3>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 bg-blue-950/50 p-4 rounded-xl border border-blue-800/30 mb-3">
        <div>
          <span className="text-xs text-blue-300/70 block">{t("detail_pump_power")}</span>
          <span className="text-xl font-bold text-cyan-300 font-heading">
            {optimiser.recommended_dosing.hypochlorite_dosing_pct}%
          </span>
        </div>
        <div>
          <span className="text-xs text-blue-300/70 block">{t("detail_recirc_time")}</span>
          <span className="text-xl font-bold text-cyan-300 font-heading">
            {optimiser.recommended_dosing.hypochlorite_dosing_hours} h
          </span>
        </div>
        <div>
          <span className="text-xs text-blue-300/70 block">{t("detail_projected_cl")}</span>
          <span className="text-xl font-bold text-emerald-400 font-heading">
            {optimiser.predicted_tomorrow.free_chlorine.toFixed(2)} mg/L
          </span>
        </div>
      </div>
      {optimiser.reasons && optimiser.reasons.length > 0 && (
        <p className="text-xs text-blue-200/80 flex items-center gap-1.5">
          <Info size={14} className="text-cyan-300 shrink-0" />
          <span>{optimiser.reasons.join(" • ")}</span>
        </p>
      )}
    </div>
  );
}
