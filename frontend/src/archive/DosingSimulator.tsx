import { useState } from "react";
import { FlaskConical } from "lucide-react";
import { useI18n } from "../i18n";

interface DosingSimulatorProps {
  initialVolume?: number;
  initialCurrentCl?: number;
  initialTargetCl?: number;
  initialPumpCapacity?: number;
}

/**
 * Archived Chemical Dosing Optimizer Simulator Component.
 * Originally displayed in CleaningPage.tsx.
 */
export default function DosingSimulator({
  initialVolume = 150,
  initialCurrentCl = 0.8,
  initialTargetCl = 1.5,
  initialPumpCapacity = 5.0,
}: DosingSimulatorProps) {
  const { t } = useI18n();

  // Interactive Dosing Simulator State
  const [volume, setVolume] = useState<number>(initialVolume);
  const [currentCl, setCurrentCl] = useState<number>(initialCurrentCl);
  const [targetCl, setTargetCl] = useState<number>(initialTargetCl);
  const [pumpCapacityLh, setPumpCapacityLh] = useState<number>(initialPumpCapacity);

  const deltaCl = Math.max(0, targetCl - currentCl);
  const requiredGramsPureCl = deltaCl * volume;
  const hypochloriteConcentration = 0.15;
  const hypochloriteDensity = 1.2;
  const requiredGramsSolution = requiredGramsPureCl / hypochloriteConcentration;
  const requiredLitersProduct = requiredGramsSolution / (hypochloriteDensity * 1000);
  const pumpHours = Number((requiredLitersProduct / (pumpCapacityLh * 0.7)).toFixed(1));
  const recommendedPumpPct = 70;

  return (
    <div className="glass-panel rounded-2xl p-6">
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
  );
}
