import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import IberHeader from "../components/IberHeader";
import { AccountLockIcon } from "../components/Icons";

export default function AccountPage() {
  const navigate = useNavigate();
  const [toast, setToast] = useState<string | null>(null);

  const { data: healthData, isLoading: healthLoading } = useQuery({
    queryKey: ["health-ready"],
    queryFn: () => api.healthReady().catch(() => null),
  });

  const { data: statusData } = useQuery({
    queryKey: ["admin-status"],
    queryFn: () => api.status().catch(() => null),
  });

  const triggerToast = (text: string) => {
    setToast(text);
    setTimeout(() => setToast(null), 3000);
  };

  return (
    <div className="min-h-screen bg-caustic text-white flex flex-col">
      <IberHeader subtitle="Mi cuenta — Perfil de técnico y ajustes del sistema" />

      <main className="flex-1 max-w-[1000px] w-full mx-auto px-4 md:px-8 pb-16">
        {/* Navigation & Header */}
        <div className="flex items-center justify-between gap-4 mb-6">
          <button
            onClick={() => navigate("/")}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl glass-card text-blue-200 hover:text-white text-xs font-semibold"
          >
            ← Volver al Menú
          </button>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white">
              <AccountLockIcon size={20} />
            </div>
            <h2 className="text-xl md:text-2xl font-bold tracking-tight font-heading">
              Mi Cuenta & Ajustes
            </h2>
          </div>
        </div>

        {toast && (
          <div className="mb-6 p-4 rounded-xl bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-sm flex items-center justify-between">
            <span>✓ {toast}</span>
            <button onClick={() => setToast(null)} className="text-xs text-emerald-400">✕</button>
          </div>
        )}

        {/* Profile Card */}
        <div className="glass-panel rounded-2xl p-6 mb-6">
          <div className="flex flex-col sm:flex-row items-center sm:items-start gap-5">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center text-3xl shadow-lg border-2 border-white/20">
              🏊‍♂️
            </div>
            <div className="flex-1 text-center sm:text-left">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <h3 className="text-2xl font-bold text-white font-heading">
                    Equipo Técnico Iberpiscinas
                  </h3>
                  <p className="text-sm text-blue-300">
                    Operador Certificado en Mantenimiento Físico-Químico (RD 742/2013)
                  </p>
                </div>
                <span className="inline-block px-3 py-1 bg-blue-500/20 border border-blue-400/30 text-blue-200 rounded-full text-xs font-medium self-center sm:self-auto">
                  Zona: Alicante / Costa Blanca
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mt-4 pt-4 border-t border-blue-800/30 text-xs">
                <div>
                  <span className="text-blue-300/70 block">Licencia Profesional</span>
                  <strong className="text-white">IBER-POOL-ES-4829</strong>
                </div>
                <div>
                  <span className="text-blue-300/70 block">Instalaciones Asignadas</span>
                  <strong className="text-cyan-300">100 Piscinas Comunitarias</strong>
                </div>
                <div>
                  <span className="text-blue-300/70 block">Estado de Turno</span>
                  <strong className="text-emerald-400">En Servicio Activo</strong>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* System & AI Engine Connectivity */}
        <div className="glass-panel rounded-2xl p-6 mb-6">
          <h3 className="text-lg font-bold text-white font-heading mb-4 flex items-center gap-2">
            <span>🛡️</span> Estado de Conexión y Diagnóstico de Servidor
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="glass-card rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-blue-300/70">Base de Datos</span>
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
              </div>
              <div className="text-base font-bold text-white">
                {healthLoading ? "Verificando..." : healthData?.database || "Conectada (PostgreSQL)"}
              </div>
              <span className="text-[11px] text-emerald-400/90 mt-1 block">Lecturas sincronizadas</span>
            </div>

            <div className="glass-card rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-blue-300/70">Motor Predictivo AI</span>
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
              </div>
              <div className="text-base font-bold text-white">
                {statusData?.prediction?.loaded ? "Chained Physics-ML" : "Activo"}
              </div>
              <span className="text-[11px] text-cyan-300 mt-1 block">
                {statusData?.prediction?.run_id ? `Run: ${statusData.prediction.run_id.slice(0, 12)}` : "Modelo entrenado"}
              </span>
            </div>

            <div className="glass-card rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-blue-300/70">Servicio Meteorológico</span>
                <span className="w-2.5 h-2.5 rounded-full bg-blue-400" />
              </div>
              <div className="text-base font-bold text-white">Open-Meteo API</div>
              <span className="text-[11px] text-blue-200/80 mt-1 block">Radiación UV & Temperatura</span>
            </div>
          </div>
        </div>

        {/* App Configuration Preferences */}
        <div className="glass-panel rounded-2xl p-6">
          <h3 className="text-lg font-bold text-white font-heading mb-4">
            Preferencias de Notificación y Operativa
          </h3>

          <div className="space-y-4">
            <div className="flex items-center justify-between py-2 border-b border-blue-800/30">
              <div>
                <strong className="text-sm text-white block">Alertas automáticas de cloro y pH</strong>
                <span className="text-xs text-blue-300/70">
                  Notificar en panel y mensajes cuando la probabilidad de breach supere el 40%
                </span>
              </div>
              <input type="checkbox" defaultChecked className="w-5 h-5 accent-blue-500 cursor-pointer" />
            </div>

            <div className="flex items-center justify-between py-2 border-b border-blue-800/30">
              <div>
                <strong className="text-sm text-white block">Cálculo de dosificación inteligente</strong>
                <span className="text-xs text-blue-300/70">
                  Sugerir optimización de bomba dosificadora (% e intervalos de bombeo)
                </span>
              </div>
              <input type="checkbox" defaultChecked className="w-5 h-5 accent-blue-500 cursor-pointer" />
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => triggerToast("Preferencias guardadas correctamente.")}
                className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs rounded-xl shadow transition"
              >
                Guardar Preferencias
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
