import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import type { UploadPreview } from "../types";
import IberHeader from "../components/IberHeader";
import { ToolsIncidentIcon } from "../components/Icons";

const TARGET_FIELDS = [
  { key: "pool_id", label: "ID de Piscina (Obligatorio)", required: true },
  { key: "reading_date", label: "Fecha de Lectura (Obligatorio)", required: true },
  { key: "free_chlorine", label: "Cloro Libre (mg/L)", required: false },
  { key: "ph", label: "pH", required: false },
  { key: "turbidity", label: "Turbidez (NTU)", required: false },
  { key: "pool_volume_m3", label: "Volumen (m³)", required: false },
  { key: "community_name", label: "Nombre Comunidad", required: false },
  { key: "hypochlorite_dosing_pct", label: "% Bomba Dosificadora", required: false },
  { key: "hypochlorite_dosing_hours", label: "Horas de Dosificación", required: false },
];

export default function IncidentsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<"breaches" | "manual" | "csv" | "logs">("breaches");
  const [msg, setMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // Queries
  const { data: fleetData, isLoading: fleetLoading } = useQuery({
    queryKey: ["fleet", "incidents-data"],
    queryFn: () => api.fleet({ page_size: 100 }),
  });

  const { data: ingestLogs, isLoading: logsLoading } = useQuery({
    queryKey: ["admin-ingest-logs"],
    queryFn: () => api.admin.ingestLog(),
  });

  // Manual Reading Form
  const [manualForm, setManualForm] = useState({
    pool_id: "",
    community_name: "",
    reading_date: new Date().toISOString().slice(0, 16),
    ph: "7.4",
    free_chlorine: "1.5",
    turbidity: "0.5",
    pool_volume_m3: "150",
    hypochlorite_dosing_pct: "50",
    hypochlorite_dosing_hours: "3",
  });

  // CSV Upload Flow
  const [preview, setPreview] = useState<UploadPreview | null>(null);
  const [mapping, setMapping] = useState<Record<string, string>>({});

  const uploadMut = useMutation({
    mutationFn: (f: File) => api.uploadFile(f),
    onSuccess: (data) => {
      setPreview(data);
      setMapping(data.suggested_mapping);
      setMsg({ type: "success", text: `Archivo analizado: ${data.total_rows} filas detectadas.` });
    },
    onError: (err: Error) => setMsg({ type: "error", text: `Error al subir archivo: ${err.message}` }),
  });

  const mapColumnsMut = useMutation({
    mutationFn: () => api.mapColumns(preview!.upload_id, mapping),
    onSuccess: (res) => {
      setMsg({
        type: "success",
        text: `¡Importación completada! Se procesaron ${res.loaded_rows} lecturas de ${res.pool_count} piscinas.`,
      });
      setPreview(null);
      queryClient.invalidateQueries({ queryKey: ["fleet"] });
      queryClient.invalidateQueries({ queryKey: ["admin-ingest-logs"] });
    },
    onError: (err: Error) => setMsg({ type: "error", text: `Error en la importación: ${err.message}` }),
  });

  const addReadingMut = useMutation({
    mutationFn: () =>
      api.addReading({
        pool_id: manualForm.pool_id,
        reading_date: manualForm.reading_date,
        ph: manualForm.ph ? parseFloat(manualForm.ph) : undefined,
        free_chlorine: manualForm.free_chlorine ? parseFloat(manualForm.free_chlorine) : undefined,
        turbidity: manualForm.turbidity ? parseFloat(manualForm.turbidity) : undefined,
        pool_volume_m3: manualForm.pool_volume_m3 ? parseFloat(manualForm.pool_volume_m3) : undefined,
        community_name: manualForm.community_name || undefined,
        hypochlorite_dosing_pct: manualForm.hypochlorite_dosing_pct
          ? parseFloat(manualForm.hypochlorite_dosing_pct)
          : undefined,
        hypochlorite_dosing_hours: manualForm.hypochlorite_dosing_hours
          ? parseFloat(manualForm.hypochlorite_dosing_hours)
          : undefined,
      }),
    onSuccess: (res) => {
      setMsg({ type: "success", text: `Lectura guardada con éxito para la piscina ${res.pool_id}.` });
      setManualForm((prev) => ({ ...prev, pool_id: "", community_name: "" }));
      queryClient.invalidateQueries({ queryKey: ["fleet"] });
    },
    onError: (err: Error) => setMsg({ type: "error", text: `Error al guardar: ${err.message}` }),
  });

  const incidentPools = (fleetData?.items || []).filter(
    (p) =>
      p.urgency === "Immediate" ||
      p.urgency === "URGENT" ||
      (p.free_chlorine != null && (p.free_chlorine < 0.5 || p.free_chlorine > 2.0)) ||
      (p.ph != null && (p.ph < 7.2 || p.ph > 8.0))
  );

  return (
    <div className="min-h-screen bg-caustic text-white flex flex-col">
      <IberHeader subtitle="Incidencias — Gestión de no conformidades e ingesta de datos" />

      <main className="flex-1 max-w-[1200px] w-full mx-auto px-4 md:px-8 pb-16">
        {/* Navigation & Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate("/")}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl glass-card text-blue-200 hover:text-white text-xs font-semibold"
            >
              ← Volver al Menú
            </button>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white">
                <ToolsIncidentIcon size={20} />
              </div>
              <h2 className="text-xl md:text-2xl font-bold tracking-tight font-heading">
                Incidencias & Registro
              </h2>
            </div>
          </div>
        </div>

        {/* Message Banner */}
        {msg && (
          <div
            className={`mb-6 p-4 rounded-xl text-sm flex items-center justify-between ${
              msg.type === "success"
                ? "bg-emerald-500/20 border border-emerald-500/40 text-emerald-300"
                : "bg-red-500/20 border border-red-500/40 text-red-300"
            }`}
          >
            <span>{msg.text}</span>
            <button onClick={() => setMsg(null)} className="text-xs">✕</button>
          </div>
        )}

        {/* Navigation Tabs */}
        <div className="flex gap-2 mb-6 overflow-x-auto pb-1">
          <TabButton
            label="Incidencias Activas"
            count={incidentPools.length}
            active={activeTab === "breaches"}
            color="text-red-400"
            onClick={() => setActiveTab("breaches")}
          />
          <TabButton
            label="Registrar Lectura Manual"
            active={activeTab === "manual"}
            onClick={() => setActiveTab("manual")}
          />
          <TabButton
            label="Importar Archivo CSV"
            active={activeTab === "csv"}
            onClick={() => setActiveTab("csv")}
          />
          <TabButton
            label="Historial de Ingesta"
            count={ingestLogs?.length || 0}
            active={activeTab === "logs"}
            onClick={() => setActiveTab("logs")}
          />
        </div>

        {/* Tab 1: Incidencias Activas */}
        {activeTab === "breaches" && (
          <div>
            {fleetLoading ? (
              <div className="glass-panel rounded-2xl p-12 text-center text-blue-200">
                Verificando parámetros...
              </div>
            ) : incidentPools.length === 0 ? (
              <div className="glass-panel rounded-2xl p-12 text-center text-emerald-300">
                <div className="text-4xl mb-3">✓</div>
                <h4 className="text-lg font-bold font-heading">Sin incidencias sanitarias activas</h4>
                <p className="text-xs text-blue-200/80 mt-1">
                  Todas las piscinas monitorizadas cumplen los parámetros RD 742/2013.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {incidentPools.map((pool) => (
                  <div key={pool.pool_id} className="glass-card rounded-2xl p-5 border border-red-500/40 bg-red-950/20">
                    <div className="flex items-start justify-between gap-2 mb-3">
                      <div>
                        <h4 className="font-bold text-white font-heading text-base">
                          {pool.community_name || pool.pool_id}
                        </h4>
                        <span className="text-xs text-blue-300/60 font-mono">{pool.pool_id}</span>
                      </div>
                      <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-red-500/30 text-red-300 border border-red-500/40">
                        Atención Urgente
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-2 text-center bg-blue-950/60 p-3 rounded-xl border border-blue-800/40 mb-3">
                      <div>
                        <span className="text-[10px] text-blue-300/70 block">Cloro Libre</span>
                        <span className="text-sm font-bold text-red-400">
                          {pool.free_chlorine != null ? `${pool.free_chlorine.toFixed(2)}` : "—"} mg/L
                        </span>
                      </div>
                      <div>
                        <span className="text-[10px] text-blue-300/70 block">pH</span>
                        <span className="text-sm font-bold text-white">
                          {pool.ph != null ? pool.ph.toFixed(2) : "—"}
                        </span>
                      </div>
                      <div>
                        <span className="text-[10px] text-blue-300/70 block">Riesgo 24-48h</span>
                        <span className="text-sm font-bold text-amber-400">
                          {Math.round((pool.breach_proba || 0) * 100)}%
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-2">
                      <span className="text-[11px] text-blue-300/60">
                        Última lectura: {pool.last_reading_date?.slice(0, 10) || "N/D"}
                      </span>
                      <button
                        onClick={() => navigate(`/piscinas/${pool.pool_id}`)}
                        className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-semibold"
                      >
                        Ver Diagnóstico y Dosis →
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Registrar Lectura Manual */}
        {activeTab === "manual" && (
          <div className="glass-panel rounded-2xl p-6 max-w-[800px] mx-auto">
            <h3 className="text-lg font-bold text-white font-heading mb-4">
              Registrar Nueva Lectura de Campo
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="text-xs text-blue-300/80 block mb-1">ID de Piscina *</label>
                <input
                  type="text"
                  placeholder="ej. ESP-001"
                  value={manualForm.pool_id}
                  onChange={(e) => setManualForm({ ...manualForm, pool_id: e.target.value })}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">Nombre Comunidad</label>
                <input
                  type="text"
                  placeholder="ej. Comunidad Los Naranjos"
                  value={manualForm.community_name}
                  onChange={(e) => setManualForm({ ...manualForm, community_name: e.target.value })}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">Fecha y Hora de Medición *</label>
                <input
                  type="datetime-local"
                  value={manualForm.reading_date}
                  onChange={(e) => setManualForm({ ...manualForm, reading_date: e.target.value })}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">Volumen de Piscina (m³)</label>
                <input
                  type="number"
                  value={manualForm.pool_volume_m3}
                  onChange={(e) => setManualForm({ ...manualForm, pool_volume_m3: e.target.value })}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">Cloro Libre (mg/L)</label>
                <input
                  type="number"
                  step="0.05"
                  value={manualForm.free_chlorine}
                  onChange={(e) => setManualForm({ ...manualForm, free_chlorine: e.target.value })}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">pH</label>
                <input
                  type="number"
                  step="0.05"
                  value={manualForm.ph}
                  onChange={(e) => setManualForm({ ...manualForm, ph: e.target.value })}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">Turbidez (NTU)</label>
                <input
                  type="number"
                  step="0.1"
                  value={manualForm.turbidity}
                  onChange={(e) => setManualForm({ ...manualForm, turbidity: e.target.value })}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">Potencia Dosificadora (%)</label>
                <input
                  type="number"
                  value={manualForm.hypochlorite_dosing_pct}
                  onChange={(e) => setManualForm({ ...manualForm, hypochlorite_dosing_pct: e.target.value })}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>
            </div>

            <div className="flex justify-end pt-3">
              <button
                disabled={!manualForm.pool_id || !manualForm.reading_date || addReadingMut.isPending}
                onClick={() => addReadingMut.mutate()}
                className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs rounded-xl shadow transition disabled:opacity-50"
              >
                {addReadingMut.isPending ? "Guardando..." : "Guardar Lectura en Base de Datos"}
              </button>
            </div>
          </div>
        )}

        {/* Tab 3: Importar Archivo CSV */}
        {activeTab === "csv" && (
          <div className="glass-panel rounded-2xl p-6">
            {!preview ? (
              <div className="text-center py-10 border-2 border-dashed border-blue-600/40 rounded-2xl p-8 hover:border-blue-400 transition">
                <div className="text-4xl mb-3">📁</div>
                <h4 className="text-base font-bold text-white mb-1 font-heading">
                  Cargar Archivo CSV o Excel con Lecturas
                </h4>
                <p className="text-xs text-blue-300/70 mb-4 max-w-md mx-auto">
                  El sistema detectará automáticamente las columnas y te permitirá mapearlas a los campos de Iberpiscinas.
                </p>
                <label className="inline-block px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs rounded-xl cursor-pointer shadow transition">
                  <span>Seleccionar Archivo CSV</span>
                  <input
                    type="file"
                    accept=".csv"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) uploadMut.mutate(file);
                    }}
                  />
                </label>
              </div>
            ) : (
              <div>
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h4 className="font-bold text-white font-heading">Mapeo de Columnas ({preview.filename})</h4>
                    <span className="text-xs text-blue-300/70">{preview.total_rows} filas encontradas</span>
                  </div>
                  <button
                    onClick={() => setPreview(null)}
                    className="text-xs text-blue-300 hover:text-white"
                  >
                    Cancelar
                  </button>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
                  {TARGET_FIELDS.map((f) => (
                    <div key={f.key} className="glass-card rounded-xl p-3">
                      <label className="text-xs font-semibold text-blue-200 block mb-1">{f.label}</label>
                      <select
                        value={mapping[f.key] || ""}
                        onChange={(e) => setMapping({ ...mapping, [f.key]: e.target.value })}
                        className="w-full px-2.5 py-1.5 rounded-lg bg-blue-950 text-xs text-white border border-blue-700/50 focus:outline-none"
                      >
                        <option value="">(Ignorar columna)</option>
                        {preview.columns.map((c) => (
                          <option key={c} value={c}>
                            {c}
                          </option>
                        ))}
                      </select>
                    </div>
                  ))}
                </div>

                <div className="flex justify-end gap-3">
                  <button
                    onClick={() => setPreview(null)}
                    className="px-4 py-2 glass-card text-xs text-blue-200"
                  >
                    Volver
                  </button>
                  <button
                    disabled={mapColumnsMut.isPending}
                    onClick={() => mapColumnsMut.mutate()}
                    className="px-5 py-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs rounded-xl shadow"
                  >
                    {mapColumnsMut.isPending ? "Importando..." : "Ejecutar Importación"}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 4: Historial de Ingesta */}
        {activeTab === "logs" && (
          <div className="glass-panel rounded-2xl p-6">
            <h3 className="text-lg font-bold text-white font-heading mb-4">
              Registro Histórico de Importaciones
            </h3>

            {logsLoading ? (
              <p className="text-xs text-blue-200">Cargando registros...</p>
            ) : (ingestLogs?.length || 0) === 0 ? (
              <p className="text-xs text-blue-300/70">No hay registros de ingesta previos.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-blue-800/40 text-blue-300/70">
                      <th className="pb-3 font-semibold">Fecha</th>
                      <th className="pb-3 font-semibold">Fuente / Archivo</th>
                      <th className="pb-3 font-semibold">Piscinas</th>
                      <th className="pb-3 font-semibold">Filas Procesadas</th>
                      <th className="pb-3 font-semibold">Omitidas</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-blue-900/30">
                    {ingestLogs?.map((log) => (
                      <tr key={log.id} className="hover:bg-blue-600/10">
                        <td className="py-3 text-blue-200/80">{log.created_at?.slice(0, 19).replace("T", " ")}</td>
                        <td className="py-3 font-mono text-cyan-300">{log.filename || log.source}</td>
                        <td className="py-3 text-white font-semibold">{log.pool_count}</td>
                        <td className="py-3 text-emerald-400 font-semibold">{log.row_count}</td>
                        <td className="py-3 text-blue-300/50">{log.skipped_count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

function TabButton({
  label,
  count,
  active,
  color,
  onClick,
}: {
  label: string;
  count?: number;
  active: boolean;
  color?: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 rounded-xl text-xs font-semibold transition whitespace-nowrap ${
        active
          ? "bg-blue-600 text-white shadow"
          : "glass-card text-blue-200/80 hover:text-white"
      }`}
    >
      {label} {count !== undefined && <span className={color || "text-blue-300"}>({count})</span>}
    </button>
  );
}
