import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import type { UploadPreview } from "../types";
import IberHeader from "../components/IberHeader";
import { ToolsIncidentIcon } from "../components/Icons";
import { useI18n } from "../i18n";

export default function IncidentsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { t, lang } = useI18n();
  const [activeTab, setActiveTab] = useState<"breaches" | "manual" | "csv" | "logs">("breaches");
  const [msg, setMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const targetFields = [
    { key: "pool_id", label: lang === "en" ? "Pool ID (Required)" : "ID de Piscina (Obligatorio)", required: true },
    { key: "reading_date", label: lang === "en" ? "Reading Date (Required)" : "Fecha de Lectura (Obligatorio)", required: true },
    { key: "free_chlorine", label: lang === "en" ? "Free Chlorine (mg/L)" : "Cloro Libre (mg/L)", required: false },
    { key: "ph", label: "pH", required: false },
    { key: "turbidity", label: lang === "en" ? "Turbidity (NTU)" : "Turbidez (NTU)", required: false },
    { key: "pool_volume_m3", label: lang === "en" ? "Volume (m³)" : "Volumen (m³)", required: false },
    { key: "community_name", label: lang === "en" ? "Community Name" : "Nombre Comunidad", required: false },
    { key: "hypochlorite_dosing_pct", label: lang === "en" ? "Dosing Pump %" : "% Bomba Dosificadora", required: false },
    { key: "hypochlorite_dosing_hours", label: lang === "en" ? "Dosing Hours" : "Horas de Dosificación", required: false },
  ];

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
      setMsg({
        type: "success",
        text: lang === "en"
          ? `File analyzed: ${data.total_rows} rows detected.`
          : `Archivo analizado: ${data.total_rows} filas detectadas.`,
      });
    },
    onError: (err: Error) =>
      setMsg({
        type: "error",
        text: `${lang === "en" ? "Upload error" : "Error al subir archivo"}: ${err.message}`,
      }),
  });

  const mapColumnsMut = useMutation({
    mutationFn: () => api.mapColumns(preview!.upload_id, mapping),
    onSuccess: (res) => {
      setMsg({
        type: "success",
        text: lang === "en"
          ? `Import successful! Loaded ${res.loaded_rows} readings across ${res.pool_count} pools.`
          : `¡Importación completada! Se procesaron ${res.loaded_rows} lecturas de ${res.pool_count} piscinas.`,
      });
      setPreview(null);
      queryClient.invalidateQueries({ queryKey: ["fleet"] });
      queryClient.invalidateQueries({ queryKey: ["admin-ingest-logs"] });
    },
    onError: (err: Error) =>
      setMsg({
        type: "error",
        text: `${lang === "en" ? "Import error" : "Error en la importación"}: ${err.message}`,
      }),
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
      setMsg({
        type: "success",
        text: lang === "en"
          ? `Reading saved successfully for pool ${res.pool_id}.`
          : `Lectura guardada con éxito para la piscina ${res.pool_id}.`,
      });
      setManualForm((prev) => ({ ...prev, pool_id: "", community_name: "" }));
      queryClient.invalidateQueries({ queryKey: ["fleet"] });
    },
    onError: (err: Error) => setMsg({ type: "error", text: `Error: ${err.message}` }),
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
      <IberHeader subtitle={t("incidents_subtitle")} />

      <main className="flex-1 max-w-[1200px] w-full mx-auto px-4 md:px-8 pb-16">
        {/* Navigation & Header */}
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
                <ToolsIncidentIcon size={20} />
              </div>
              <h2 className="text-xl md:text-2xl font-bold tracking-tight font-heading">
                {t("incidents_title")}
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
            label={t("incidents_tab_breaches")}
            count={incidentPools.length}
            active={activeTab === "breaches"}
            color="text-red-400"
            onClick={() => setActiveTab("breaches")}
          />
          <TabButton
            label={t("incidents_tab_manual")}
            active={activeTab === "manual"}
            onClick={() => setActiveTab("manual")}
          />
          <TabButton
            label={t("incidents_tab_csv")}
            active={activeTab === "csv"}
            onClick={() => setActiveTab("csv")}
          />
          <TabButton
            label={t("incidents_tab_logs")}
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
                {t("hub_loading")}
              </div>
            ) : incidentPools.length === 0 ? (
              <div className="glass-panel rounded-2xl p-12 text-center text-emerald-300">
                <div className="text-4xl mb-3">✓</div>
                <h4 className="text-lg font-bold font-heading">{t("incidents_no_breaches_title")}</h4>
                <p className="text-xs text-blue-200/80 mt-1">
                  {t("incidents_no_breaches_desc")}
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
                        {t("incidents_urgent_notice")}
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-2 text-center bg-blue-950/60 p-3 rounded-xl border border-blue-800/40 mb-3">
                      <div>
                        <span className="text-[10px] text-blue-300/70 block">{t("pools_cl")}</span>
                        <span className="text-sm font-bold text-red-400">
                          {pool.free_chlorine != null ? `${pool.free_chlorine.toFixed(2)}` : "—"} mg/L
                        </span>
                      </div>
                      <div>
                        <span className="text-[10px] text-blue-300/70 block">{t("pools_ph")}</span>
                        <span className="text-sm font-bold text-white">
                          {pool.ph != null ? pool.ph.toFixed(2) : "—"}
                        </span>
                      </div>
                      <div>
                        <span className="text-[10px] text-blue-300/70 block">{t("pools_risk_24_48h")}</span>
                        <span className="text-sm font-bold text-amber-400">
                          {Math.round((pool.breach_proba || 0) * 100)}%
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-2">
                      <span className="text-[11px] text-blue-300/60">
                        {t("pools_last_reading")}: {pool.last_reading_date?.slice(0, 10) || "N/D"}
                      </span>
                      <button
                        onClick={() => navigate(`/piscinas/${pool.pool_id}`)}
                        className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-semibold"
                      >
                        {t("pools_diagnosis")}
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
              {t("incidents_manual_title")}
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="text-xs text-blue-300/80 block mb-1">{t("incidents_pool_id_req")}</label>
                <input
                  type="text"
                  placeholder="ej. ESP-001"
                  value={manualForm.pool_id}
                  onChange={(e) => setManualForm({ ...manualForm, pool_id: e.target.value })}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">{t("incidents_comm_name")}</label>
                <input
                  type="text"
                  placeholder="ej. Comunidad Los Naranjos"
                  value={manualForm.community_name}
                  onChange={(e) => setManualForm({ ...manualForm, community_name: e.target.value })}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">{t("incidents_date_req")}</label>
                <input
                  type="datetime-local"
                  value={manualForm.reading_date}
                  onChange={(e) => setManualForm({ ...manualForm, reading_date: e.target.value })}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">{t("incidents_volume_m3")}</label>
                <input
                  type="number"
                  value={manualForm.pool_volume_m3}
                  onChange={(e) => setManualForm({ ...manualForm, pool_volume_m3: e.target.value })}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">{t("incidents_free_cl")}</label>
                <input
                  type="number"
                  step="0.05"
                  value={manualForm.free_chlorine}
                  onChange={(e) => setManualForm({ ...manualForm, free_chlorine: e.target.value })}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">{t("incidents_ph")}</label>
                <input
                  type="number"
                  step="0.05"
                  value={manualForm.ph}
                  onChange={(e) => setManualForm({ ...manualForm, ph: e.target.value })}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">{t("incidents_turb")}</label>
                <input
                  type="number"
                  step="0.1"
                  value={manualForm.turbidity}
                  onChange={(e) => setManualForm({ ...manualForm, turbidity: e.target.value })}
                  className="w-full px-3.5 py-2 rounded-xl glass-card text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="text-xs text-blue-300/80 block mb-1">{t("incidents_dosing_pct")}</label>
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
                {addReadingMut.isPending ? "..." : t("incidents_save_btn")}
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
                  {t("incidents_csv_drag_title")}
                </h4>
                <p className="text-xs text-blue-300/70 mb-4 max-w-md mx-auto">
                  {t("incidents_csv_drag_desc")}
                </p>
                <label className="inline-block px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs rounded-xl cursor-pointer shadow transition">
                  <span>{t("incidents_select_file")}</span>
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
                    <h4 className="font-bold text-white font-heading">{t("incidents_mapping_title")} ({preview.filename})</h4>
                    <span className="text-xs text-blue-300/70">{preview.total_rows} rows</span>
                  </div>
                  <button
                    onClick={() => setPreview(null)}
                    className="text-xs text-blue-300 hover:text-white"
                  >
                    {t("incidents_cancel")}
                  </button>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
                  {targetFields.map((f) => (
                    <div key={f.key} className="glass-card rounded-xl p-3">
                      <label className="text-xs font-semibold text-blue-200 block mb-1">{f.label}</label>
                      <select
                        value={mapping[f.key] || ""}
                        onChange={(e) => setMapping({ ...mapping, [f.key]: e.target.value })}
                        className="w-full px-2.5 py-1.5 rounded-lg bg-blue-950 text-xs text-white border border-blue-700/50 focus:outline-none"
                      >
                        <option value="">(Ignore column)</option>
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
                    {t("incidents_back")}
                  </button>
                  <button
                    disabled={mapColumnsMut.isPending}
                    onClick={() => mapColumnsMut.mutate()}
                    className="px-5 py-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs rounded-xl shadow"
                  >
                    {mapColumnsMut.isPending ? "..." : t("incidents_run_import")}
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
              {t("incidents_logs_title")}
            </h3>

            {logsLoading ? (
              <p className="text-xs text-blue-200">...</p>
            ) : (ingestLogs?.length || 0) === 0 ? (
              <p className="text-xs text-blue-300/70">No logs.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-blue-800/40 text-blue-300/70">
                      <th className="pb-3 font-semibold">Date</th>
                      <th className="pb-3 font-semibold">Source / File</th>
                      <th className="pb-3 font-semibold">Pools</th>
                      <th className="pb-3 font-semibold">Loaded Rows</th>
                      <th className="pb-3 font-semibold">Skipped</th>
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
