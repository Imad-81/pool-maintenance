import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  UploadCloud,
  PenTool,
  X,
  CheckCircle2,
  AlertTriangle,
  FileSpreadsheet,
} from "lucide-react";
import { api } from "../api";
import type { UploadPreview } from "../types";

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

const REQUIRED_TARGETS = [
  { key: "pool_id", label: "Pool ID (Required)", required: true },
  { key: "reading_date", label: "Reading Date (Required)", required: true },
  { key: "free_chlorine", label: "Free Chlorine mg/L", required: false },
  { key: "ph", label: "pH Level", required: false },
  { key: "turbidity", label: "Turbidity NTU", required: false },
  { key: "pool_volume_m3", label: "Volume m³", required: false },
  { key: "community_name", label: "Community Name", required: false },
  { key: "hypochlorite_dosing_pct", label: "Dosing Pump %", required: false },
  { key: "hypochlorite_dosing_hours", label: "Dosing Hours", required: false },
];

export default function IngestModal({ isOpen, onClose }: Props) {
  const queryClient = useQueryClient();
  const [tab, setTab] = useState<"upload" | "manual">("upload");
  const [msg, setMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // --- Upload flow state ---
  const [preview, setPreview] = useState<UploadPreview | null>(null);
  const [mapping, setMapping] = useState<Record<string, string>>({});

  // --- Manual form state ---
  const [manualForm, setManualForm] = useState({
    pool_id: "",
    community_name: "",
    reading_date: new Date().toISOString().slice(0, 16),
    ph: "7.4",
    free_chlorine: "1.5",
    turbidity: "0.5",
    pool_volume_m3: "150",
    hypochlorite_dosing_pct: "",
    hypochlorite_dosing_hours: "",
  });

  const uploadFileMut = useMutation({
    mutationFn: (f: File) => api.uploadFile(f),
    onSuccess: (data) => {
      setPreview(data);
      setMapping(data.suggested_mapping);
      setMsg(null);
    },
    onError: (err: Error) => setMsg({ type: "error", text: err.message }),
  });

  const mapColumnsMut = useMutation({
    mutationFn: () => api.mapColumns(preview!.upload_id, mapping),
    onSuccess: (res) => {
      setMsg({
        type: "success",
        text: `Successfully imported ${res.loaded_rows} readings across ${res.pool_count} pools! (${res.skipped_count} skipped)`,
      });
      queryClient.invalidateQueries({ queryKey: ["fleet"] });
      setPreview(null);
    },
    onError: (err: Error) => setMsg({ type: "error", text: err.message }),
  });

  const manualMut = useMutation({
    mutationFn: () =>
      api.addReading({
        pool_id: manualForm.pool_id,
        community_name: manualForm.community_name || undefined,
        reading_date: manualForm.reading_date,
        ph: manualForm.ph ? parseFloat(manualForm.ph) : null,
        free_chlorine: manualForm.free_chlorine ? parseFloat(manualForm.free_chlorine) : null,
        turbidity: manualForm.turbidity ? parseFloat(manualForm.turbidity) : null,
        pool_volume_m3: manualForm.pool_volume_m3 ? parseFloat(manualForm.pool_volume_m3) : null,
        hypochlorite_dosing_pct: manualForm.hypochlorite_dosing_pct ? parseFloat(manualForm.hypochlorite_dosing_pct) : null,
        hypochlorite_dosing_hours: manualForm.hypochlorite_dosing_hours ? parseFloat(manualForm.hypochlorite_dosing_hours) : null,
      }),
    onSuccess: (res) => {
      setMsg({ type: "success", text: `Reading saved for pool '${res.pool_id}'!` });
      queryClient.invalidateQueries({ queryKey: ["fleet"] });
      setManualForm((prev) => ({ ...prev, pool_id: "", community_name: "" }));
    },
    onError: (err: Error) => setMsg({ type: "error", text: err.message }),
  });

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="bg-[#1a1d27] border border-[#2d3141] rounded-2xl w-full max-w-2xl overflow-hidden shadow-2xl animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="p-5 border-b border-[#2d3141] flex justify-between items-center bg-[#21242f]">
          <div>
            <h3 className="text-lg font-bold text-[#e8eaed]">Data Ingestion Studio</h3>
            <p className="text-xs text-[#9aa0a6]">Import technician logs or add single measurements to PostgreSQL</p>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg bg-[#2a2e3b] text-[#9aa0a6] hover:text-white flex items-center justify-center transition cursor-pointer"
          >
            <X size={16} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-[#2d3141] bg-[#141820]">
          <button
            onClick={() => { setTab("upload"); setMsg(null); }}
            className={`flex-1 py-3 text-sm font-semibold transition border-b-2 flex items-center justify-center gap-2 ${
              tab === "upload" ? "border-[#4f8ff7] text-[#4f8ff7] bg-[#1a1d27]" : "border-transparent text-[#9aa0a6] hover:text-white"
            }`}
          >
            <FileSpreadsheet size={16} />
            <span>File Upload (CSV / Excel)</span>
          </button>
          <button
            onClick={() => { setTab("manual"); setMsg(null); }}
            className={`flex-1 py-3 text-sm font-semibold transition border-b-2 flex items-center justify-center gap-2 ${
              tab === "manual" ? "border-[#4f8ff7] text-[#4f8ff7] bg-[#1a1d27]" : "border-transparent text-[#9aa0a6] hover:text-white"
            }`}
          >
            <PenTool size={16} />
            <span>Manual Reading Entry</span>
          </button>
        </div>

        {/* Feedback Alert */}
        {msg && (
          <div
            className={`m-5 p-3 rounded-lg text-sm border flex items-center gap-2 ${
              msg.type === "success"
                ? "bg-green-500/10 border-green-500/30 text-green-400"
                : "bg-red-500/10 border-red-500/30 text-red-400"
            }`}
          >
            {msg.type === "success" ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
            <span>{msg.text}</span>
          </div>
        )}

        <div className="p-6">
          {tab === "upload" ? (
            <div>
              {!preview ? (
                <div className="border-2 border-dashed border-[#2d3141] hover:border-[#4f8ff7] rounded-xl p-8 text-center transition cursor-pointer bg-[#141820]">
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={(e) => {
                      if (e.target.files?.[0]) {
                        uploadFileMut.mutate(e.target.files[0]);
                      }
                    }}
                    className="hidden"
                    id="file-upload-input"
                  />
                  <label htmlFor="file-upload-input" className="cursor-pointer flex flex-col items-center gap-2">
                    <div className="w-12 h-12 rounded-full bg-[#4f8ff7]/10 flex items-center justify-center text-[#4f8ff7]">
                      <UploadCloud size={24} />
                    </div>
                    <div className="text-sm font-semibold text-[#e8eaed]">Click to upload or drag & drop</div>
                    <div className="text-xs text-[#6b7280]">Supports CSV and Excel files (.csv, .xlsx, .xls) up to 15MB</div>
                  </label>
                  {uploadFileMut.isPending && (
                    <div className="mt-4 text-xs text-[#4f8ff7]">Analyzing headers and rows...</div>
                  )}
                </div>
              ) : (
                <div>
                  <div className="flex justify-between items-center mb-4">
                    <div>
                      <span className="text-xs font-semibold text-green-400 uppercase tracking-wider">File Loaded</span>
                      <h4 className="text-sm font-bold text-[#e8eaed]">{preview.filename} ({preview.total_rows} rows)</h4>
                    </div>
                    <button
                      onClick={() => setPreview(null)}
                      className="text-xs text-[#9aa0a6] hover:text-white underline cursor-pointer"
                    >
                      Choose different file
                    </button>
                  </div>

                  {/* Mapping Table */}
                  <div className="bg-[#141820] border border-[#2d3141] rounded-xl p-4 mb-4 max-h-64 overflow-y-auto">
                    <div className="text-xs font-semibold text-[#9aa0a6] uppercase tracking-wider mb-2">Column Mapping</div>
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      {REQUIRED_TARGETS.map((target) => (
                        <div key={target.key} className="flex flex-col gap-1">
                          <label className="text-[#9aa0a6] font-medium">
                            {target.label} {target.required && <span className="text-red-400">*</span>}
                          </label>
                          <select
                            value={mapping[target.key] || ""}
                            onChange={(e) => setMapping((prev) => ({ ...prev, [target.key]: e.target.value }))}
                            className="bg-[#1a1d27] border border-[#2d3141] rounded-lg px-2.5 py-1.5 text-[#e8eaed] outline-none focus:border-[#4f8ff7]"
                          >
                            <option value="">-- Ignore / Not present --</option>
                            {preview.columns.map((c) => (
                              <option key={c} value={c}>
                                {c}
                              </option>
                            ))}
                          </select>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex justify-end gap-3">
                    <button
                      onClick={() => setPreview(null)}
                      className="px-4 py-2 rounded-lg border border-[#2d3141] text-xs font-semibold text-[#9aa0a6] hover:text-white cursor-pointer"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={() => mapColumnsMut.mutate()}
                      disabled={mapColumnsMut.isPending}
                      className="px-5 py-2 rounded-lg bg-[#4f8ff7] hover:bg-[#3d7ae0] text-white text-xs font-semibold shadow-lg transition disabled:opacity-50 cursor-pointer"
                    >
                      {mapColumnsMut.isPending ? "Importing..." : "Confirm & Import Readings"}
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                manualMut.mutate();
              }}
              className="grid grid-cols-2 gap-4 text-xs"
            >
              <div className="col-span-2 sm:col-span-1">
                <label className="block text-[#9aa0a6] font-medium mb-1">Pool ID *</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Cabo Verde (19)"
                  value={manualForm.pool_id}
                  onChange={(e) => setManualForm({ ...manualForm, pool_id: e.target.value })}
                  className="w-full bg-[#141820] border border-[#2d3141] rounded-lg px-3 py-2 text-[#e8eaed] outline-none focus:border-[#4f8ff7]"
                />
              </div>

              <div className="col-span-2 sm:col-span-1">
                <label className="block text-[#9aa0a6] font-medium mb-1">Community / Urbanization</label>
                <input
                  type="text"
                  placeholder="e.g. Cabo Verde"
                  value={manualForm.community_name}
                  onChange={(e) => setManualForm({ ...manualForm, community_name: e.target.value })}
                  className="w-full bg-[#141820] border border-[#2d3141] rounded-lg px-3 py-2 text-[#e8eaed] outline-none focus:border-[#4f8ff7]"
                />
              </div>

              <div className="col-span-2 sm:col-span-1">
                <label className="block text-[#9aa0a6] font-medium mb-1">Reading Date & Time *</label>
                <input
                  type="datetime-local"
                  required
                  value={manualForm.reading_date}
                  onChange={(e) => setManualForm({ ...manualForm, reading_date: e.target.value })}
                  className="w-full bg-[#141820] border border-[#2d3141] rounded-lg px-3 py-2 text-[#e8eaed] outline-none focus:border-[#4f8ff7]"
                />
              </div>

              <div className="col-span-2 sm:col-span-1">
                <label className="block text-[#9aa0a6] font-medium mb-1">Pool Volume (m³)</label>
                <input
                  type="number"
                  step="0.1"
                  value={manualForm.pool_volume_m3}
                  onChange={(e) => setManualForm({ ...manualForm, pool_volume_m3: e.target.value })}
                  className="w-full bg-[#141820] border border-[#2d3141] rounded-lg px-3 py-2 text-[#e8eaed] outline-none focus:border-[#4f8ff7]"
                />
              </div>

              {/* Chemical parameters */}
              <div>
                <label className="block text-[#9aa0a6] font-medium mb-1">Free Chlorine (mg/L)</label>
                <input
                  type="number"
                  step="0.01"
                  value={manualForm.free_chlorine}
                  onChange={(e) => setManualForm({ ...manualForm, free_chlorine: e.target.value })}
                  className="w-full bg-[#141820] border border-[#2d3141] rounded-lg px-3 py-2 text-[#e8eaed] outline-none focus:border-[#4f8ff7]"
                />
                <span className="text-[10px] text-[#6b7280]">RD 742/2013: 0.5–2.0 (Spanish opt: 1.0–2.5)</span>
              </div>

              <div>
                <label className="block text-[#9aa0a6] font-medium mb-1">pH Level</label>
                <input
                  type="number"
                  step="0.01"
                  value={manualForm.ph}
                  onChange={(e) => setManualForm({ ...manualForm, ph: e.target.value })}
                  className="w-full bg-[#141820] border border-[#2d3141] rounded-lg px-3 py-2 text-[#e8eaed] outline-none focus:border-[#4f8ff7]"
                />
                <span className="text-[10px] text-[#6b7280]">RD 742/2013: 7.2–8.0</span>
              </div>

              <div>
                <label className="block text-[#9aa0a6] font-medium mb-1">Turbidity (NTU)</label>
                <input
                  type="number"
                  step="0.01"
                  value={manualForm.turbidity}
                  onChange={(e) => setManualForm({ ...manualForm, turbidity: e.target.value })}
                  className="w-full bg-[#141820] border border-[#2d3141] rounded-lg px-3 py-2 text-[#e8eaed] outline-none focus:border-[#4f8ff7]"
                />
                <span className="text-[10px] text-[#6b7280]">RD 742/2013: ≤ 5.0 (Client: ≤ 1.0)</span>
              </div>

              <div>
                <label className="block text-[#9aa0a6] font-medium mb-1">Hypochlorite Dosing % / Hours</label>
                <div className="flex gap-2">
                  <input
                    type="number"
                    step="1"
                    placeholder="Pump %"
                    value={manualForm.hypochlorite_dosing_pct}
                    onChange={(e) => setManualForm({ ...manualForm, hypochlorite_dosing_pct: e.target.value })}
                    className="w-1/2 bg-[#141820] border border-[#2d3141] rounded-lg px-3 py-2 text-[#e8eaed] outline-none focus:border-[#4f8ff7]"
                  />
                  <input
                    type="number"
                    step="0.5"
                    placeholder="Hours"
                    value={manualForm.hypochlorite_dosing_hours}
                    onChange={(e) => setManualForm({ ...manualForm, hypochlorite_dosing_hours: e.target.value })}
                    className="w-1/2 bg-[#141820] border border-[#2d3141] rounded-lg px-3 py-2 text-[#e8eaed] outline-none focus:border-[#4f8ff7]"
                  />
                </div>
              </div>

              <div className="col-span-2 pt-4 flex justify-end gap-3">
                <button
                  type="button"
                  onClick={onClose}
                  className="px-4 py-2 rounded-lg border border-[#2d3141] text-xs font-semibold text-[#9aa0a6] hover:text-white cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={manualMut.isPending}
                  className="px-5 py-2 rounded-lg bg-[#4f8ff7] hover:bg-[#3d7ae0] text-white text-xs font-semibold shadow-lg transition disabled:opacity-50 cursor-pointer"
                >
                  {manualMut.isPending ? "Saving..." : "Save Reading"}
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
