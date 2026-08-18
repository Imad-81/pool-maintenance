import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import IberHeader from "../components/IberHeader";
import { ChatMessagesIcon } from "../components/Icons";

interface MessageItem {
  id: string;
  pool_id?: string;
  title: string;
  content: string;
  time: string;
  level: "critical" | "warning" | "info" | "success";
  read: boolean;
}

export default function MessagesPage() {
  const navigate = useNavigate();
  const [filter, setFilter] = useState<"all" | "critical" | "warning" | "info">("all");
  const [readIds, setReadIds] = useState<Set<string>>(new Set());

  const { data: fleetData, isLoading } = useQuery({
    queryKey: ["fleet", "messages-data"],
    queryFn: () => api.fleet({ page_size: 100 }),
  });

  const generatedMessages: MessageItem[] = useMemo(() => {
    const items = fleetData?.items || [];
    const msgs: MessageItem[] = [];

    // Urgent / Immediate breach messages
    items
      .filter((p) => p.urgency === "Immediate" || p.urgency === "URGENT")
      .forEach((p, idx) => {
        msgs.push({
          id: `crit-${p.pool_id}-${idx}`,
          pool_id: p.pool_id,
          title: `Alerta Sanitaria Urgente: ${p.community_name || p.pool_id}`,
          content: `El sistema predictivo detectó parámetros fuera de rango RD 742/2013 (Cloro: ${
            p.free_chlorine?.toFixed(2) ?? "N/D"
          } mg/L, pH: ${p.ph?.toFixed(2) ?? "N/D"}). Se requiere visita prioritaria y ajuste de dosificación.`,
          time: "Hace 15 minutos",
          level: "critical",
          read: false,
        });
      });

    // Warning / Advised messages
    items
      .filter((p) => p.urgency === "Advised" || p.urgency === "Soon" || (p.breach_proba || 0) > 0.4)
      .forEach((p, idx) => {
        msgs.push({
          id: `warn-${p.pool_id}-${idx}`,
          pool_id: p.pool_id,
          title: `Aviso Predictivo: ${p.community_name || p.pool_id}`,
          content: `Probabilidad del ${Math.round(
            (p.breach_proba || 0) * 100
          )}% de descenso de cloro libre en las próximas 24-48h por radiación solar alta en Alicante.`,
          time: "Hace 2 horas",
          level: "warning",
          read: false,
        });
      });

    // Routine communications
    msgs.push({
      id: "info-weather-sync",
      title: "Actualización Meteorológica Completada",
      content:
        "Se han sincronizado las previsiones de radiación UV y temperatura de Open-Meteo para todas las instalaciones de la provincia de Alicante.",
      time: "Hoy, 08:30",
      level: "info",
      read: false,
    });

    msgs.push({
      id: "info-model-ready",
      title: "Motor Chained Physics-ML Operativo",
      content:
        "Calibración de modelos de decaimiento químico lista para la temporada estival de piscinas comunitarias.",
      time: "Ayer",
      level: "success",
      read: true,
    });

    return msgs;
  }, [fleetData]);

  const filteredMessages = generatedMessages.filter((m) => {
    if (filter === "all") return true;
    return m.level === filter;
  });

  const markAllRead = () => {
    const all = new Set(generatedMessages.map((m) => m.id));
    setReadIds(all);
  };

  const toggleRead = (id: string) => {
    setReadIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="min-h-screen bg-caustic text-white flex flex-col">
      <IberHeader subtitle="Mensajes — Centro de alertas y notificaciones" />

      <main className="flex-1 max-w-[1100px] w-full mx-auto px-4 md:px-8 pb-16">
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
                <ChatMessagesIcon size={20} />
              </div>
              <h2 className="text-xl md:text-2xl font-bold tracking-tight font-heading">
                Mensajes & Avisos
              </h2>
            </div>
          </div>

          <button
            onClick={markAllRead}
            className="px-3 py-1.5 rounded-xl glass-card text-blue-300 hover:text-white text-xs font-medium self-start sm:self-auto"
          >
            ✓ Marcar todos como leídos
          </button>
        </div>

        {/* Filter Tabs */}
        <div className="flex gap-2 mb-6 overflow-x-auto pb-1">
          <FilterTab
            label="Todos"
            count={generatedMessages.length}
            active={filter === "all"}
            onClick={() => setFilter("all")}
          />
          <FilterTab
            label="Críticos"
            count={generatedMessages.filter((m) => m.level === "critical").length}
            active={filter === "critical"}
            color="text-red-400"
            onClick={() => setFilter("critical")}
          />
          <FilterTab
            label="Advertencias"
            count={generatedMessages.filter((m) => m.level === "warning").length}
            active={filter === "warning"}
            color="text-amber-400"
            onClick={() => setFilter("warning")}
          />
          <FilterTab
            label="Informativos"
            count={generatedMessages.filter((m) => m.level === "info" || m.level === "success").length}
            active={filter === "info"}
            onClick={() => setFilter("info")}
          />
        </div>

        {/* Message List */}
        {isLoading ? (
          <div className="glass-panel rounded-2xl p-12 text-center text-blue-200">
            <div className="animate-spin text-2xl mb-2">💬</div>
            Cargando notificaciones...
          </div>
        ) : filteredMessages.length === 0 ? (
          <div className="glass-panel rounded-2xl p-12 text-center text-blue-300/70">
            No hay mensajes en esta categoría.
          </div>
        ) : (
          <div className="space-y-3">
            {filteredMessages.map((msg) => {
              const isRead = readIds.has(msg.id) || msg.read;
              const borderStyles =
                msg.level === "critical"
                  ? "border-red-500/50 bg-red-950/20"
                  : msg.level === "warning"
                  ? "border-amber-500/40 bg-amber-950/20"
                  : "border-blue-800/40 bg-blue-950/30";

              return (
                <div
                  key={msg.id}
                  className={`glass-card rounded-2xl p-5 border transition-all ${borderStyles} ${
                    isRead ? "opacity-75" : ""
                  }`}
                >
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">
                        {msg.level === "critical" ? "🚨" : msg.level === "warning" ? "⚠️" : "📢"}
                      </span>
                      <h4 className="font-bold text-white text-sm sm:text-base font-heading">
                        {msg.title}
                      </h4>
                    </div>
                    <span className="text-[11px] text-blue-300/60 font-mono whitespace-nowrap">
                      {msg.time}
                    </span>
                  </div>

                  <p className="text-xs sm:text-sm text-blue-100/90 leading-relaxed mb-3">
                    {msg.content}
                  </p>

                  <div className="flex items-center justify-between pt-2 border-t border-blue-900/40 text-xs">
                    <div className="flex items-center gap-3">
                      {msg.pool_id && (
                        <button
                          onClick={() => navigate(`/piscinas/${msg.pool_id}`)}
                          className="text-cyan-300 hover:underline font-semibold"
                        >
                          Ver Diagnóstico de Piscina →
                        </button>
                      )}
                    </div>
                    <button
                      onClick={() => toggleRead(msg.id)}
                      className="text-blue-300/70 hover:text-white text-[11px]"
                    >
                      {isRead ? "Marcar como no leído" : "Marcar como leído"}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}

function FilterTab({
  label,
  count,
  active,
  color,
  onClick,
}: {
  label: string;
  count: number;
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
      {label} <span className={color || "text-blue-300"}>({count})</span>
    </button>
  );
}
