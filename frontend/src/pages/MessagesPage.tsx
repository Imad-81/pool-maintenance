import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  AlertOctagon,
  AlertTriangle,
  Bell,
  CheckCheck,
  Loader2,
} from "lucide-react";
import { api } from "../api";
import IberHeader from "../components/IberHeader";
import { ChatMessagesIcon } from "../components/Icons";
import { useI18n } from "../i18n";

interface IncidentMessage {
  id: string;
  pool_id?: string;
  title: string;
  content: string;
  time: string;
  level: "critical" | "warning" | "info" | "success";
  read: boolean;
}

export default function MessagesPage() {
  const { t, lang } = useI18n();
  const navigate = useNavigate();
  const [filter, setFilter] = useState<"all" | "critical" | "warning" | "info">("all");
  const [readIds, setReadIds] = useState<Set<string>>(new Set());

  // Fetch real fleet predictions to generate intelligent notifications
  const { data: fleetData, isLoading } = useQuery({
    queryKey: ["fleet"],
    queryFn: () => api.fleet({ page_size: 50 }),
  });

  const generatedMessages: IncidentMessage[] = useMemo(() => {
    if (!fleetData?.items) return [];

    const msgs: IncidentMessage[] = [];

    // Filter pools with critical urgency
    fleetData.items.forEach((p, idx) => {
      if (p.urgency === "Immediate" || p.urgency === "URGENT") {
        msgs.push({
          id: `crit-${p.pool_id}-${idx}`,
          pool_id: p.pool_id,
          title: `${t("urg_immediate")}: ${p.community_name || p.pool_id}`,
          content: `Cloro: ${p.free_chlorine?.toFixed(2) || "0.4"} mg/L, pH: ${p.ph?.toFixed(2) || "8.1"}. ${p.urgency}`,
          time: "10:30",
          level: "critical",
          read: false,
        });
      } else if (p.urgency === "Advised" || p.urgency === "Soon" || p.urgency === "Monitor") {
        msgs.push({
          id: `warn-${p.pool_id}-${idx}`,
          pool_id: p.pool_id,
          title: `${t("urg_advised")}: ${p.community_name || p.pool_id}`,
          content: `Cloro: ${p.free_chlorine?.toFixed(2) || "1.1"} mg/L.`,
          time: "09:15",
          level: "warning",
          read: false,
        });
      }
    });

    // Add general routine summary messages
    msgs.push({
      id: "info-weather-sync",
      title: t("messages_wx_sync_title"),
      content: t("messages_wx_sync_body"),
      time: "08:00",
      level: "info",
      read: true,
    });

    msgs.push({
      id: "info-model-ready",
      title: t("messages_model_ready_title"),
      content: t("messages_model_ready_body"),
      time: lang === "en" ? "Yesterday" : "Ayer",
      level: "info",
      read: true,
    });

    return msgs;
  }, [fleetData, lang, t]);

  const filteredMessages = useMemo(() => {
    if (filter === "all") return generatedMessages;
    if (filter === "critical") return generatedMessages.filter((m) => m.level === "critical");
    if (filter === "warning") return generatedMessages.filter((m) => m.level === "warning");
    return generatedMessages.filter((m) => m.level === "info" || m.level === "success");
  }, [generatedMessages, filter]);

  const toggleRead = (id: string) => {
    setReadIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const markAllAsRead = () => {
    const all = new Set(generatedMessages.map((m) => m.id));
    setReadIds(all);
  };

  return (
    <div className="min-h-screen bg-caustic text-white flex flex-col">
      <IberHeader subtitle={t("messages_subtitle")} />

      <main className="flex-1 max-w-[1400px] w-full mx-auto px-4 md:px-8 pb-16">
        {/* Navigation & Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate("/")}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl glass-card text-blue-200 hover:text-white text-xs font-semibold cursor-pointer"
            >
              <ArrowLeft size={14} />
              <span>{t("backToMenu")}</span>
            </button>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white">
                <ChatMessagesIcon size={20} />
              </div>
              <h2 className="text-xl md:text-2xl font-bold tracking-tight font-heading">
                {t("messages_title")}
              </h2>
            </div>
          </div>

          <button
            onClick={markAllAsRead}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl glass-card text-xs text-blue-200 hover:text-white font-semibold transition self-start sm:self-auto cursor-pointer"
          >
            <CheckCheck size={14} />
            <span>{t("messages_mark_all_read")}</span>
          </button>
        </div>

        {/* Filter Tabs */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
          <FilterTab
            label={t("messages_tab_all")}
            count={generatedMessages.length}
            active={filter === "all"}
            color="text-white"
            onClick={() => setFilter("all")}
          />
          <FilterTab
            label={t("messages_tab_critical")}
            count={generatedMessages.filter((m) => m.level === "critical").length}
            active={filter === "critical"}
            color="text-red-400"
            onClick={() => setFilter("critical")}
          />
          <FilterTab
            label={t("messages_tab_warnings")}
            count={generatedMessages.filter((m) => m.level === "warning").length}
            active={filter === "warning"}
            color="text-amber-400"
            onClick={() => setFilter("warning")}
          />
          <FilterTab
            label={t("messages_tab_info")}
            count={generatedMessages.filter((m) => m.level === "info" || m.level === "success").length}
            active={filter === "info"}
            color="text-cyan-300"
            onClick={() => setFilter("info")}
          />
        </div>

        {/* Message List */}
        {isLoading ? (
          <div className="glass-panel rounded-2xl p-12 text-center text-blue-200">
            <Loader2 size={28} className="animate-spin text-cyan-400 mx-auto mb-2" />
            <p>{t("messages_loading")}</p>
          </div>
        ) : filteredMessages.length === 0 ? (
          <div className="glass-panel rounded-2xl p-12 text-center text-blue-300/70">
            {t("messages_empty")}
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
                      {msg.level === "critical" ? (
                        <AlertOctagon size={18} className="text-red-400 shrink-0" />
                      ) : msg.level === "warning" ? (
                        <AlertTriangle size={18} className="text-amber-400 shrink-0" />
                      ) : (
                        <Bell size={18} className="text-blue-400 shrink-0" />
                      )}
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
                          className="text-cyan-300 hover:underline font-semibold cursor-pointer"
                        >
                          {t("messages_view_pool")}
                        </button>
                      )}
                    </div>
                    <button
                      onClick={() => toggleRead(msg.id)}
                      className="text-blue-300/70 hover:text-white text-[11px] cursor-pointer"
                    >
                      {isRead ? t("messages_mark_unread") : t("messages_mark_read")}
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
  color: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`glass-card rounded-xl p-3 text-left transition-all cursor-pointer ${
        active ? "ring-2 ring-blue-400 bg-blue-600/30" : "hover:bg-blue-600/15"
      }`}
    >
      <div className="text-[11px] font-medium text-blue-200/70">{label}</div>
      <div className={`text-xl font-bold font-heading ${color}`}>{count}</div>
    </button>
  );
}
