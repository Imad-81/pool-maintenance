import { Link } from "react-router-dom";
import {
  PoolLadderIcon,
  AccountLockIcon,
  ChatMessagesIcon,
  AnalyticsFlaskIcon,
  SkimmerNetIcon,
  ToolsIncidentIcon,
} from "./Icons";

interface HubMenuProps {
  urgentCount?: number;
  totalPools?: number;
}

export default function HubMenu({ urgentCount = 0, totalPools = 0 }: HubMenuProps) {
  const menuItems = [
    {
      id: "piscinas",
      title: "Mis piscinas",
      to: "/piscinas",
      icon: PoolLadderIcon,
      badge: totalPools > 0 ? `${totalPools}` : undefined,
      urgentBadge: urgentCount > 0 ? `${urgentCount}` : undefined,
    },
    {
      id: "cuenta",
      title: "Mi cuenta",
      to: "/cuenta",
      icon: AccountLockIcon,
    },
    {
      id: "mensajes",
      title: "Mensajes",
      to: "/mensajes",
      icon: ChatMessagesIcon,
      urgentBadge: urgentCount > 0 ? `${urgentCount}` : undefined,
    },
    {
      id: "analiticas",
      title: "Analíticas",
      to: "/analiticas",
      icon: AnalyticsFlaskIcon,
    },
    {
      id: "limpiezas",
      title: "Limpiezas",
      to: "/limpiezas",
      icon: SkimmerNetIcon,
    },
    {
      id: "incidencias",
      title: "Incidencias",
      to: "/incidencias",
      icon: ToolsIncidentIcon,
      urgentBadge: urgentCount > 0 ? `${urgentCount}` : undefined,
    },
  ];

  return (
    <div className="w-full max-w-[540px] mx-auto px-4 py-2">
      <div className="grid grid-cols-3 gap-3 md:gap-4">
        {menuItems.map((item) => {
          const IconComp = item.icon;
          return (
            <Link
              key={item.id}
              to={item.to}
              className="iber-tile relative group flex flex-col items-center justify-center rounded-2xl aspect-square p-2.5 sm:p-4 text-center cursor-pointer transition-all focus:outline-none focus:ring-2 focus:ring-cyan-300"
            >
              {/* Optional Urgent/Counter Badge */}
              {item.urgentBadge && (
                <span className="absolute top-2 right-2 flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-[10px] font-bold text-white shadow-md animate-pulse">
                  {item.urgentBadge}
                </span>
              )}

              {/* Icon Container */}
              <div className="flex items-center justify-center mb-2 sm:mb-3 text-white group-hover:scale-110 transition-transform duration-200">
                <IconComp size={42} className="w-8 h-8 sm:w-11 sm:h-11 drop-shadow" />
              </div>

              {/* Label */}
              <span className="text-xs sm:text-sm font-semibold tracking-wide text-white leading-tight drop-shadow-sm font-heading">
                {item.title}
              </span>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
