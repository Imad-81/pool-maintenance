import { Link, useLocation } from "react-router-dom";
import { BarChart3, Waves } from "lucide-react";
import LanguageSwitcher from "./LanguageSwitcher";
import { useI18n } from "../i18n";

interface IberHeaderProps {
  subtitle?: string;
}

export default function IberHeader({ subtitle }: IberHeaderProps) {
  const location = useLocation();
  const { t } = useI18n();

  const isAnalytics =
    location.pathname === "/analiticas" ||
    location.pathname === "/analytics" ||
    location.pathname === "/admin";

  return (
    <header className="relative w-full pt-5 pb-3 px-5 md:px-10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 z-20">
      <div className="flex flex-col">
        <Link to="/" className="group inline-flex flex-col">
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-white font-heading drop-shadow-md group-hover:text-blue-100 transition">
            {t("brandName")}
          </h1>
          <p className="text-sm md:text-base text-blue-100/90 font-light tracking-wide mt-0.5 drop-shadow">
            {subtitle || t("brandSubtitle")}
          </p>
        </Link>
      </div>

      <div className="flex items-center gap-3 self-end sm:self-center">
        {/* Top Header Navigation: My Pools & Analytics */}
        <div className="flex items-center gap-1.5 bg-blue-950/50 p-1 rounded-xl border border-blue-800/40 backdrop-blur-md">
          <Link
            to="/"
            className={`flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              !isAnalytics
                ? "bg-blue-600 text-white shadow-md font-bold"
                : "text-blue-200 hover:text-white hover:bg-blue-800/30"
            }`}
          >
            <Waves size={15} />
            <span>{t("hub_pools")}</span>
          </Link>
          <Link
            to="/analiticas"
            className={`flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              isAnalytics
                ? "bg-blue-600 text-white shadow-md font-bold"
                : "text-blue-200 hover:text-white hover:bg-blue-800/30"
            }`}
          >
            <BarChart3 size={15} />
            <span>{t("hub_analytics")}</span>
          </Link>
        </div>

        {/* Language Switcher */}
        <LanguageSwitcher />
      </div>
    </header>
  );
}
