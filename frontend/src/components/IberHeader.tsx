import { Link, useLocation, useNavigate } from "react-router-dom";
import { CloseCircleIcon } from "./Icons";
import LanguageSwitcher from "./LanguageSwitcher";
import { useI18n } from "../i18n";

interface IberHeaderProps {
  showClose?: boolean;
  onClose?: () => void;
  subtitle?: string;
}

export default function IberHeader({ showClose = true, onClose, subtitle }: IberHeaderProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const { t } = useI18n();
  const isHome = location.pathname === "/";

  const handleClose = () => {
    if (onClose) {
      onClose();
    } else if (!isHome) {
      navigate("/");
    }
  };

  return (
    <header className="relative w-full pt-5 pb-3 px-5 md:px-10 flex items-start justify-between z-20">
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

      <div className="flex items-center gap-3">
        {/* Language Switcher */}
        <LanguageSwitcher />

        {/* Circular Exit/Back Button */}
        {showClose && (
          <button
            onClick={handleClose}
            type="button"
            aria-label={isHome ? t("brandName") : t("closeMenuTitle")}
            className="p-1 rounded-full text-white/90 hover:text-white hover:bg-white/10 active:scale-95 transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-blue-400"
            title={isHome ? t("brandName") : t("closeMenuTitle")}
          >
            <CloseCircleIcon size={38} className="drop-shadow-md" />
          </button>
        )}
      </div>
    </header>
  );
}
