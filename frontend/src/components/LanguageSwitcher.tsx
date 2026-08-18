import { useI18n, type Language } from "../i18n";

interface LanguageSwitcherProps {
  className?: string;
  variant?: "header" | "inline";
}

export default function LanguageSwitcher({ className = "", variant = "header" }: LanguageSwitcherProps) {
  const { lang, setLang } = useI18n();

  const handleSelect = (selectedLang: Language) => {
    setLang(selectedLang);
  };

  if (variant === "inline") {
    return (
      <div className={`inline-flex items-center gap-1.5 bg-blue-950/90 p-1.5 rounded-2xl border border-blue-600/40 shadow-md ${className}`}>
        <button
          type="button"
          onClick={() => handleSelect("en")}
          className={`flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-bold transition-all ${
            lang === "en"
              ? "bg-blue-600 text-white shadow-md ring-1 ring-white/30"
              : "text-blue-200/80 hover:text-white hover:bg-blue-800/30"
          }`}
        >
          <span className="text-base leading-none">🇬🇧</span>
          <span>English</span>
        </button>
        <button
          type="button"
          onClick={() => handleSelect("es")}
          className={`flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-bold transition-all ${
            lang === "es"
              ? "bg-blue-600 text-white shadow-md ring-1 ring-white/30"
              : "text-blue-200/80 hover:text-white hover:bg-blue-800/30"
          }`}
        >
          <span className="text-base leading-none">🇪🇸</span>
          <span>Español</span>
        </button>
      </div>
    );
  }

  return (
    <div
      className={`flex items-center bg-[#061f42]/85 backdrop-blur-md rounded-full p-1 border border-cyan-400/40 shadow-lg ${className}`}
      role="group"
      aria-label="Language selector"
    >
      <button
        type="button"
        onClick={() => handleSelect("en")}
        className={`flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-extrabold tracking-wide transition-all duration-200 ${
          lang === "en"
            ? "bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-md ring-1 ring-cyan-300/40"
            : "text-blue-200/70 hover:text-white hover:bg-white/10"
        }`}
        title="Switch to English"
      >
        <span className="text-xs leading-none">🇬🇧</span>
        <span>EN</span>
      </button>
      <button
        type="button"
        onClick={() => handleSelect("es")}
        className={`flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-extrabold tracking-wide transition-all duration-200 ${
          lang === "es"
            ? "bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-md ring-1 ring-cyan-300/40"
            : "text-blue-200/70 hover:text-white hover:bg-white/10"
        }`}
        title="Cambiar a Español"
      >
        <span className="text-xs leading-none">🇪🇸</span>
        <span>ES</span>
      </button>
    </div>
  );
}
