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
      <div className={`inline-flex items-center gap-1 bg-blue-950/80 p-1 rounded-xl border border-blue-800/40 ${className}`}>
        <button
          type="button"
          onClick={() => handleSelect("es")}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
            lang === "es"
              ? "bg-blue-600 text-white shadow-md"
              : "text-blue-300 hover:text-white"
          }`}
        >
          🇪🇸 Español
        </button>
        <button
          type="button"
          onClick={() => handleSelect("en")}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
            lang === "en"
              ? "bg-blue-600 text-white shadow-md"
              : "text-blue-300 hover:text-white"
          }`}
        >
          🇬🇧 English
        </button>
      </div>
    );
  }

  return (
    <div
      className={`flex items-center bg-blue-950/60 backdrop-blur-md rounded-full p-0.5 border border-blue-400/30 shadow-sm ${className}`}
      role="group"
      aria-label="Language selector"
    >
      <button
        type="button"
        onClick={() => handleSelect("es")}
        className={`px-2.5 py-1 rounded-full text-xs font-bold transition-all duration-150 ${
          lang === "es"
            ? "bg-blue-500 text-white shadow-sm"
            : "text-blue-200/75 hover:text-white hover:bg-blue-800/30"
        }`}
        title="Cambiar a Español"
      >
        ES
      </button>
      <button
        type="button"
        onClick={() => handleSelect("en")}
        className={`px-2.5 py-1 rounded-full text-xs font-bold transition-all duration-150 ${
          lang === "en"
            ? "bg-blue-500 text-white shadow-sm"
            : "text-blue-200/75 hover:text-white hover:bg-blue-800/30"
        }`}
        title="Switch to English"
      >
        EN
      </button>
    </div>
  );
}
