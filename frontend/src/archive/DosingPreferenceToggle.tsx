import { useI18n } from "../i18n";

/**
 * Archived Dosing Preference Option Component.
 * Originally displayed in AccountPage.tsx preferences section.
 */
export default function DosingPreferenceToggle() {
  const { t } = useI18n();

  return (
    <div className="flex items-center justify-between py-2 border-b border-blue-800/30">
      <div>
        <strong className="text-sm text-white block">{t("account_pref_dosing")}</strong>
        <span className="text-xs text-blue-300/70">
          {t("account_pref_dosing_desc")}
        </span>
      </div>
      <input type="checkbox" defaultChecked className="w-5 h-5 accent-blue-500 cursor-pointer" />
    </div>
  );
}
