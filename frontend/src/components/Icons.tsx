import React from "react";

interface IconProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  className?: string;
}

// 1. Mis piscinas: Swimming pool ladder & water ripples
export function PoolLadderIcon({ size = 44, className = "text-white", ...props }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      {...props}
    >
      {/* Pool Handrails */}
      <path d="M14 10 V 26" />
      <path d="M22 10 V 26" />
      <path d="M14 10 C 14 6, 18 6, 18 10" />
      <path d="M22 10 C 22 6, 26 6, 26 10" />
      <path d="M26 10 V 26" />

      {/* Ladder Rungs */}
      <line x1="14" y1="15" x2="26" y2="15" />
      <line x1="14" y1="20" x2="26" y2="20" />
      <line x1="14" y1="25" x2="26" y2="25" />

      {/* Pool Deck Edge */}
      <line x1="8" y1="26" x2="38" y2="26" strokeWidth="2.5" />

      {/* Water Waves */}
      <path d="M8 32 C 12 30, 15 34, 19 32 C 23 30, 26 34, 30 32 C 34 30, 37 34, 40 32" strokeWidth="1.8" />
      <path d="M10 38 C 14 36, 17 40, 21 38 C 25 36, 28 40, 32 38 C 36 36, 39 40, 42 38" strokeWidth="1.8" />
    </svg>
  );
}

// 2. Mi cuenta: Lock / Shield Badge
export function AccountLockIcon({ size = 44, className = "text-white", ...props }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      {...props}
    >
      {/* Padlock Shackle */}
      <path d="M17 21 V 15 C 17 11.134 20.134 8 24 8 C 27.866 8 31 11.134 31 15 V 21" />
      {/* Padlock Body */}
      <rect x="13" y="21" width="22" height="18" rx="4" fill="currentColor" fillOpacity="0.05" />
      {/* Shackle Stripes / Detail matching image */}
      <line x1="13" y1="27" x2="35" y2="27" strokeWidth="1.8" />
      <line x1="16" y1="27" x2="20" y2="33" strokeWidth="1.8" />
      <line x1="22" y1="27" x2="26" y2="33" strokeWidth="1.8" />
      <line x1="28" y1="27" x2="32" y2="33" strokeWidth="1.8" />
      {/* Keyhole */}
      <circle cx="24" cy="35" r="1.5" fill="currentColor" />
    </svg>
  );
}

// 3. Mensajes: Two Chat / Speech Bubbles
export function ChatMessagesIcon({ size = 44, className = "text-white", ...props }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      {...props}
    >
      {/* Main Bubble */}
      <path d="M25 11 C 16.716 11 10 16.82 10 24 C 10 27.054 11.21 29.852 13.25 32.05 L 12 38 L 18.25 36.2 C 20.3 36.72 22.58 37 25 37 C 33.284 37 40 31.18 40 24 C 40 16.82 33.284 11 25 11 Z" />
      {/* Chat Dots */}
      <circle cx="19" cy="24" r="1.8" fill="currentColor" />
      <circle cx="25" cy="24" r="1.8" fill="currentColor" />
      <circle cx="31" cy="24" r="1.8" fill="currentColor" />
    </svg>
  );
}

// 4. Analíticas: Laboratory Chemistry Beaker / Flask with Bubbles
export function AnalyticsFlaskIcon({ size = 44, className = "text-white", ...props }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      {...props}
    >
      {/* Flask Neck & Body */}
      <path d="M20 8 H 28" />
      <path d="M21 8 V 16 L 12 34 C 10.5 37 12.5 40 16 40 H 32 C 35.5 40 37.5 37 36 34 L 27 16 V 8" />
      {/* Liquid level */}
      <path d="M15.5 29 C 19 31, 23 27, 27 29 C 30 30.5, 32.5 29, 32.5 29" strokeWidth="1.8" />
      {/* Effervescent bubbles */}
      <circle cx="21" cy="34" r="1.5" fill="currentColor" />
      <circle cx="27" cy="33" r="1.5" fill="currentColor" />
      <circle cx="24" cy="24" r="1" fill="currentColor" />
      <circle cx="17" cy="18" r="1.2" fill="currentColor" />
      <circle cx="31" cy="20" r="1.2" fill="currentColor" />
    </svg>
  );
}

// 5. Limpiezas: Pool Skimmer Net / Cleaning
export function SkimmerNetIcon({ size = 44, className = "text-white", ...props }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      {...props}
    >
      {/* Telescopic Pole */}
      <line x1="33" y1="10" x2="25" y2="24" strokeWidth="2.5" />
      {/* Skimmer Frame */}
      <rect
        x="15"
        y="22"
        width="18"
        height="14"
        rx="2"
        transform="rotate(-15 24 29)"
        fill="currentColor"
        fillOpacity="0.05"
      />
      {/* Net Mesh lines */}
      <line x1="18" y1="25" x2="30" y2="28" strokeWidth="1.5" strokeDasharray="1 2" />
      <line x1="16" y1="31" x2="28" y2="34" strokeWidth="1.5" strokeDasharray="1 2" />
      {/* Water Surface Splash / Wave */}
      <path d="M8 32 C 12 30, 15 34, 18 32" strokeWidth="1.8" />
      <path d="M30 34 C 33 32, 36 36, 40 33" strokeWidth="1.8" />
      {/* Water bubbles */}
      <circle cx="12" cy="28" r="1" fill="currentColor" />
      <circle cx="36" cy="29" r="1" fill="currentColor" />
    </svg>
  );
}

// 6. Incidencias: Crossed Wrench and Screwdriver
export function ToolsIncidentIcon({ size = 44, className = "text-white", ...props }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      {...props}
    >
      {/* Screwdriver (diagonal top-right to bottom-left) */}
      <path d="M34 11 L 37 14 L 25 26 L 22 23 Z" />
      <line x1="22" y1="23" x2="13" y2="32" strokeWidth="2.4" />
      <path d="M13 32 L 10 35 C 9 36, 9 38, 10 39 C 11 40, 13 40, 14 39 L 17 36 Z" fill="currentColor" fillOpacity="0.1" />

      {/* Wrench (diagonal top-left to bottom-right) */}
      <path d="M12 16 C 13.5 13 17 11.5 20 13 L 17 16 L 18 19 L 21 18 L 24 21 C 25.5 24 24 27.5 21 29 L 34 42 C 35.5 43.5 38 43.5 39.5 42 C 41 40.5 41 38 39.5 36.5 L 26.5 23.5" />
    </svg>
  );
}

// Circular Close Button (X)
export function CloseCircleIcon({ size = 38, className = "text-white", ...props }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 40 40"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      {...props}
    >
      <circle cx="20" cy="20" r="17" strokeWidth="2" />
      <line x1="14" y1="14" x2="26" y2="26" strokeWidth="2.5" />
      <line x1="26" y1="14" x2="14" y2="26" strokeWidth="2.5" />
    </svg>
  );
}
