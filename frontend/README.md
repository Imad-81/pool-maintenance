# Frontend — Spain Pool Predictive Maintenance & Operations Dashboard (V6.0)

[![React 19](https://img.shields.io/badge/React-19.x-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Vite](https://img.shields.io/badge/Vite-6.x-646CFF?logo=vite&logoColor=white)](https://vitejs.dev)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.x-FF6384?logo=chartdotjs&logoColor=white)](https://www.chartjs.org)

An industrial-grade, multi-language operations dashboard built with **React 19**, **TypeScript**, and **Vite** for the Spain (Alicante) collective-use swimming pool predictive maintenance ecosystem.

---

## 1. Key Features

- **Fleet Command Center (`FleetPage.tsx`)**:
  - Real-time fleet overview of 135+ community pools with urgency-sorted scorecards.
  - High-visibility **Today and Tomorrow 3-parameter forecast chips** (Free Chlorine, pH, Turbidity) with explicit date labels.
  - **Intelligent Visit Recommender integration**: Displays recommended visit dates, days remaining, urgency badges (🚨 Urgent, ⚠️ Advised, ℹ️ Monitor, ✅ Routine), and risk scores.
  - Interactive search, urgency filtering, pagination, and a manual **Trigger Inference** button.

- **Pool Analytics & Deep-Dive (`PoolDetailPage.tsx`)**:
  - Interactive **Chart.js** historical time-series with color-coded **RD 742/2013 Spanish regulatory bands**.
  - **Chained Multi-Step Forecast Table**: Step-by-step rolling predictions from last technician visit to tomorrow ($T+1$) with weather injection.
  - **Visit Cadence & Chemistry Status**: Clear degradation timeline and next-visit recommendation.
  - Pool physical metadata display (volume $m^3$, surface area $m^2$, filter diameter, motor count).

- **Data Ingestion Studio (`IngestModal.tsx`)**:
  - In-browser CSV and Excel file upload with **fuzzy Spanish column mapping**.
  - Real-time preflight validation, row count verification, data preview, and error reporting.
  - Direct manual reading ingestion modal for quick on-site field logging ($<100\text{ ms}$ recomputation).

- **Operations & Facility Management**:
  - **Incidents Hub (`IncidentsPage.tsx`)**: Track, filter, and resolve pool chemical, mechanical, and safety incidents with priority levels.
  - **Cleaning & Maintenance Logs (`CleaningPage.tsx`)**: Maintenance task logging, filtration cycle tracking, and pump runtime monitoring.
  - **Technician Messaging Hub (`MessagesPage.tsx`)**: Facility alerts, automated notifications, and technician-to-dispatch communications.
  - **Fleet Analytics Hub (`AnalyticsPage.tsx`)**: High-level compliance statistics, breach distributions, and regional climate correlation charts.
  - **Account & Settings (`AccountPage.tsx`)**: Technician profile, regional preferences, and active session details.
  - **Admin Control Center (`AdminPage.tsx`)**: Model status, XGBoost performance metrics ($R^2$, MAE, RMSE), one-click model retraining trigger, and Open-Meteo live weather refresh.

- **Internationalization (i18n)**:
  - Built-in lightweight, zero-dependency translation engine (`src/i18n.ts`).
  - Full **English (`en`)** and **Spanish (`es`)** translations covering all UI labels, regulatory alerts, tooltips, and metrics.
  - Language switcher (`LanguageSwitcher.tsx`) with localStorage persistence and default English support.

- **Design System & Visual Aesthetics**:
  - Bespoke Lucide-style SVG icon system (`src/components/Icons.tsx`) replacing legacy emojis.
  - Professional Mediterranean ocean-and-slate design system with glassmorphism, responsive grids, and micro-animations in `src/index.css`.
  - Unified navigation bar (`IberHeader.tsx`) and application hub launcher (`HubMenu.tsx`).

---

## 2. Directory Structure

```
frontend/
├── index.html                   # HTML5 shell with Google Fonts & metadata
├── package.json                 # React 19, TypeScript, Chart.js, Vite dependencies
├── vite.config.ts               # Vite configuration (port 5173 / proxy 8000)
├── tsconfig.json                # Strict TypeScript configuration
│
└── src/
    ├── main.tsx                 # React application entry point
    ├── App.tsx                  # Root layout, navigation routing, and tab state
    ├── api.ts                   # Axios HTTP client connecting to FastAPI backend
    ├── types.ts                 # Full TypeScript interfaces for API payloads
    ├── i18n.ts                  # Bilingual translation dictionary (ES / EN)
    ├── index.css                # Master design tokens, responsive grid, animations
    │
    ├── components/              # Shared UI components
    │   ├── IberHeader.tsx       # Top navigation bar with logo, language, and status
    │   ├── HubMenu.tsx          # Quick-launch drawer for all operational hubs
    │   ├── IngestModal.tsx      # Interactive Data Ingestion Studio modal
    │   ├── LanguageSwitcher.tsx # Spanish / English language toggle
    │   └── Icons.tsx            # Handcrafted Lucide-style SVG icons
    │
    ├── pages/                   # Main application views
    │   ├── HomePage.tsx         # Welcome splash & operational summary
    │   ├── FleetPage.tsx        # Fleet monitoring with Today/Tomorrow & Visit chips
    │   ├── PoolDetailPage.tsx   # Detailed pool analytics, charts, chained forecasts
    │   ├── IncidentsPage.tsx    # Facility incident tracker and management
    │   ├── CleaningPage.tsx     # Cleaning logs, filter maintenance, and schedules
    │   ├── MessagesPage.tsx     # Dispatch & technician communication center
    │   ├── AnalyticsPage.tsx    # Fleet-wide compliance analytics & distributions
    │   ├── AccountPage.tsx      # Technician profile and settings
    │   └── AdminPage.tsx        # Model retraining, weather monitor, health status
    │
    └── archive/                 # Archived components for reference / modular restore
        ├── README.md            # Restoration guide for archived components
        ├── DosingOptimizerCard.tsx
        ├── DosingSimulator.tsx
        ├── DosingPreferenceToggle.tsx
        └── index.ts
```

---

## 3. Getting Started

### Prerequisites
- **Node.js**: `v18.x` or newer (Node 20+ recommended)
- **npm**: `v9.x` or newer

### Installation
```bash
cd frontend
npm install
```

### Local Development Server
```bash
npm run dev
```
The application will launch at **`http://localhost:5173`** (or proxy requests to backend at `http://localhost:8000`).

### Production Build
```bash
npm run build
```
Generates an optimized production bundle in `dist/`.

### Preview Production Build
```bash
npm run preview
```

---

## 4. API Communication

The frontend interacts with the FastAPI backend via `src/api.ts`. In local development, Vite proxies `/api` calls directly to `http://localhost:8000`.

Key API methods:
- `fetchFleet(search, urgency, page, limit, date)`: Retrieves fleet summary, chemical predictions, and recommended visits.
- `fetchPool(poolId)`: Retrieves pool metadata, historical time-series, and chained multi-step forecasts.
- `triggerFleetInference()`: Forces on-demand multi-step inference recomputation.
- `uploadFile(file)`: Uploads Excel/CSV datasets to Data Ingestion Studio.
- `submitManualReading(payload)`: Ingests a single reading and triggers real-time rolling forecast updates.
- `triggerRetrain()`: Initiates asynchronous XGBoost model retraining.
- `triggerWeatherRefresh()`: Refreshes Open-Meteo weather cache for Alicante.

---

## 5. Internationalization (i18n)

Translations are managed centrally in `src/i18n.ts`. To use a translated string in any component:

```tsx
import { useTranslation } from "../i18n";

export function ExampleComponent() {
  const { t, language, setLanguage } = useTranslation();

  return (
    <div>
      <h2>{t("fleet_overview")}</h2>
      <button onClick={() => setLanguage(language === "en" ? "es" : "en")}>
        {language.toUpperCase()}
      </button>
    </div>
  );
}
```

---

## 6. Regulatory Standards Compliance

The dashboard enforces Spanish Pool Water Quality Regulations:
- **RD 742/2013** (Spanish National Law):
  - Free Chlorine: `0.5 – 2.0 mg/L` (Optimal) | `< 0.5 mg/L` (Pathogen hazard) | `> 5.0 mg/L` (Closure).
  - pH: `7.2 – 8.0` | `< 7.2` (Corrosion hazard) | `> 8.0` (Scale & chlorine inefficacy).
  - Turbidity: `≤ 5.0 NTU` (Clarity standard).
- **Decreto 85/2018** (Comunitat Valenciana Autocontrol).
