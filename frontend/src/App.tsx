import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import HomePage from "./pages/HomePage";
import FleetPage from "./pages/FleetPage";
import PoolDetailPage from "./pages/PoolDetailPage";
import AccountPage from "./pages/AccountPage";
import MessagesPage from "./pages/MessagesPage";
import AnalyticsPage from "./pages/AnalyticsPage";
import CleaningPage from "./pages/CleaningPage";
import IncidentsPage from "./pages/IncidentsPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 30000,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* Main Iberpiscinas Hub Navigation */}
          <Route path="/" element={<HomePage />} />

          {/* 1. Mis Piscinas */}
          <Route path="/piscinas" element={<FleetPage />} />
          <Route path="/piscinas/:poolId" element={<PoolDetailPage />} />
          <Route path="/pool/:poolId" element={<PoolDetailPage />} />

          {/* 2. Mi Cuenta */}
          <Route path="/cuenta" element={<AccountPage />} />

          {/* 3. Mensajes */}
          <Route path="/mensajes" element={<MessagesPage />} />

          {/* 4. Analíticas */}
          <Route path="/analiticas" element={<AnalyticsPage />} />
          <Route path="/admin" element={<AnalyticsPage />} />

          {/* 5. Limpiezas */}
          <Route path="/limpiezas" element={<CleaningPage />} />

          {/* 6. Incidencias */}
          <Route path="/incidencias" element={<IncidentsPage />} />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
