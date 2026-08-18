import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import FleetPage from "./pages/FleetPage";
import PoolDetailPage from "./pages/PoolDetailPage";
import AnalyticsPage from "./pages/AnalyticsPage";

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
          {/* Default Landing Page: Mis Piscinas / My Pools */}
          <Route path="/" element={<FleetPage />} />
          <Route path="/piscinas" element={<Navigate to="/" replace />} />
          <Route path="/piscinas/:poolId" element={<PoolDetailPage />} />
          <Route path="/pool/:poolId" element={<PoolDetailPage />} />

          {/* Analytics / Analíticas */}
          <Route path="/analiticas" element={<AnalyticsPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="/admin" element={<AnalyticsPage />} />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
