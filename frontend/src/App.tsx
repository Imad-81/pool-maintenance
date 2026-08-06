import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import FleetPage from "./pages/FleetPage";
import PoolDetailPage from "./pages/PoolDetailPage";
import AdminPage from "./pages/AdminPage";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Header />
        <main className="max-w-[1600px] mx-auto px-6 pt-6 pb-12">
          <Routes>
            <Route path="/" element={<FleetPage />} />
            <Route path="/pool/:poolId" element={<PoolDetailPage />} />
            <Route path="/admin" element={<AdminPage />} />
          </Routes>
        </main>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

function Header() {
  const loc = useLocation();
  return (
    <header className="sticky top-0 z-50 border-b border-[#2d3141] bg-gradient-to-r from-[#1e2330] to-[#141820] backdrop-blur-sm px-8 py-4 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-[#4f8ff7] to-[#7c3aed] flex items-center justify-center text-lg">
          🏊
        </div>
        <h1 className="text-lg font-semibold tracking-tight">
          Pool Maintenance<span className="text-[#9aa0a6] font-normal text-sm ml-2">Predictive Dashboard</span>
        </h1>
      </div>
      <nav className="flex gap-1 bg-[#1a1d27] p-1 rounded-lg">
        <Link to="/" className={`px-5 py-2 rounded-md text-sm font-medium transition ${loc.pathname === "/" ? "bg-[#4f8ff7] text-white" : "text-[#9aa0a6] hover:text-white hover:bg-[#2a2e3b]"}`}>
          Fleet Overview
        </Link>
        <Link to="/admin" className={`px-5 py-2 rounded-md text-sm font-medium transition ${loc.pathname === "/admin" ? "bg-[#4f8ff7] text-white" : "text-[#9aa0a6] hover:text-white hover:bg-[#2a2e3b]"}`}>
          Admin
        </Link>
      </nav>
    </header>
  );
}
