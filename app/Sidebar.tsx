"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { id: "/video_overview", label: "Video Overview" },
  { id: "/channel_analysis", label: "Channel Analysis" },
  { id: "/portfolio_recommendation", label: "Portfolio Recommendation" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const showNav = pathname !== "/";

  return (
    <aside
      className="w-64 min-h-screen flex flex-col p-6 shrink-0 relative shadow-sm"
      style={{
        zIndex: 1,
        background: "rgba(255,255,255,0.65)",
        borderRight: "1px solid rgba(148, 163, 184, 0.25)",
        backdropFilter: "blur(16px)",
      }}
    >
      <div className="mb-10">
        <Link href="/" className="block">
          <div className="text-2xl font-black tracking-tight" style={{ color: "#1e3a8a" }}>
            YouRisk
          </div>
          <div
            className="text-xs font-bold tracking-[0.2em] uppercase mt-1"
            style={{ color: "#475569" }}
          >
            Media Valuation
          </div>
        </Link>
      </div>

      {showNav && (
        <nav className="flex flex-col gap-2">
          {navItems.map(({ id, label }) => {
            const active = pathname === id;
            return (
              <Link
                key={id}
                href={id}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all ${
                  active
                    ? "bg-white/80 text-blue-700 shadow-sm"
                    : "hover:bg-white/50 text-slate-700 hover:text-blue-600"
                }`}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      )}
    </aside>
  );
}
