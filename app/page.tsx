"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

function extractVideoId(input: string): string | null {
  if (!input) return null;
  const trimmed = input.trim();
  if (/^[a-zA-Z0-9_-]{11}$/.test(trimmed)) return trimmed;
  try {
    const url = new URL(trimmed);
    const host = url.hostname.replace(/^www\./, "");
    if (host === "youtu.be") {
      return url.pathname.slice(1).split("/")[0] || null;
    }
    if (host.endsWith("youtube.com")) {
      const v = url.searchParams.get("v");
      if (v) return v;
      const parts = url.pathname.split("/").filter(Boolean);
      if (parts.length >= 2 && ["shorts", "embed", "v", "live"].includes(parts[0])) {
        return parts[1];
      }
    }
  } catch {
    return null;
  }
  return null;
}

export default function HomePage() {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (submitting) return;
    const id = extractVideoId(url);
    if (!id) {
      setError("That doesn't look like a valid YouTube video URL. Please try again.");
      return;
    }
    setError(null);
    setSubmitting(true);
    router.push(`/video_overview?url=${encodeURIComponent(url.trim())}`);
  }

  return (
    <div className="flex flex-col items-center justify-center h-[75vh] animate-in fade-in duration-1000 slide-in-from-bottom-4">
      <div className="text-center space-y-6 max-w-xl w-full">

        <h1 className="text-5xl font-black tracking-tight text-slate-900">
          Welcome to YouRisk
        </h1>
        <p className="text-lg font-medium text-slate-500 leading-relaxed px-4">
          The enterprise standard for digital media asset valuation. Stop guessing on sponsorships and start calculating true, long-term ROI.
        </p>

        <form onSubmit={handleSubmit} className="pt-6 space-y-3 px-4">
          <div className="flex gap-2">
            <input
              type="text"
              value={url}
              onChange={(e) => {
                setUrl(e.target.value);
                if (error) setError(null);
              }}
              placeholder="Paste a YouTube video URL…"
              autoFocus
              disabled={submitting}
              className={`flex-1 px-5 py-4 text-base font-medium rounded-xl bg-white/80 text-slate-900 placeholder:text-slate-400 outline-none border-2 transition-colors ${
                error
                  ? "border-red-400 focus:border-red-500"
                  : "border-slate-200 focus:border-blue-700"
              }`}
            />
            <button
              type="submit"
              disabled={submitting || !url.trim()}
              className="inline-flex items-center gap-2 px-7 py-4 text-base font-bold rounded-xl text-white shadow-md bg-gradient-to-br from-blue-900 to-blue-600 hover:shadow-lg transition-all hover:-translate-y-0.5 disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:translate-y-0"
            >
              {submitting ? "Loading…" : "Analyze"}
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M17 8l4 4m0 0l-4 4m4-4H3" />
              </svg>
            </button>
          </div>

          {error && (
            <div
              role="alert"
              className="text-left text-sm font-semibold text-red-700 bg-red-50 border border-red-200 rounded-lg px-4 py-2.5"
            >
              {error}
            </div>
          )}
        </form>

      </div>
    </div>
  );
}
