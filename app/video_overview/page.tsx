"use client";

import React, { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";

type VideoInfo = {
  id: string;
  title: string;
  publishedAt: string;
  categoryId: string;
  categoryName: string;
  viewCount: number;
  channelId: string;
  channelTitle: string;
};

type ChannelInfo = {
  id: string;
  title: string;
  country: string | null;
  subscriberCount: number;
  viewCount: number;
  videoCount: number;
};

type Stage1Prediction = {
  probability: number;
  probability_percent: number;
  risk_label: string;
  predicted_trendy: boolean;
  predicted_class: number;
};

type Stage2Prediction = {
  vpd_class: number;
  vpd_class_label: string;
  class_probabilities: number[];
  predicted_class_confidence: number;
};

type ViralityHint = {
  feature: string;
  importance: number;
  hint: string;
};

type PipelineResult = {
  stage1: Stage1Prediction;
  stage2: Stage2Prediction | null;
  virality_hints: ViralityHint[] | null;
};

/** Gradient + glow for the trending probability bar (by risk band). */
function viralBarStyle(riskLabel: string, percent: number): React.CSSProperties {
  const w = Math.min(100, Math.max(0, percent));
  let gradient: string;
  let glow: string;
  switch (riskLabel) {
    case "HIGH":
      gradient =
        "linear-gradient(90deg, #06b6d4 0%, #6366f1 35%, #a855f7 70%, #ec4899 100%)";
      glow = "0 0 16px rgba(99, 102, 241, 0.5)";
      break;
    case "MEDIUM":
      gradient =
        "linear-gradient(90deg, #22c55e 0%, #14b8a6 45%, #0ea5e9 100%)";
      glow = "0 0 12px rgba(14, 165, 233, 0.4)";
      break;
    default:
      gradient =
        "linear-gradient(90deg, #94a3b8 0%, #a8b7cf 50%, #cbd5e1 100%)";
      glow = "0 0 8px rgba(148, 163, 184, 0.3)";
  }
  return {
    width: `${w}%`,
    background: gradient,
    boxShadow: glow,
  };
}

const numberFmt = new Intl.NumberFormat("en-US");
const dateFmt = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "short",
  day: "numeric",
});

const STORAGE_KEY = "yourisk:lastScan";

function VideoOverviewContent() {
  const searchParams = useSearchParams();
  const [url, setUrl] = useState("");
  const [video, setVideo] = useState<VideoInfo | null>(null);
  const [pipeline, setPipeline] = useState<PipelineResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictNote, setPredictNote] = useState<string | null>(null);
  const autoScannedRef = useRef<string | null>(null);

  const hasData = !!video;

  async function handleScan(urlOverride?: string) {
    const target = (urlOverride ?? url).trim();
    if (!target || loading) return;
    setLoading(true);
    setError(null);
    setPredictNote(null);
    setPipeline(null);
    try {
      const res = await fetch(`/api/youtube?url=${encodeURIComponent(target)}`);
      const data = (await res.json()) as
        | { video: VideoInfo; channel: ChannelInfo | null }
        | { error: string };
      if (!res.ok || "error" in data) {
        const message = "error" in data ? data.error : "Failed to fetch video.";
        throw new Error(message);
      }
      setVideo(data.video);

      const pr = await fetch("/api/trending-predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video: data.video, channel: data.channel }),
      });
      const rawBody = await pr.text();
      let predJson: PipelineResult | { error: string; detail?: string; hint?: string } | null =
        null;
      try {
        predJson = JSON.parse(rawBody) as PipelineResult | {
          error: string;
          detail?: string;
          hint?: string;
        };
      } catch {
        setPredictNote(
          `Prediction response was not JSON (HTTP ${pr.status}). ${rawBody.slice(0, 280)}`
        );
        setPipeline(null);
      }
      if (predJson) {
        if (!pr.ok || "error" in predJson) {
          const pe = predJson as { error: string; detail?: string; hint?: string };
          let msg = pe.error || "Model prediction failed.";
          if (pe.detail) msg += ` — ${String(pe.detail).slice(0, 500)}`;
          if (pe.hint) msg += ` (${String(pe.hint).slice(0, 400)})`;
          setPredictNote(msg);
          setPipeline(null);
        } else {
          setPipeline(predJson as PipelineResult);
        }
      }

      if (typeof window !== "undefined") {
        window.localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({ video: data.video, channel: data.channel, ts: Date.now() })
        );
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : "Unexpected error.";
      setError(message);
      setVideo(null);
      setPipeline(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    const fromQuery = searchParams.get("url");
    if (!fromQuery) return;
    if (autoScannedRef.current === fromQuery) return;
    autoScannedRef.current = fromQuery;
    setUrl(fromQuery);
    handleScan(fromQuery);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

        .vo-root {
          font-family: 'DM Sans', sans-serif;
          background: #f5f4f0;
          min-height: 100vh;
          padding: 40px;
          color: #1e293b;
        }

        .vo-label {
          font-family: 'DM Mono', monospace;
          font-size: 10px;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #64748b;
        }

        /* Header */
        .vo-header {
          margin-bottom: 36px;
          display: flex;
          align-items: flex-end;
          justify-content: space-between;
        }

        .vo-title {
          font-size: 28px;
          font-weight: 700;
          letter-spacing: -0.03em;
          color: #0f172a; /* Dark Navy Text */
          line-height: 1;
          margin: 0 0 6px;
        }

        .vo-subtitle {
          font-size: 13px;
          color: #64748b;
          font-weight: 400;
          margin: 0;
        }

        .vo-badge {
          font-family: 'DM Mono', monospace;
          font-size: 11px;
          padding: 5px 10px;
          border-radius: 4px;
          background: #e2e8f0;
          color: #475569;
          letter-spacing: 0.05em;
        }

        /* Input Row */
        .vo-input-row {
          display: flex;
          gap: 12px;
          margin-bottom: 28px;
          align-items: stretch;
        }

        .vo-input {
          flex: 1;
          padding: 14px 18px;
          font-family: 'DM Sans', sans-serif;
          font-size: 14px;
          font-weight: 500;
          color: #0f172a;
          background: #fff;
          border: 1.5px solid #cbd5e1;
          border-radius: 10px;
          outline: none;
          transition: border-color 0.15s;
        }

        .vo-input::placeholder { color: #64748b; }
        .vo-input:focus { border-color: #1e3a8a; } /* Navy Blue focus */

        .vo-btn {
          padding: 14px 28px;
          font-family: 'DM Sans', sans-serif;
          font-size: 13px;
          font-weight: 600;
          letter-spacing: 0.02em;
          color: #fff;
          background: #1e3a8a; /* Primary Navy Blue */
          border: none;
          border-radius: 10px;
          cursor: pointer;
          transition: background 0.15s, transform 0.1s;
          white-space: nowrap;
        }

        .vo-btn:hover { background: #172554; } /* Darker Navy on hover */
        .vo-btn:active { transform: scale(0.98); }

        /* Divider */
        .vo-divider {
          height: 1px;
          background: #cbd5e1;
          margin-bottom: 28px;
        }

        /* Stats Strip */
        .vo-stats {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1px;
          background: #cbd5e1;
          border: 1px solid #cbd5e1;
          border-radius: 12px;
          overflow: hidden;
          margin-bottom: 20px;
        }

        .vo-stat {
          background: #fff;
          padding: 20px 24px;
        }

        .vo-stat-value {
          font-size: 22px;
          font-weight: 700;
          letter-spacing: -0.03em;
          color: #0f172a;
          margin-top: 8px;
          line-height: 1;
        }

        /* Cards Row */
        .vo-cards {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
        }

        .vo-card {
          background: #fff;
          border: 1px solid #cbd5e1;
          border-radius: 12px;
          padding: 26px;
        }

        .vo-card-title {
          font-size: 14px;
          font-weight: 600;
          color: #0f172a;
          margin: 0 0 4px;
        }

        .vo-card-sub {
          font-size: 12px;
          color: #64748b;
          margin: 0 0 24px;
        }

        /* Probability Row */
        .vo-prob-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 14px;
        }

        .vo-prob-value {
          font-family: 'DM Mono', monospace;
          font-size: 28px;
          font-weight: 500;
          color: #0f172a;
          letter-spacing: -0.02em;
        }

        .vo-pill {
          font-family: 'DM Mono', monospace;
          font-size: 11px;
          padding: 4px 10px;
          border-radius: 20px;
          background: #e0e7ff; /* Light Navy tint */
          color: #1e3a8a; /* Navy Blue text */
          letter-spacing: 0.04em;
        }

        .vo-track {
          height: 8px;
          border-radius: 99px;
          background: linear-gradient(180deg, #e8eef5 0%, #f1f5f9 100%);
          overflow: hidden;
          box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.06);
        }

        .vo-fill {
          height: 100%;
          border-radius: 99px;
          min-width: 0;
          transition:
            width 0.8s cubic-bezier(0.16, 1, 0.3, 1),
            box-shadow 0.35s ease,
            filter 0.35s ease;
        }

        /* Days Card */
        .vo-days-number {
          font-size: 60px;
          font-weight: 700;
          letter-spacing: -0.05em;
          color: #0f172a;
          line-height: 1;
          margin: 8px 0 4px;
        }

        .vo-days-unit {
          font-size: 14px;
          font-weight: 500;
          color: #64748b;
          display: block;
          margin-bottom: 20px;
        }

        .vo-peak-row {
          display: flex;
          align-items: center;
          gap: 10px;
          padding-top: 18px;
          border-top: 1px solid #f1f5f9;
        }

        .vo-dot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: #1e3a8a; /* Primary Navy Blue */
          flex-shrink: 0;
        }

        .vo-peak-label {
          font-size: 12px;
          color: #64748b;
        }

        .vo-peak-date {
          font-family: 'DM Mono', monospace;
          font-size: 12px;
          color: #0f172a;
          font-weight: 500;
          margin-left: auto;
        }

        /* Fade-in */
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        .vo-reveal {
          animation: fadeUp 0.4s ease forwards;
        }

        .vo-reveal-1 { animation-delay: 0.05s; opacity: 0; }
        .vo-reveal-2 { animation-delay: 0.12s; opacity: 0; }
        .vo-reveal-3 { animation-delay: 0.19s; opacity: 0; }
      `}</style>

      <div className="vo-root">
        {/* Header */}
        <div className="vo-header">
          <div>
            <h1 className="vo-title">Video Asset Overview</h1>
            <p className="vo-subtitle">Analyze a video's trajectory and viral risk profile.</p>
          </div>
        </div>

        {/* Input Row */}
        <div className="vo-input-row">
          <input
            type="text"
            className="vo-input"
            placeholder="Paste YouTube video URL…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleScan()}
            disabled={loading}
          />
          <button
            className="vo-btn"
            onClick={() => handleScan()}
            disabled={loading || !url.trim()}
            style={loading || !url.trim() ? { opacity: 0.6, cursor: "not-allowed" } : undefined}
          >
            {loading ? "Scanning…" : "Scan Asset →"}
          </button>
        </div>

        {error && (
          <div
            style={{
              padding: "12px 16px",
              borderRadius: 10,
              background: "#fef2f2",
              border: "1px solid #fecaca",
              color: "#b91c1c",
              fontSize: 13,
              marginBottom: 20,
            }}
          >
            {error}
          </div>
        )}

        {hasData && video && (
          <>
            <div className="vo-divider" />

            {/* Stats Strip */}
            <div className="vo-stats vo-reveal vo-reveal-1">
              <div className="vo-stat">
                <div className="vo-label">Current Views</div>
                <div className="vo-stat-value">{numberFmt.format(video.viewCount)}</div>
              </div>
              <div className="vo-stat">
                <div className="vo-label">Publish Date</div>
                <div className="vo-stat-value">
                  {dateFmt.format(new Date(video.publishedAt))}
                </div>
              </div>
              <div className="vo-stat">
                <div className="vo-label">Primary Category</div>
                <div className="vo-stat-value">{video.categoryName}</div>
              </div>
            </div>

            {/* Cards */}
            <div className="vo-cards vo-reveal vo-reveal-2">
              {/* Trending Prediction (Stage 1) */}
              <div className="vo-card">
                <p className="vo-card-title">Trending Prediction</p>
                {pipeline ? (
                  <>
                    <div className="vo-prob-row">
                      <span className="vo-prob-value">{pipeline.stage1.probability_percent}%</span>
                      <span className="vo-pill">{pipeline.stage1.risk_label}</span>
                    </div>
                    <div className="vo-track">
                      <div
                        className="vo-fill"
                        style={viralBarStyle(
                          pipeline.stage1.risk_label,
                          pipeline.stage1.probability_percent
                        )}
                      />
                    </div>
                    <p style={{ fontSize: 12, color: "#64748b", marginTop: 14, marginBottom: 0 }}>
                      {pipeline.stage1.predicted_trendy
                        ? "Likely to trend — see post-trend outlook alongside."
                        : "Lower trending score — see levers below."}
                    </p>
                  </>
                ) : (
                  <p style={{ fontSize: 13, color: "#64748b", marginTop: 8 }}>
                    {predictNote
                      ? `Prediction unavailable: ${predictNote}`
                      : "Run a scan to load model scores (requires YTBTrendingDashboard/models/*.json + Python)."}
                  </p>
                )}
              </div>

              <div className="vo-card vo-reveal vo-reveal-3">
                {pipeline && pipeline.stage1.predicted_trendy && pipeline.stage2 ? (
                  <>
                    <p className="vo-card-title">Post-trend outlook</p>
                    <div className="vo-days-number" style={{ fontSize: 22, lineHeight: 1.3, fontWeight: 600 }}>
                      {pipeline.stage2.vpd_class_label}
                    </div>
                    <span className="vo-days-unit">
                      Confidence {Math.round(pipeline.stage2.predicted_class_confidence * 100)}% · fading /
                      steady / thriving
                    </span>
                    <div className="vo-peak-row" style={{ flexWrap: "wrap", gap: 8 }}>
                      {pipeline.stage2.class_probabilities.map((p, i) => (
                        <span
                          key={i}
                          style={{
                            fontFamily: "DM Mono, monospace",
                            fontSize: 11,
                            padding: "4px 8px",
                            borderRadius: 6,
                            background: "#f1f5f9",
                            color: "#475569",
                          }}
                        >
                          {i}: {(p * 100).toFixed(0)}%
                        </span>
                      ))}
                    </div>
                  </>
                ) : (
                  <>
                    <p className="vo-card-title">Post-trend outlook</p>
                    <p style={{ fontSize: 13, color: "#64748b", marginTop: 8 }}>—</p>
                  </>
                )}
              </div>
            </div>

            {pipeline && pipeline.virality_hints && pipeline.virality_hints.length > 0 && (
              <div
                className="vo-reveal vo-reveal-3"
                style={{
                  marginTop: 20,
                  background: "#fff",
                  border: "1px solid #cbd5e1",
                  borderRadius: 12,
                  padding: "22px 26px",
                }}
              >
                <p className="vo-card-title" style={{ marginBottom: 4 }}>
                  Virality levers (top signals)
                </p>
                <p className="vo-card-sub" style={{ marginBottom: 18 }}>
                  Strongest global signals from the trending model
                </p>
                <ol style={{ margin: 0, paddingLeft: 18, color: "#334155", fontSize: 13, lineHeight: 1.55 }}>
                  {pipeline.virality_hints.map((h) => (
                    <li key={h.feature} style={{ marginBottom: 10 }}>
                      <strong style={{ fontFamily: "DM Mono, monospace", fontSize: 12 }}>{h.feature}</strong>
                      <span style={{ color: "#94a3b8", marginLeft: 8 }}>
                        (weight {h.importance.toFixed(4)})
                      </span>
                      <div style={{ color: "#64748b", marginTop: 4 }}>{h.hint}</div>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}

export default function VideoOverview() {
  return (
    <Suspense fallback={null}>
      <VideoOverviewContent />
    </Suspense>
  );
}