"use client";

import React, { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

type ChannelInfo = {
  id: string;
  title: string;
  country: string | null;
  subscriberCount: number;
  viewCount: number;
  videoCount: number;
};

type StoredScan = {
  video: { channelTitle: string };
  channel: ChannelInfo | null;
};

type RecentVideo = {
  videoId: string;
  title: string;
  url: string;
  thumbnailUrl: string;
  publishedAt: string;
  viewCount: number;
  likeCount: number;
  commentCount: number;
};

const STORAGE_KEY = "yourisk:lastScan";
const numberFmt = new Intl.NumberFormat("en-US");
const compactFmt = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

let regionNames: Intl.DisplayNames | null = null;
function getCountryName(code: string | null): string {
  if (!code) return "Unknown";
  if (typeof Intl !== "undefined" && "DisplayNames" in Intl) {
    if (!regionNames) {
      try {
        regionNames = new Intl.DisplayNames(["en"], { type: "region" });
      } catch {
        regionNames = null;
      }
    }
    if (regionNames) {
      try {
        return regionNames.of(code) ?? code;
      } catch {
        return code;
      }
    }
  }
  return code;
}

type ScoreResult = {
  score: number;
  tier: string;
  tierColor: string;
};

function calculateEngagementQuality(videos: RecentVideo[]): number {
  if (videos.length === 0) return 0;
  const totalLikes = videos.reduce((sum, v) => sum + v.likeCount, 0);
  const totalComments = videos.reduce((sum, v) => sum + v.commentCount, 0);
  const totalViews = videos.reduce((sum, v) => sum + v.viewCount, 0);
  if (totalViews === 0) return 0;
  const likeRatio = totalLikes / totalViews;
  const commentRatio = totalComments / totalViews;
  const combined = (likeRatio + commentRatio) * 100;
  // Healthy engagement: 5-15% combined, penalize extremes
  if (combined < 0.5) return 0;
  if (combined > 20) return 40; // Suspicious
  return Math.min(100, (combined / 15) * 100);
}

function calculateChannelEfficiency(
  channel: ChannelInfo,
  videos: RecentVideo[]
): number {
  const { subscriberCount, viewCount, videoCount } = channel;
  if (subscriberCount === 0 || videoCount === 0) return 0;

  const viewsPerSub = viewCount / Math.max(subscriberCount, 1);
  const viewsPerVideo = viewCount / videoCount;
  const subsPerVideo = subscriberCount / videoCount;

  // Healthy: 2-20 views per subscriber, >1k views per video, >100 subs per video
  const viewPerSubScore = Math.min(100, (viewsPerSub / 10) * 100);
  const viewPerVideoScore = Math.min(100, (Math.log10(viewsPerVideo + 1) / 4) * 100);
  const subsPerVideoScore = Math.min(100, (Math.log10(subsPerVideo + 1) / 3) * 100);

  return (viewPerSubScore + viewPerVideoScore + subsPerVideoScore) / 3;
}

function calculateUploadConsistency(videos: RecentVideo[]): number {
  if (videos.length < 2) return 50; // Insufficient data

  const sorted = [...videos].sort(
    (a, b) => new Date(a.publishedAt).getTime() - new Date(b.publishedAt).getTime()
  );

  const gaps: number[] = [];
  for (let i = 1; i < sorted.length; i++) {
    const daysBetween =
      (new Date(sorted[i].publishedAt).getTime() -
       new Date(sorted[i - 1].publishedAt).getTime()) / (1000 * 60 * 60 * 24);
    gaps.push(daysBetween);
  }

  const avgGap = gaps.reduce((a, b) => a + b, 0) / gaps.length;
  const variance = gaps.reduce((sum, gap) => sum + Math.pow(gap - avgGap, 2), 0) / gaps.length;
  const stdDev = Math.sqrt(variance);
  const consistency = 100 - Math.min(100, (stdDev / avgGap) * 100);

  // Penalty for very long gaps (>60 days = inactive)
  if (avgGap > 60) return Math.max(20, consistency * 0.6);

  return Math.max(0, consistency);
}

function calculateContentStability(videos: RecentVideo[]): number {
  if (videos.length < 2) return 50;

  const views = videos.map(v => v.viewCount);
  const avgViews = views.reduce((a, b) => a + b, 0) / views.length;
  const variance = views.reduce((sum, v) => sum + Math.pow(v - avgViews, 2), 0) / views.length;
  const stdDev = Math.sqrt(variance);
  const cv = stdDev / (avgViews || 1); // Coefficient of variation

  // Stable = low CV, unstable = high CV
  const stability = 100 - Math.min(100, cv * 100);

  return Math.max(0, stability);
}

function calculateReliabilityScore(
  channel: ChannelInfo | null,
  videos: RecentVideo[]
): ScoreResult {
  if (!channel || videos.length === 0) {
    return { score: 0, tier: "Insufficient Data", tierColor: "bg-gray-50 border-gray-200 text-gray-700" };
  }

  const engagement = calculateEngagementQuality(videos) * 0.35;
  const efficiency = calculateChannelEfficiency(channel, videos) * 0.25;
  const consistency = calculateUploadConsistency(videos) * 0.20;
  const stability = calculateContentStability(videos) * 0.20;

  const score = engagement + efficiency + consistency + stability;

  let tier = "F";
  let tierColor = "bg-red-50 border-red-200 text-red-700";

  if (score >= 90) {
    tier = "AAA";
    tierColor = "bg-emerald-50 border-emerald-200 text-emerald-700";
  } else if (score >= 75) {
    tier = "A";
    tierColor = "bg-green-50 border-green-200 text-green-700";
  } else if (score >= 60) {
    tier = "B";
    tierColor = "bg-blue-50 border-blue-200 text-blue-700";
  } else if (score >= 40) {
    tier = "C";
    tierColor = "bg-yellow-50 border-yellow-200 text-yellow-700";
  } else if (score >= 20) {
    tier = "D";
    tierColor = "bg-orange-50 border-orange-200 text-orange-700";
  }

  return {
    score: Math.round(score),
    tier: tier === "AAA" ? `High Trust Tier — ${tier}` : `Tier ${tier}`,
    tierColor,
  };
}

export default function ChannelAnalysis() {
  const [channel, setChannel] = useState<ChannelInfo | null>(null);
  const [channelTitle, setChannelTitle] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [recentVideos, setRecentVideos] = useState<RecentVideo[]>([]);
  const [recentLoading, setRecentLoading] = useState(false);
  const [recentError, setRecentError] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as StoredScan;
        setChannel(parsed.channel ?? null);
        setChannelTitle(parsed.video?.channelTitle ?? null);
      }
    } catch {
      // ignore corrupted storage
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (!channel?.id) return;
    const controller = new AbortController();
    setRecentLoading(true);
    setRecentError(null);

    fetch(`/api/youtube/channel-videos?channelId=${encodeURIComponent(channel.id)}&max=5`, {
      signal: controller.signal,
    })
      .then(async (res) => {
        const data = (await res.json()) as
          | { items: RecentVideo[] }
          | { error: string };
        if (!res.ok || "error" in data) {
          throw new Error("error" in data ? data.error : "Failed to load videos.");
        }
        setRecentVideos(data.items);
      })
      .catch((e: unknown) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setRecentError(e instanceof Error ? e.message : "Unexpected error.");
      })
      .finally(() => setRecentLoading(false));

    return () => controller.abort();
  }, [channel?.id]);

  const countryCode = channel?.country ?? null;
  const countryDisplay = getCountryName(countryCode);

  // Sort ascending by publish date so the time series reads left → right (oldest → newest).
  const chartData = useMemo(
    () =>
      [...recentVideos].sort(
        (a, b) =>
          new Date(a.publishedAt).getTime() - new Date(b.publishedAt).getTime()
      ),
    [recentVideos]
  );
  const thumbnailByVideoId = useMemo(() => {
    const m = new Map<string, RecentVideo>();
    recentVideos.forEach((v) => m.set(v.videoId, v));
    return m;
  }, [recentVideos]);

  const reliabilityScore = useMemo(
    () => calculateReliabilityScore(channel, recentVideos),
    [channel, recentVideos]
  );

  type AxisTickProps = {
    x?: number;
    y?: number;
    payload?: { value: string };
  };
  const formatDDMMYY = (iso: string) => {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "";
    const dd = String(d.getDate()).padStart(2, "0");
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const yy = String(d.getFullYear()).slice(-2);
    return `${dd}/${mm}/${yy}`;
  };

  const ThumbnailXTick = (props: AxisTickProps) => {
    const { x = 0, y = 0, payload } = props;
    const v = payload ? thumbnailByVideoId.get(payload.value) : undefined;
    if (!v) return null;
    const w = 84;
    const h = 47;
    return (
      <g transform={`translate(${x - w / 2}, ${y + 8})`}>
        <a href={v.url} target="_blank" rel="noopener noreferrer">
          <image
            href={v.thumbnailUrl}
            width={w}
            height={h}
            preserveAspectRatio="xMidYMid slice"
            style={{ borderRadius: 6 }}
          />
        </a>
        <text
          x={w / 2}
          y={h + 14}
          textAnchor="middle"
          fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
          fontSize={10}
          fontWeight={600}
          fill="#475569"
        >
          {formatDDMMYY(v.publishedAt)}
        </text>
      </g>
    );
  };

  const seriesMeta = [
    { key: "viewCount", label: "Views", color: "#1e3a8a" },
    { key: "likeCount", label: "Likes", color: "#2563eb" },
    { key: "commentCount", label: "Comments", color: "#60a5fa" },
  ] as const;

  return (
    <div className="space-y-6">

      {/* Header - Base Fade In */}
      <div className="mb-6 animate-in fade-in slide-in-from-bottom-4 duration-700 fill-mode-both">
        <h1 className="text-3xl font-black tracking-tight text-slate-900">Channel Analysis</h1>
        <p className="text-sm font-medium mt-2 text-slate-600">
          {channelTitle
            ? `High-level integrity and global viral footprint for ${channelTitle}.`
            : "High-level channel integrity and global viral footprint."}
        </p>
      </div>

      {loaded && !channel && (
        <div className="rounded-2xl border border-amber-200 bg-amber-50/80 px-5 py-4 text-sm font-medium text-amber-800 animate-in fade-in duration-500">
          Scan a video on the Video Overview page first to populate channel data here.
        </div>
      )}

      {/* Top Stats Grid - Delay 150ms */}
      <div className="grid grid-cols-3 gap-4 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-150 fill-mode-both">
        <div className="bg-white/85 border border-slate-300 rounded-2xl p-5 shadow-sm">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Channel Origin</div>
          <div className="flex items-baseline gap-2">
            <div className="text-2xl font-black text-slate-900">{countryCode ?? "—"}</div>
            <div className="text-sm font-bold text-slate-400">{countryDisplay}</div>
          </div>
        </div>
        <div className="bg-white/85 border border-slate-300 rounded-2xl p-5 shadow-sm">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Total Subscribers</div>
          <div className="text-2xl font-black text-slate-900">
            {channel ? numberFmt.format(channel.subscriberCount) : "—"}
          </div>
        </div>
        <div className="bg-white/85 border border-slate-300 rounded-2xl p-5 shadow-sm">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Total Views</div>
          <div className="text-2xl font-black text-slate-900">
            {channel ? numberFmt.format(channel.viewCount) : "—"}
          </div>
        </div>
      </div>

      {/* Bottom Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

        {/* Reliability Score Donut Chart - Delay 300ms */}
        <div className="bg-white/85 border border-slate-300 rounded-2xl p-6 shadow-sm flex flex-col items-center justify-center col-span-1 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-300 fill-mode-both">
          <div className="w-full text-left mb-6">
            <div className="text-lg font-bold text-slate-900 mb-1">Reliability Score</div>
            <div className="text-sm font-medium text-slate-500">Overall creator trust tier</div>
          </div>

          <div className="relative flex items-center justify-center w-40 h-40 mb-4">
            {/* Donut Background */}
            <svg viewBox="0 0 100 100" className="w-full h-full transform -rotate-90">
              <circle cx="50" cy="50" r="40" fill="transparent" stroke="#e2e8f0" strokeWidth="12" />
              {/* Donut Fill - calculated from score */}
              <circle
                cx="50" cy="50" r="40"
                fill="transparent"
                stroke="#2563eb"
                strokeWidth="12"
                strokeDasharray="251.2"
                strokeDashoffset={String(251.2 * (1 - reliabilityScore.score / 100))}
                strokeLinecap="round"
                className="drop-shadow-sm"
              />
            </svg>
            {/* Center Text */}
            <div className="absolute flex flex-col items-center justify-center">
              <span className="text-4xl font-black text-slate-900 tracking-tighter">
                {reliabilityScore.score}
              </span>
              <span className="text-xs font-bold text-slate-400 uppercase tracking-widest mt-1">/ 100</span>
            </div>
          </div>

          <div className={`text-sm font-bold rounded-xl px-4 py-2.5 border w-full text-center ${reliabilityScore.tierColor}`}>
            {reliabilityScore.tier}
          </div>
        </div>

        {/* Recent Videos Time Series - Delay 500ms */}
        <div className="bg-white/85 border border-slate-300 rounded-2xl p-6 shadow-sm col-span-2 flex flex-col animate-in fade-in slide-in-from-bottom-4 duration-700 delay-500 fill-mode-both">
          <div>
            <div className="text-lg font-bold text-slate-900 mb-1">5 Most Recent Videos</div>
            <div className="text-sm font-medium text-slate-500 mb-6">
              Views, likes, and comments over time (oldest → newest)
            </div>
          </div>

          {recentLoading && (
            <div className="text-sm font-medium text-slate-500">Loading recent videos…</div>
          )}

          {recentError && (
            <div className="text-sm font-medium text-red-700 bg-red-50 border border-red-200 rounded-lg px-4 py-3">
              {recentError}
            </div>
          )}

          {!recentLoading && !recentError && !channel && (
            <div className="text-sm font-medium text-slate-500">
              Scan a video on the Video Overview page first to load this channel's videos.
            </div>
          )}

          {!recentLoading && !recentError && channel && recentVideos.length === 0 && (
            <div className="text-sm font-medium text-slate-500">
              No recent videos found for this channel.
            </div>
          )}

          {!recentLoading && !recentError && chartData.length > 0 && (
            <div className="space-y-2">
              {seriesMeta.map((s, i) => {
                const isLast = i === seriesMeta.length - 1;
                return (
                  <div key={s.key}>
                    <div className="text-xs font-bold uppercase tracking-wider text-slate-500 ml-2 mb-1">
                      {s.label}
                    </div>
                    <div
                      className="w-full"
                      style={{ height: isLast ? 220 : 130 }}
                    >
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={chartData}
                          margin={{
                            top: 8,
                            right: 24,
                            left: 8,
                            bottom: isLast ? 80 : 4,
                          }}
                          barCategoryGap="25%"
                        >
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="#e2e8f0"
                            vertical={false}
                          />
                          <XAxis
                            dataKey="videoId"
                            interval={0}
                            tickLine={false}
                            axisLine={{ stroke: "#cbd5e1" }}
                            tick={isLast ? <ThumbnailXTick /> : false}
                            height={isLast ? 80 : 8}
                          />
                          <YAxis
                            tickFormatter={(v: number) => compactFmt.format(v)}
                            tick={{ fill: "#64748b", fontSize: 11, fontWeight: 600 }}
                            axisLine={{ stroke: "#cbd5e1" }}
                            tickLine={{ stroke: "#cbd5e1" }}
                            width={56}
                          />
                          <Tooltip
                            cursor={{ fill: "rgba(30, 58, 138, 0.05)" }}
                            labelFormatter={(label) => {
                              const id =
                                typeof label === "string" ? label : String(label);
                              return thumbnailByVideoId.get(id)?.title ?? id;
                            }}
                            formatter={(value) => [
                              numberFmt.format(Number(value ?? 0)),
                              s.label,
                            ]}
                            contentStyle={{
                              background: "rgba(255,255,255,0.95)",
                              border: "1px solid #cbd5e1",
                              borderRadius: 10,
                              fontSize: 12,
                              maxWidth: 320,
                            }}
                            labelStyle={{ fontWeight: 700, color: "#0f172a" }}
                          />
                          <Bar
                            dataKey={s.key}
                            fill={s.color}
                            radius={[4, 4, 0, 0]}
                            maxBarSize={60}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}