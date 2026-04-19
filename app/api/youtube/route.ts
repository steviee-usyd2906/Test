import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const CATEGORY_MAP: Record<string, string> = {
  "1": "Film & Animation",
  "2": "Autos & Vehicles",
  "10": "Music",
  "15": "Pets & Animals",
  "17": "Sports",
  "18": "Short Movies",
  "19": "Travel & Events",
  "20": "Gaming",
  "21": "Videoblogging",
  "22": "People & Blogs",
  "23": "Comedy",
  "24": "Entertainment",
  "25": "News & Politics",
  "26": "Howto & Style",
  "27": "Education",
  "28": "Science & Technology",
  "29": "Nonprofits & Activism",
  "30": "Movies",
  "31": "Anime/Animation",
  "32": "Action/Adventure",
  "33": "Classics",
  "34": "Comedy",
  "35": "Documentary",
  "36": "Drama",
  "37": "Family",
  "38": "Foreign",
  "39": "Horror",
  "40": "Sci-Fi/Fantasy",
  "41": "Thriller",
  "42": "Shorts",
  "43": "Shows",
  "44": "Trailers",
};

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

export async function GET(req: Request) {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: "Missing API_KEY in environment" },
      { status: 500 }
    );
  }

  const { searchParams } = new URL(req.url);
  const input = searchParams.get("url") || searchParams.get("v") || "";
  const videoId = extractVideoId(input);
  if (!videoId) {
    return NextResponse.json(
      { error: "Invalid YouTube URL or video ID" },
      { status: 400 }
    );
  }

  const videoRes = await fetch(
    `https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id=${videoId}&key=${apiKey}`,
    { cache: "no-store" }
  );
  if (!videoRes.ok) {
    const err = await videoRes.text();
    return NextResponse.json(
      { error: `YouTube videos API error: ${err}` },
      { status: videoRes.status }
    );
  }
  const videoData = await videoRes.json();
  const video = videoData?.items?.[0];
  if (!video) {
    return NextResponse.json({ error: "Video not found" }, { status: 404 });
  }

  const channelId: string = video.snippet.channelId;

  const channelRes = await fetch(
    `https://www.googleapis.com/youtube/v3/channels?part=snippet,statistics&id=${channelId}&key=${apiKey}`,
    { cache: "no-store" }
  );
  if (!channelRes.ok) {
    const err = await channelRes.text();
    return NextResponse.json(
      { error: `YouTube channels API error: ${err}` },
      { status: channelRes.status }
    );
  }
  const channelData = await channelRes.json();
  const channel = channelData?.items?.[0];

  return NextResponse.json({
    video: {
      id: videoId,
      title: video.snippet.title,
      publishedAt: video.snippet.publishedAt,
      categoryId: video.snippet.categoryId,
      categoryName: CATEGORY_MAP[video.snippet.categoryId] ?? "Unknown",
      viewCount: Number(video.statistics?.viewCount ?? 0),
      likeCount: Number(video.statistics?.likeCount ?? 0),
      commentCount: Number(video.statistics?.commentCount ?? 0),
      channelId,
      channelTitle: video.snippet.channelTitle,
    },
    channel: channel
      ? {
          id: channel.id,
          title: channel.snippet.title,
          country: channel.snippet.country ?? null,
          subscriberCount: Number(channel.statistics?.subscriberCount ?? 0),
          viewCount: Number(channel.statistics?.viewCount ?? 0),
          videoCount: Number(channel.statistics?.videoCount ?? 0),
        }
      : null,
  });
}
