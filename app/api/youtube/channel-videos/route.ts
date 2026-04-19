import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type ChannelVideo = {
  videoId: string;
  title: string;
  url: string;
  thumbnailUrl: string;
  publishedAt: string;
  viewCount: number;
  likeCount: number;
  commentCount: number;
};

export async function GET(req: Request) {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: "Missing API_KEY in environment" },
      { status: 500 }
    );
  }

  const { searchParams } = new URL(req.url);
  const channelId = searchParams.get("channelId");
  const max = Math.min(Math.max(Number(searchParams.get("max") ?? "5"), 1), 25);
  if (!channelId) {
    return NextResponse.json(
      { error: "channelId is required" },
      { status: 400 }
    );
  }

  const searchParamsApi = new URLSearchParams({
    part: "snippet",
    channelId,
    order: "date",
    type: "video",
    maxResults: String(max),
    key: apiKey,
  });
  const searchRes = await fetch(
    `https://www.googleapis.com/youtube/v3/search?${searchParamsApi.toString()}`,
    { cache: "no-store" }
  );
  if (!searchRes.ok) {
    const text = await searchRes.text();
    return NextResponse.json(
      { error: `YouTube search API error: ${text}` },
      { status: searchRes.status }
    );
  }
  const searchData = await searchRes.json();
  const ids: string[] = (searchData.items ?? [])
    .map((it: { id?: { videoId?: string } }) => it.id?.videoId)
    .filter((id: string | undefined): id is string => typeof id === "string");

  if (ids.length === 0) {
    return NextResponse.json({ items: [] });
  }

  const videosParams = new URLSearchParams({
    part: "snippet,statistics",
    id: ids.join(","),
    key: apiKey,
  });
  const videosRes = await fetch(
    `https://www.googleapis.com/youtube/v3/videos?${videosParams.toString()}`,
    { cache: "no-store" }
  );
  if (!videosRes.ok) {
    const text = await videosRes.text();
    return NextResponse.json(
      { error: `YouTube videos API error: ${text}` },
      { status: videosRes.status }
    );
  }
  const videosData = await videosRes.json();

  type Thumb = { url: string; width?: number; height?: number };
  const items: ChannelVideo[] = (videosData.items ?? []).map(
    (it: {
      id: string;
      snippet: {
        title: string;
        publishedAt: string;
        thumbnails: Record<string, Thumb | undefined>;
      };
      statistics: {
        viewCount?: string;
        likeCount?: string;
        commentCount?: string;
      };
    }) => {
      const t = it.snippet.thumbnails;
      const thumb =
        t.medium?.url ?? t.high?.url ?? t.default?.url ?? t.standard?.url ?? "";
      return {
        videoId: it.id,
        title: it.snippet.title,
        url: `https://www.youtube.com/watch?v=${it.id}`,
        thumbnailUrl: thumb,
        publishedAt: it.snippet.publishedAt,
        viewCount: Number(it.statistics?.viewCount ?? 0),
        likeCount: Number(it.statistics?.likeCount ?? 0),
        commentCount: Number(it.statistics?.commentCount ?? 0),
      };
    }
  );

  items.sort(
    (a, b) =>
      new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime()
  );

  return NextResponse.json({ items });
}
