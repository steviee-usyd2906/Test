import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RecommendedVideo = {
  videoId: string;
  title: string;
  channelTitle: string;
  publishedAt: string;
  url: string;
};

async function fetchMostPopular(
  apiKey: string,
  regionCode: string,
  categoryId: string,
  maxResults: number
): Promise<{ items: RecommendedVideo[]; rawError?: string; status: number }> {
  const params = new URLSearchParams({
    part: "snippet",
    chart: "mostPopular",
    regionCode,
    videoCategoryId: categoryId,
    maxResults: String(maxResults),
    key: apiKey,
  });
  const res = await fetch(
    `https://www.googleapis.com/youtube/v3/videos?${params.toString()}`,
    { cache: "no-store" }
  );
  if (!res.ok) {
    const text = await res.text();
    return { items: [], rawError: text, status: res.status };
  }
  const data = await res.json();
  const items: RecommendedVideo[] = (data.items ?? []).map(
    (it: {
      id: string;
      snippet: { title: string; channelTitle: string; publishedAt: string };
    }) => ({
      videoId: it.id,
      title: it.snippet.title,
      channelTitle: it.snippet.channelTitle,
      publishedAt: it.snippet.publishedAt,
      url: `https://www.youtube.com/watch?v=${it.id}`,
    })
  );
  return { items, status: res.status };
}

async function fetchSearchFallback(
  apiKey: string,
  regionCode: string,
  query: string,
  maxResults: number
): Promise<{ items: RecommendedVideo[]; rawError?: string; status: number }> {
  const params = new URLSearchParams({
    part: "snippet",
    type: "video",
    order: "viewCount",
    regionCode,
    q: query,
    maxResults: String(maxResults),
    key: apiKey,
  });
  const res = await fetch(
    `https://www.googleapis.com/youtube/v3/search?${params.toString()}`,
    { cache: "no-store" }
  );
  if (!res.ok) {
    const text = await res.text();
    return { items: [], rawError: text, status: res.status };
  }
  const data = await res.json();
  const items: RecommendedVideo[] = (data.items ?? [])
    .filter((it: { id?: { videoId?: string } }) => it.id?.videoId)
    .map(
      (it: {
        id: { videoId: string };
        snippet: { title: string; channelTitle: string; publishedAt: string };
      }) => ({
        videoId: it.id.videoId,
        title: it.snippet.title,
        channelTitle: it.snippet.channelTitle,
        publishedAt: it.snippet.publishedAt,
        url: `https://www.youtube.com/watch?v=${it.id.videoId}`,
      })
    );
  return { items, status: res.status };
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
  const categoryId = searchParams.get("categoryId");
  const regionCode = searchParams.get("regionCode");
  const excludeId = searchParams.get("excludeId");
  const categoryName = searchParams.get("categoryName") ?? "";
  const max = Math.min(Math.max(Number(searchParams.get("max") ?? "5"), 1), 25);

  if (!categoryId || !regionCode) {
    return NextResponse.json(
      { error: "categoryId and regionCode are required" },
      { status: 400 }
    );
  }

  const overFetch = max + (excludeId ? 1 : 0);

  let source: "mostPopular" | "search" = "mostPopular";
  let result = await fetchMostPopular(apiKey, regionCode, categoryId, overFetch);

  if (result.items.length === 0 && categoryName) {
    source = "search";
    const fallback = await fetchSearchFallback(
      apiKey,
      regionCode,
      categoryName,
      overFetch
    );
    if (fallback.items.length > 0) {
      result = fallback;
    } else if (fallback.rawError && result.rawError == null) {
      result = fallback;
    }
  }

  if (result.items.length === 0 && result.rawError) {
    return NextResponse.json(
      { error: `YouTube API error: ${result.rawError}` },
      { status: result.status }
    );
  }

  const items = result.items
    .filter((v) => !excludeId || v.videoId !== excludeId)
    .slice(0, max);

  return NextResponse.json({ items, source });
}
