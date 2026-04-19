import { NextResponse } from "next/server";
import { predictTrendingPipeline } from "@/lib/ml-predictor";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

type Body = {
  video?: Record<string, unknown>;
  channel?: Record<string, unknown> | null;
};

export async function POST(req: Request) {
  let body: Body;
  try {
    body = (await req.json()) as Body;
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (!body.video || typeof body.video !== "object") {
    return NextResponse.json({ error: "Missing video object" }, { status: 400 });
  }

  const v = body.video as Record<string, unknown>;
  const channel =
    body.channel && typeof body.channel === "object"
      ? body.channel
      : {
          id: (v.channelId as string) ?? "",
          title: (v.channelTitle as string) ?? "unknown",
          country: null,
          subscriberCount: 0,
          viewCount: 0,
          videoCount: 0,
        };

  try {
    const result = await predictTrendingPipeline({ video: body.video, channel });
    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        error: "Prediction failed",
        detail: message,
        hint: "Ensure models/stage1_xgb.json + stage1_meta.json (+ stage2) exist under models/.",
      },
      { status: 502 }
    );
  }
}
