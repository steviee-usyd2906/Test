import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import readline from "readline";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const CATEGORY_NAMES: Record<number, string> = {
  1: "Film & Animation",
  2: "Autos & Vehicles",
  10: "Music",
  15: "Pets & Animals",
  17: "Sports",
  19: "Travel & Events",
  20: "Gaming",
  22: "People & Blogs",
  23: "Comedy",
  24: "Entertainment",
  25: "News & Politics",
  26: "Howto & Style",
  27: "Education",
  28: "Science & Technology",
  29: "Nonprofits & Activism",
};

type QuantilePrediction = {
  id: number;
  name: string;
  p10: number;
  p50: number;
  p90: number;
  empP10: number;
  empP50: number;
  empP90: number;
  nVideos: number;
};

function resolveDataHackRoot(): string | null {
  const env = process.env.DATA_HACK_ROOT;
  if (env && fs.existsSync(path.join(env, "output"))) {
    return env;
  }
  const parent = path.resolve(process.cwd(), "..");
  if (fs.existsSync(path.join(parent, "output"))) {
    return parent;
  }
  return null;
}

async function parseQuantilePredictions(
  filePath: string
): Promise<QuantilePrediction[]> {
  return new Promise((resolve, reject) => {
    const results: QuantilePrediction[] = [];
    const rl = readline.createInterface({
      input: fs.createReadStream(filePath),
      crlfDelay: Infinity,
    });

    let isHeader = true;
    rl.on("line", (line) => {
      if (isHeader) {
        isHeader = false;
        return;
      }
      const parts = line.split(",");
      if (parts.length >= 7) {
        const categoryId = parseInt(parts[0].trim(), 10);
        const nVideos = parseInt(parts[1].trim(), 10);
        const p10 = parseFloat(parts[2].trim());
        const p50 = parseFloat(parts[3].trim());
        const p90 = parseFloat(parts[4].trim());
        const empP10 = parseFloat(parts[5].trim());
        const empP50 = parseFloat(parts[6].trim());
        const empP90 = parseFloat(parts[7].trim());

        if (!isNaN(categoryId) && CATEGORY_NAMES[categoryId]) {
          results.push({
            id: categoryId,
            name: CATEGORY_NAMES[categoryId],
            p10,
            p50,
            p90,
            empP10,
            empP50,
            empP90,
            nVideos,
          });
        }
      }
    });

    rl.on("error", reject);
    rl.on("close", () => resolve(results));
  });
}

export async function GET() {
  try {
    const dataHackRoot = resolveDataHackRoot();
    if (!dataHackRoot) {
      return NextResponse.json(
        {
          error:
            "Could not find Data_hack directory. Set DATA_HACK_ROOT or run from YTBTrendingDashboard.",
        },
        { status: 500 }
      );
    }

    const filePath = path.join(
      dataHackRoot,
      "yt_trend/out/category_quantile_predictions.csv"
    );
    if (!fs.existsSync(filePath)) {
      return NextResponse.json(
        { error: `File not found: ${filePath}` },
        { status: 404 }
      );
    }

    const categories = await parseQuantilePredictions(filePath);
    return NextResponse.json({
      categories,
      metadata: {
        modelAccuracy: {
          r2: 0.878,
          hit_within_7d: 0.732,
          coverage: 0.808,
        },
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
