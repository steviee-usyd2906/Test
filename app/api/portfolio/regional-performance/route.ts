import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import readline from "readline";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

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

const COUNTRIES = ["US", "GB", "CA", "DE", "FR", "IN", "JP", "KR", "MX", "RU", "BR"];

type CategoryPerformance = {
  categoryId: number;
  categoryName: string;
  share: number;
  count: number;
};

type RegionalData = {
  countryCode: string;
  categories: CategoryPerformance[];
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

async function parseCategoryDistribution(
  filePath: string
): Promise<Map<number, number>> {
  return new Promise((resolve, reject) => {
    const categoryCount = new Map<number, number>();
    const rl = readline.createInterface({
      input: fs.createReadStream(filePath),
      crlfDelay: Infinity,
    });

    let isHeader = true;
    let categoryIdIdx = -1,
      trendyIdx = -1;

    rl.on("line", (line) => {
      if (isHeader) {
        const headers = line.split(",");
        categoryIdIdx = headers.indexOf("category_id");
        trendyIdx = headers.indexOf("trendy");
        isHeader = false;
        return;
      }

      const parts = line.split(",");
      if (categoryIdIdx < 0 || trendyIdx < 0 || parts.length <= Math.max(categoryIdIdx, trendyIdx)) {
        return;
      }

      const trendy = parseInt(parts[trendyIdx].trim(), 10);
      if (trendy !== 1) return;

      const categoryId = parseInt(parts[categoryIdIdx].trim(), 10);
      if (!CATEGORY_NAMES[categoryId]) return;

      categoryCount.set(categoryId, (categoryCount.get(categoryId) ?? 0) + 1);
    });

    rl.on("error", reject);
    rl.on("close", () => resolve(categoryCount));
  });
}

async function parseRegionalPerformance(dataHackRoot: string): Promise<RegionalData[]> {
  const results: RegionalData[] = [];

  for (const country of COUNTRIES) {
    const filePath = path.join(dataHackRoot, `output/${country}_final.csv`);
    if (!fs.existsSync(filePath)) {
      continue;
    }

    const categoryCount = await parseCategoryDistribution(filePath);
    const total = Array.from(categoryCount.values()).reduce((a, b) => a + b, 0);

    const categories: CategoryPerformance[] = Array.from(categoryCount.entries())
      .map(([catId, count]) => ({
        categoryId: catId,
        categoryName: CATEGORY_NAMES[catId],
        count,
        share: total > 0 ? Math.round((count / total) * 10000) / 100 : 0,
      }))
      .sort((a, b) => b.count - a.count);

    results.push({
      countryCode: country,
      categories,
    });
  }

  return results;
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

    const regions = await parseRegionalPerformance(dataHackRoot);
    return NextResponse.json({ regions });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
