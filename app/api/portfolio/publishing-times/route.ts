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

const DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

type TimeFrequency = {
  value: number | string;
  label: string;
  count: number;
};

type PublishingTimesData = {
  categoryId: number;
  categoryName: string;
  topMonths: TimeFrequency[];
  topDays: TimeFrequency[];
  topHours: TimeFrequency[];
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

function extractTimeComponents(publishTime: string) {
  const dt = new Date(publishTime);
  if (isNaN(dt.getTime())) return null;
  return {
    month: dt.getUTCMonth() + 1,
    dow: dt.getUTCDay(),
    hour: dt.getUTCHours(),
  };
}

function getTopN(
  map: Map<number, number>,
  n: number,
  labelFn: (val: number) => string
): TimeFrequency[] {
  return Array.from(map.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([value, count]) => ({
      value,
      label: labelFn(value),
      count,
    }));
}

async function parsePublishingTimes(
  filePath: string
): Promise<PublishingTimesData[]> {
  return new Promise((resolve, reject) => {
    const categoryStats = new Map<
      number,
      {
        months: Map<number, number>;
        dows: Map<number, number>;
        hours: Map<number, number>;
      }
    >();

    const rl = readline.createInterface({
      input: fs.createReadStream(filePath),
      crlfDelay: Infinity,
    });

    let isHeader = true;
    let categoryIdIdx = -1,
      publishTimeIdx = -1,
      trendyIdx = -1;

    rl.on("line", (line) => {
      if (isHeader) {
        const headers = line.split(",");
        categoryIdIdx = headers.indexOf("category_id");
        publishTimeIdx = headers.indexOf("publish_time");
        trendyIdx = headers.indexOf("trendy");
        isHeader = false;
        return;
      }

      const parts = line.split(",");
      if (
        categoryIdIdx < 0 ||
        publishTimeIdx < 0 ||
        trendyIdx < 0 ||
        parts.length <= Math.max(categoryIdIdx, publishTimeIdx, trendyIdx)
      ) {
        return;
      }

      const trendy = parseInt(parts[trendyIdx].trim(), 10);
      if (trendy !== 1) return;

      const categoryId = parseInt(parts[categoryIdIdx].trim(), 10);
      const publishTime = parts[publishTimeIdx].trim();

      const times = extractTimeComponents(publishTime);
      if (!times || !CATEGORY_NAMES[categoryId]) return;

      if (!categoryStats.has(categoryId)) {
        categoryStats.set(categoryId, {
          months: new Map(),
          dows: new Map(),
          hours: new Map(),
        });
      }

      const stats = categoryStats.get(categoryId)!;
      stats.months.set(times.month, (stats.months.get(times.month) ?? 0) + 1);
      stats.dows.set(times.dow, (stats.dows.get(times.dow) ?? 0) + 1);
      stats.hours.set(times.hour, (stats.hours.get(times.hour) ?? 0) + 1);
    });

    rl.on("error", reject);
    rl.on("close", () => {
      const results: PublishingTimesData[] = Array.from(
        categoryStats.entries()
      )
        .map(([categoryId, stats]) => ({
          categoryId,
          categoryName: CATEGORY_NAMES[categoryId],
          topMonths: getTopN(stats.months, 3, (m) => MONTH_NAMES[m - 1]),
          topDays: getTopN(stats.dows, 3, (d) => DAY_NAMES[d]),
          topHours: getTopN(stats.hours, 3, (h) => `${h}:00`),
        }))
        .sort((a, b) => a.categoryId - b.categoryId);
      resolve(results);
    });
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

    const filePath = path.join(dataHackRoot, "stage2_training_data.csv");
    if (!fs.existsSync(filePath)) {
      return NextResponse.json(
        { error: `File not found: ${filePath}` },
        { status: 404 }
      );
    }

    const categories = await parsePublishingTimes(filePath);
    return NextResponse.json({ categories });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
