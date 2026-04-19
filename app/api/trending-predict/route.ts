import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 120;

/** Walk up from `startDir` looking for `scripts/run_predict_pipeline.py` (handles odd `cwd`). */
function findScriptsDirFromWalk(startDir: string, maxDepth = 10): string | null {
  let cur = path.resolve(startDir);
  for (let i = 0; i < maxDepth; i++) {
    const scriptFile = path.join(cur, "scripts", "run_predict_pipeline.py");
    if (fs.existsSync(scriptFile)) {
      return path.join(cur, "scripts");
    }
    const parent = path.dirname(cur);
    if (parent === cur) break;
    cur = parent;
  }
  return null;
}

function resolveScriptDir(): string | null {
  const cwd = process.cwd();
  const candidates = [
    process.env.DASHBOARD_ROOT
      ? path.join(process.env.DASHBOARD_ROOT, "scripts")
      : "",
    path.join(cwd, "scripts"),
    findScriptsDirFromWalk(cwd),
    path.join(cwd, "YTBTrendingDashboard", "scripts"),
    process.env.DATA_HACK_ROOT
      ? path.join(process.env.DATA_HACK_ROOT, "YTBTrendingDashboard", "scripts")
      : "",
  ].filter((p): p is string => Boolean(p));

  const seen = new Set<string>();
  for (const dir of candidates) {
    if (seen.has(dir)) continue;
    seen.add(dir);
    if (fs.existsSync(path.join(dir, "run_predict_pipeline.py"))) {
      return dir;
    }
  }
  return null;
}

type Body = {
  video?: Record<string, unknown>;
  channel?: Record<string, unknown> | null;
};

function buildPythonAttempts(script: string): [string, string[]][] {
  const out: [string, string[]][] = [];
  if (process.env.PYTHON_PATH?.trim()) {
    out.push([process.env.PYTHON_PATH.trim(), [script]]);
  }
  if (process.platform === "win32") {
    out.push(["python", [script]]);
    out.push(["py", ["-3", script]]);
    out.push(["python3", [script]]);
  } else {
    out.push(["python3", [script]]);
    out.push(["python", [script]]);
  }
  return out;
}

/** Parse JSON from predictor stdout; tolerate leading warnings / log lines from libraries. */
function parsePredictorStdout(raw: string): unknown | null {
  const t = raw.trim();
  if (!t) return null;
  try {
    return JSON.parse(t);
  } catch {
    const first = t.indexOf("{");
    const last = t.lastIndexOf("}");
    if (first >= 0 && last > first) {
      try {
        return JSON.parse(t.slice(first, last + 1));
      } catch {
        return null;
      }
    }
    return null;
  }
}

function runPredictor(
  cmd: string,
  args: string[],
  cwd: string,
  stdinBody: string,
  extraEnv: NodeJS.ProcessEnv
): Promise<{ code: number; stdout: string; stderr: string; spawnErr?: string }> {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, {
      cwd,
      stdio: ["pipe", "pipe", "pipe"],
      env: extraEnv,
    });
    let out = "";
    let err = "";
    let spawnErr: string | undefined;
    child.stdout?.on("data", (c: Buffer) => {
      out += c.toString("utf8");
    });
    child.stderr?.on("data", (c: Buffer) => {
      err += c.toString("utf8");
    });
    child.on("error", (e: NodeJS.ErrnoException) => {
      spawnErr = e.code === "ENOENT" ? `${cmd}: not found on PATH` : e.message;
    });
    child.on("close", (exitCode) => {
      resolve({ stdout: out, stderr: err, code: exitCode ?? 0, spawnErr });
    });
    child.stdin?.write(stdinBody, "utf8");
    child.stdin?.end();
  });
}

export async function POST(req: Request) {
  const scriptDir = resolveScriptDir();
  if (!scriptDir) {
    return NextResponse.json(
      {
        error: "Could not find scripts/run_predict_pipeline.py.",
        hint:
          "Run `npm run dev` from the YTBTrendingDashboard folder, or set DASHBOARD_ROOT to that folder. cwd was: " +
          process.cwd(),
      },
      { status: 500 }
    );
  }

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

  const payload = { video: body.video, channel };

  const script = path.join(scriptDir, "run_predict_pipeline.py");
  const mlDir = path.join(scriptDir, "..", "ml");
  const baseEnv = {
    ...process.env,
    PYTHONUTF8: "1",
    PYTHONUNBUFFERED: "1",
    PYTHONPATH: [mlDir, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
  };

  const stdinJson = JSON.stringify(payload);
  const attempts = buildPythonAttempts(script);
  const tried: string[] = [];
  let lastStdout = "";
  let lastStderr = "";
  let lastSpawnErr: string | undefined;

  const modelHint =
    "Ensure models/stage1_xgb.json + stage1_meta.json (+ stage2) exist under models/. From repo root: pip install -r requirements.txt && python ml/stage_1_xg_boost.py && python ml/stage_2.py";

  for (const [cmd, args] of attempts) {
    tried.push([cmd, ...args].join(" "));
    const { code, stdout, stderr, spawnErr } = await runPredictor(
      cmd,
      args,
      scriptDir,
      stdinJson,
      baseEnv
    );
    lastStdout = stdout;
    lastStderr = stderr;
    lastSpawnErr = spawnErr;

    const text = stdout.trim();
    if (text) {
      const parsed = parsePredictorStdout(text);
      if (parsed !== null && typeof parsed === "object") {
        if ("error" in parsed) {
          return NextResponse.json(
            { ...(parsed as Record<string, unknown>), hint: modelHint },
            { status: 502 }
          );
        }
        if (code === 0) {
          return NextResponse.json(parsed);
        }
      } else if (code === 0) {
        return NextResponse.json(
          {
            error: "Invalid JSON from predictor",
            detail: text.slice(0, 600),
            hint: modelHint,
          },
          { status: 502 }
        );
      }
    }

    if (spawnErr && !text) {
      continue;
    }
  }

  const detail = [lastSpawnErr, lastStderr, lastStdout].filter(Boolean).join("\n").slice(0, 800);
  return NextResponse.json(
    {
      error: "Prediction failed (Python did not return a valid result).",
      detail,
      hint: `Tried: ${tried.join(" | ")}. ${modelHint} Set PYTHON_PATH if python is not on PATH.`,
    },
    { status: 502 }
  );
}
