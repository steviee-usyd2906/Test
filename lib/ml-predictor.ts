/**
 * TypeScript port of the XGBoost prediction pipeline.
 * Replaces Python dependency for Vercel deployment compatibility.
 */
import fs from "fs";
import path from "path";
import { loadXGBoostModel, XGBoostJSONModel, XGBoostFlatModel } from "./xgboost-json";

// Category trending speed lookup (same as Python training)
const CAT_TREND_SPEED: Record<number, { p10: number; p50: number; p90: number }> = {
  25: { p10: 2.338, p50: 2.864, p90: 6.51 },
  28: { p10: 3.017, p50: 3.643, p90: 8.31 },
  2: { p10: 3.179, p50: 4.077, p90: 7.498 },
  27: { p10: 3.194, p50: 4.504, p90: 9.825 },
  23: { p10: 3.304, p50: 4.564, p90: 9.22 },
  17: { p10: 3.211, p50: 4.719, p90: 8.776 },
  19: { p10: 3.408, p50: 5.064, p90: 10.074 },
  15: { p10: 3.822, p50: 5.086, p90: 9.307 },
  29: { p10: 3.32, p50: 5.456, p90: 10.775 },
  22: { p10: 3.962, p50: 5.561, p90: 10.211 },
  26: { p10: 4.119, p50: 6.178, p90: 10.899 },
  24: { p10: 4.906, p50: 11.01, p90: 20.475 },
  10: { p10: 6.029, p50: 13.873, p90: 23.914 },
  20: { p10: 5.949, p50: 13.906, p90: 24.299 },
  1: { p10: 6.11, p50: 13.943, p90: 24.016 },
};

// Default fill values from training (used when data is missing)
const DEFAULT_FILL_VALUES = {
  channel_views: 288722463.0,
  channel_subscribers: 393000.0,
  channel_videos: 728.0,
  cat_trend_p10: 3.962,
  cat_trend_p50: 5.561,
  cat_trend_p90: 10.775,
  cat_trend_spread: 6.78,
};

// Feature hints for virality insights
const FEATURE_HINTS: Record<string, string> = {
  category_id: "Category matters: some niches trend faster than others.",
  publish_hour: "Upload hour affects early exposure and competition.",
  publish_dayofweek: "Day of week shifts audience availability and browse patterns.",
  publish_period: "Time-of-day bucket (morning/afternoon/evening/night) shapes launch dynamics.",
  is_weekend: "Weekend vs weekday posting changes competition and watch time.",
  title_length: "Title length changes click curiosity and clarity in browse surfaces.",
  title_word_count: "Word count signals specificity vs generic clickbait.",
  title_caps_ratio: "Heavy caps can read as loud or spammy—balance matters for CTR.",
  title_has_exclaim: "Exclamation can lift urgency but may reduce trust if overused.",
  title_has_question: "Questions can improve curiosity-driven clicks when matched to intent.",
  title_has_number: "Numbers/lists often improve skim-ability and perceived concreteness.",
  title_has_emoji: "Emoji can stand out in feeds but impact varies by category and brand tone.",
  channel_views: "Total channel views proxy overall reach and brand familiarity.",
  channel_subscribers: "Subscriber scale affects early velocity and recommendation ceilings.",
  channel_videos: "Catalog size changes per-video attention and posting cadence effects.",
  subs_per_video: "Subs per upload hints at channel efficiency vs breadth of output.",
  views_per_video: "Average views per upload signals typical performance level.",
  log_subscribers: "Log-scaled audience size captures diminishing returns at huge scale.",
  log_channel_views: "Log-scaled channel views stabilizes heavy-tailed channel popularity.",
  cat_trend_p10: "Category-specific fast-trending timing (lower tail) vs your publish context.",
  cat_trend_p50: "Typical category time-to-trend dynamics vs your timing signals.",
  cat_trend_p90: "Slow-trend tail for the category—helps calibrate patience vs burst potential.",
  cat_trend_spread: "How variable trending speed is within the category (risk/volatility proxy).",
};

// Evergreen and timely words for Stage 2
const EVERGREEN_WORDS = ["how", "tutorial", "guide", "tips", "review", "best", "top", "learn", "explained"];
const TIMELY_WORDS = ["breaking", "live", "update", "news", "today", "tonight", "new", "just", "official"];

// Stage 1 features in order
const STAGE1_FEATURES = [
  "category_id",
  "publish_hour",
  "publish_dayofweek",
  "publish_period",
  "is_weekend",
  "title_length",
  "title_word_count",
  "title_caps_ratio",
  "title_has_exclaim",
  "title_has_question",
  "title_has_number",
  "title_has_emoji",
  "channel_views",
  "channel_subscribers",
  "channel_videos",
  "subs_per_video",
  "views_per_video",
  "log_subscribers",
  "log_channel_views",
  "cat_trend_p10",
  "cat_trend_p50",
  "cat_trend_p90",
  "cat_trend_spread",
];

// Stage 2 features in order
const STAGE2_FEATURES = [
  "category_id",
  "days_since_publish",
  "likes_per_view",
  "comments_per_view",
  "like_comment_ratio",
  "views_per_day",
  "log_views",
  "log_vpd",
  "publish_hour",
  "publish_dayofweek",
  "publish_period",
  "is_weekend",
  "title_length",
  "title_word_count",
  "title_caps_ratio",
  "title_has_exclaim",
  "title_has_question",
  "title_has_number",
  "title_evergreen_count",
  "title_timely_count",
  "cat_trend_p10",
  "cat_trend_p50",
  "cat_trend_p90",
  "cat_trend_spread",
];

// Stage 2 class labels
const STAGE2_CLASS_LABELS = [
  "Fading — low sustained views/day after trending ends",
  "Steady — moderate post-trend velocity",
  "Thriving — high post-trend velocity",
];

// Helper functions
function hourBucket(h: number): number {
  if (h >= 6 && h < 12) return 0;
  if (h >= 12 && h < 17) return 1;
  if (h >= 17 && h < 21) return 2;
  return 3;
}

function log1p(x: number): number {
  return Math.log(1 + x);
}

function hasEmoji(str: string): boolean {
  return /[^\w\s,.\-!?;:'"()#@&/\\]/.test(str);
}

function countWords(text: string, words: string[]): number {
  const parts = text.toLowerCase().split(/\s+/);
  return words.filter((w) => parts.includes(w)).length;
}

function getCatTrendSpeed(categoryId: number): { p10: number; p50: number; p90: number; spread: number } {
  const speed = CAT_TREND_SPEED[categoryId] || CAT_TREND_SPEED[22];
  return {
    p10: speed.p10,
    p50: speed.p50,
    p90: speed.p90,
    spread: speed.p90 - speed.p10,
  };
}

function daysSincePublish(publishTime: string | Date | null): number {
  if (!publishTime) return 0;
  const pub = new Date(publishTime);
  if (isNaN(pub.getTime())) return 0;
  const now = new Date();
  return Math.max(0, (now.getTime() - pub.getTime()) / (1000 * 60 * 60 * 24));
}

function riskLabel(p: number): string {
  if (p >= 0.66) return "HIGH";
  if (p >= 0.33) return "MEDIUM";
  return "LOW";
}

// Input types
interface VideoInput {
  title?: string;
  categoryId?: string | number;
  category_id?: string | number;
  publishedAt?: string;
  publish_time?: string;
  viewCount?: number;
  views?: number;
  likeCount?: number;
  likes?: number;
  commentCount?: number;
  comments?: number;
  channelId?: string;
  channelTitle?: string;
  channel_title?: string;
  tags?: string;
  description?: string;
}

interface ChannelInput {
  viewCount?: number;
  channel_views?: number;
  subscriberCount?: number;
  channel_subscribers?: number;
  videoCount?: number;
  channel_videos?: number;
  title?: string;
}

interface PredictInput {
  video?: VideoInput;
  channel?: ChannelInput | null;
  title?: string;
  categoryId?: string | number;
  category_id?: string | number;
  publishedAt?: string;
  publish_time?: string;
  views?: number;
  likes?: number;
  comments?: number;
  channel_views?: number;
  channel_subscribers?: number;
  channel_videos?: number;
  channel_title?: string;
}

interface NormalizedRecord {
  title: string;
  category_id: number;
  publish_time: string;
  views: number;
  likes: number;
  comments: number;
  channel_views: number;
  channel_subscribers: number;
  channel_videos: number;
  channel_title: string;
  tags: string;
  description: string;
}

function normalizeRecord(input: PredictInput): NormalizedRecord {
  const video = input.video || {};
  const channel = input.channel || {};

  const title = video.title || input.title || "";
  const categoryId = Number(
    video.categoryId || video.category_id || input.categoryId || input.category_id || 24
  );
  const publishTime = video.publishedAt || video.publish_time || input.publishedAt || input.publish_time || "";
  const views = Number(video.viewCount || video.views || input.views || 0);
  const likes = Number(video.likeCount || video.likes || input.likes || 0);
  const comments = Number(video.commentCount || video.comments || input.comments || 0);

  const channelViews = Number(
    channel.viewCount || channel.channel_views || input.channel_views || DEFAULT_FILL_VALUES.channel_views
  );
  const channelSubscribers = Number(
    channel.subscriberCount || channel.channel_subscribers || input.channel_subscribers || DEFAULT_FILL_VALUES.channel_subscribers
  );
  const channelVideos = Number(
    channel.videoCount || channel.channel_videos || input.channel_videos || DEFAULT_FILL_VALUES.channel_videos
  );
  const channelTitle = channel.title || video.channelTitle || video.channel_title || input.channel_title || "unknown";

  return {
    title,
    category_id: categoryId,
    publish_time: publishTime,
    views,
    likes,
    comments,
    channel_views: channelViews,
    channel_subscribers: channelSubscribers,
    channel_videos: channelVideos,
    channel_title: channelTitle,
    tags: video.tags || "[none]",
    description: video.description || "",
  };
}

// Feature engineering for Stage 1
function engineerStage1Features(record: NormalizedRecord): number[] {
  const pubDate = record.publish_time ? new Date(record.publish_time) : new Date();
  const publishHour = isNaN(pubDate.getTime()) ? 12 : pubDate.getUTCHours();
  const publishDayOfWeek = isNaN(pubDate.getTime()) ? 0 : pubDate.getUTCDay();
  const publishPeriod = hourBucket(publishHour);
  const isWeekend = publishDayOfWeek >= 5 ? 1 : 0;

  const title = record.title || "";
  const titleLength = title.length;
  const titleWordCount = title.split(/\s+/).filter((w) => w.length > 0).length;
  const titleCapsRatio = title.length > 0 ? title.split("").filter((c) => c >= "A" && c <= "Z").length / title.length : 0;
  const titleHasExclaim = title.includes("!") ? 1 : 0;
  const titleHasQuestion = title.includes("?") ? 1 : 0;
  const titleHasNumber = /\d/.test(title) ? 1 : 0;
  const titleHasEmoji = hasEmoji(title) ? 1 : 0;

  const channelViews = record.channel_views || DEFAULT_FILL_VALUES.channel_views;
  const channelSubscribers = record.channel_subscribers || DEFAULT_FILL_VALUES.channel_subscribers;
  const channelVideos = record.channel_videos || DEFAULT_FILL_VALUES.channel_videos;
  const subsPerVideo = channelSubscribers / (channelVideos + 1);
  const viewsPerVideo = channelViews / (channelVideos + 1);
  const logSubscribers = log1p(channelSubscribers);
  const logChannelViews = log1p(channelViews);

  const catSpeed = getCatTrendSpeed(record.category_id);

  return [
    record.category_id,
    publishHour,
    publishDayOfWeek,
    publishPeriod,
    isWeekend,
    titleLength,
    titleWordCount,
    titleCapsRatio,
    titleHasExclaim,
    titleHasQuestion,
    titleHasNumber,
    titleHasEmoji,
    channelViews,
    channelSubscribers,
    channelVideos,
    subsPerVideo,
    viewsPerVideo,
    logSubscribers,
    logChannelViews,
    catSpeed.p10,
    catSpeed.p50,
    catSpeed.p90,
    catSpeed.spread,
  ];
}

// Feature engineering for Stage 2
function engineerStage2Features(record: NormalizedRecord): number[] {
  const pubDate = record.publish_time ? new Date(record.publish_time) : new Date();
  const publishHour = isNaN(pubDate.getTime()) ? 12 : pubDate.getUTCHours();
  const publishDayOfWeek = isNaN(pubDate.getTime()) ? 0 : pubDate.getUTCDay();
  const publishPeriod = hourBucket(publishHour);
  const isWeekend = publishDayOfWeek >= 5 ? 1 : 0;

  const dsp = daysSincePublish(record.publish_time);
  const views = record.views || 0;
  const likes = record.likes || 0;
  const comments = record.comments || 0;

  const likesPerView = likes / (views + 1);
  const commentsPerView = comments / (views + 1);
  const likeCommentRatio = likes / (comments + 1);
  const viewsPerDay = views / (dsp + 1);
  const logViews = log1p(views);
  const logVpd = log1p(viewsPerDay);

  const title = record.title || "";
  const titleLength = title.length;
  const titleWordCount = title.split(/\s+/).filter((w) => w.length > 0).length;
  const titleCapsRatio = title.length > 0 ? title.split("").filter((c) => c >= "A" && c <= "Z").length / title.length : 0;
  const titleHasExclaim = title.includes("!") ? 1 : 0;
  const titleHasQuestion = title.includes("?") ? 1 : 0;
  const titleHasNumber = /\d/.test(title) ? 1 : 0;
  const titleEvergreenCount = countWords(title, EVERGREEN_WORDS);
  const titleTimelyCount = countWords(title, TIMELY_WORDS);

  const catSpeed = getCatTrendSpeed(record.category_id);

  return [
    record.category_id,
    dsp,
    likesPerView,
    commentsPerView,
    likeCommentRatio,
    viewsPerDay,
    logViews,
    logVpd,
    publishHour,
    publishDayOfWeek,
    publishPeriod,
    isWeekend,
    titleLength,
    titleWordCount,
    titleCapsRatio,
    titleHasExclaim,
    titleHasQuestion,
    titleHasNumber,
    titleEvergreenCount,
    titleTimelyCount,
    catSpeed.p10,
    catSpeed.p50,
    catSpeed.p90,
    catSpeed.spread,
  ];
}

// Model cache
let stage1Model: XGBoostJSONModel | XGBoostFlatModel | null = null;
let stage2Model: XGBoostJSONModel | XGBoostFlatModel | null = null;
let stage1Meta: Record<string, unknown> | null = null;
let stage2Meta: Record<string, unknown> | null = null;

function getModelsDir(): string {
  const candidates = [
    path.join(process.cwd(), "models"),
    path.join(process.cwd(), "..", "models"),
    "/var/task/models",
    path.resolve(__dirname, "..", "models"),
  ];

  for (const dir of candidates) {
    if (fs.existsSync(path.join(dir, "stage1_xgb.json"))) {
      return dir;
    }
  }

  return path.join(process.cwd(), "models");
}

function loadStage1Model(): { model: XGBoostJSONModel | XGBoostFlatModel; meta: Record<string, unknown> } {
  if (stage1Model && stage1Meta) {
    return { model: stage1Model, meta: stage1Meta };
  }

  const modelsDir = getModelsDir();
  const modelPath = path.join(modelsDir, "stage1_xgb.json");
  const metaPath = path.join(modelsDir, "stage1_meta.json");

  if (!fs.existsSync(modelPath)) {
    throw new Error(`Stage 1 model not found at ${modelPath}`);
  }
  if (!fs.existsSync(metaPath)) {
    throw new Error(`Stage 1 meta not found at ${metaPath}`);
  }

  const modelJson = JSON.parse(fs.readFileSync(modelPath, "utf-8"));
  const meta = JSON.parse(fs.readFileSync(metaPath, "utf-8"));

  stage1Model = loadXGBoostModel(modelJson);
  stage1Meta = meta;

  return { model: stage1Model, meta: stage1Meta };
}

function loadStage2Model(): { model: XGBoostJSONModel | XGBoostFlatModel; meta: Record<string, unknown> } {
  if (stage2Model && stage2Meta) {
    return { model: stage2Model, meta: stage2Meta };
  }

  const modelsDir = getModelsDir();
  const modelPath = path.join(modelsDir, "stage2_xgb.json");
  const metaPath = path.join(modelsDir, "stage2_meta.json");

  if (!fs.existsSync(modelPath)) {
    throw new Error(`Stage 2 model not found at ${modelPath}`);
  }
  if (!fs.existsSync(metaPath)) {
    throw new Error(`Stage 2 meta not found at ${metaPath}`);
  }

  const modelJson = JSON.parse(fs.readFileSync(modelPath, "utf-8"));
  const meta = JSON.parse(fs.readFileSync(metaPath, "utf-8"));

  stage2Model = loadXGBoostModel(modelJson);
  stage2Meta = meta;

  return { model: stage2Model, meta: stage2Meta };
}

// Result types
interface Stage1Result {
  probability: number;
  probability_percent: number;
  risk_label: string;
  predicted_trendy: boolean;
  predicted_class: number;
  features_used: string[];
}

interface Stage2Result {
  vpd_class: number;
  vpd_class_label: string;
  class_probabilities: number[];
  predicted_class_confidence: number;
}

interface ViralityHint {
  feature: string;
  importance: number;
  hint: string;
}

interface PredictionResult {
  stage1: Stage1Result;
  stage2: Stage2Result | null;
  virality_hints: ViralityHint[] | null;
}

export function predictStage1(input: PredictInput): Stage1Result {
  const record = normalizeRecord(input);
  const features = engineerStage1Features(record);

  const { model } = loadStage1Model();

  // For binary classification
  const probability = model.predictProba(features);

  return {
    probability: Math.round(probability * 1000000) / 1000000,
    probability_percent: Math.round(probability * 100),
    risk_label: riskLabel(probability),
    predicted_trendy: probability >= 0.5,
    predicted_class: probability >= 0.5 ? 1 : 0,
    features_used: STAGE1_FEATURES,
  };
}

export function predictStage2(input: PredictInput): Stage2Result {
  const record = normalizeRecord(input);
  const features = engineerStage2Features(record);

  const { model } = loadStage2Model();

  // For multi-class classification
  const probabilities = model.predictProbaMulti(features);
  const predictedClass = probabilities.indexOf(Math.max(...probabilities));

  return {
    vpd_class: predictedClass,
    vpd_class_label: STAGE2_CLASS_LABELS[predictedClass] || String(predictedClass),
    class_probabilities: probabilities.map((p) => Math.round(p * 1000000) / 1000000),
    predicted_class_confidence: Math.round(Math.max(...probabilities) * 1000000) / 1000000,
  };
}

function getViralityHints(meta: Record<string, unknown>): ViralityHint[] {
  const cached = meta.virality_top_features as ViralityHint[] | undefined;
  if (cached && Array.isArray(cached) && cached.length >= 5) {
    return cached.slice(0, 5).map((h) => ({
      feature: h.feature,
      importance: h.importance,
      hint: h.hint || FEATURE_HINTS[h.feature] || `\`${h.feature}\` is one of the strongest global drivers.`,
    }));
  }

  return [
    { feature: "publish_dayofweek", importance: 0.091, hint: FEATURE_HINTS.publish_dayofweek },
    { feature: "cat_trend_spread", importance: 0.086, hint: FEATURE_HINTS.cat_trend_spread },
    { feature: "cat_trend_p50", importance: 0.073, hint: FEATURE_HINTS.cat_trend_p50 },
    { feature: "cat_trend_p90", importance: 0.063, hint: FEATURE_HINTS.cat_trend_p90 },
    { feature: "channel_videos", importance: 0.055, hint: FEATURE_HINTS.channel_videos },
  ];
}

export function predictTrendingPipeline(input: PredictInput): PredictionResult {
  const stage1 = predictStage1(input);

  const result: PredictionResult = {
    stage1,
    stage2: null,
    virality_hints: null,
  };

  if (!stage1.predicted_trendy) {
    const { meta } = loadStage1Model();
    result.virality_hints = getViralityHints(meta);
    return result;
  }

  try {
    result.stage2 = predictStage2(input);
  } catch (error) {
    console.error("Stage 2 prediction failed:", error);
  }

  return result;
}
