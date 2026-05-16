/**
 * gen_presentation.js
 * -------------------
 * Generates the 15-minute final project PPTX presentation.
 * Chapman University color scheme: Red #CC0000, Gold #FFB81C
 *
 * Run: node reports/gen_presentation.js
 * Output: reports/PPE_Final_Presentation.pptx
 */

"use strict";

const pptxgen = require("pptxgenjs");
const path    = require("path");
const fs      = require("fs");

// ── Paths ─────────────────────────────────────────────────────────────────────
const BASE   = path.resolve(__dirname, "..");
const PLOTS  = path.join(BASE, "results", "plots");
const MODELS = path.join(BASE, "results", "models");

function img(dir, name) {
  const p = path.join(dir, name);
  return fs.existsSync(p) ? p : null;
}

// ── Chapman Color Palette ─────────────────────────────────────────────────────
const RED       = "CC0000";
const DARK_RED  = "8B0000";
const GOLD      = "FFB81C";
const DARK_GOLD = "C88F00";
const WHITE     = "FFFFFF";
const DARK      = "1E1E1E";
const GRAY      = "5A5A5A";
const LGRAY     = "F4F4F4";
const MGRAY     = "CCCCCC";
const BGWHITE   = "FDFCFB";

// Category badge colors
const CAT_A = "6B21A8";  // purple  — custom
const CAT_B = "1D4ED8";  // blue    — from scratch
const CAT_C = "B45309";  // amber   — zero-shot
const CAT_D = "15803D";  // green   — fine-tuned

// ── Helpers ───────────────────────────────────────────────────────────────────
function mkShadow() {
  return { type: "outer", color: "000000", blur: 5, offset: 2, angle: 135, opacity: 0.13 };
}

function addFooter(slide, pres) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.45, w: 10, h: 0.175,
    fill: { color: RED }, line: { color: RED }
  });
  slide.addText("PPE Detection & Segmentation  —  Final Project", {
    x: 0.25, y: 5.45, w: 6.5, h: 0.175,
    fontSize: 7.5, color: WHITE, valign: "middle", margin: 0
  });
  slide.addText("Chapman University  |  AI / ML", {
    x: 6.8, y: 5.45, w: 3, h: 0.175,
    fontSize: 7.5, color: WHITE, align: "right", valign: "middle", margin: 0
  });
}

function addRedHeader(slide, pres, title, subtitle) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.62,
    fill: { color: RED }, line: { color: RED }
  });
  slide.addText(title, {
    x: 0.3, y: 0, w: 9.4, h: 0.62,
    fontSize: 22, bold: true, color: WHITE, valign: "middle", margin: 0
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.3, y: 0.65, w: 9.4, h: 0.32,
      fontSize: 11, color: GRAY, italic: true, margin: 0
    });
  }
}

function addCatBadge(slide, x, y, letter, color) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.36, h: 0.25,
    fill: { color }, line: { color }, shadow: mkShadow()
  });
  slide.addText(`(${letter})`, {
    x, y, w: 0.36, h: 0.25,
    fontSize: 9, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
  });
}

function statBox(slide, pres, x, y, w, h, value, label, accent) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: WHITE }, line: { color: accent, width: 2 }, shadow: mkShadow()
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h: 0.06,
    fill: { color: accent }, line: { color: accent }
  });
  slide.addText(value, {
    x, y: y + 0.1, w, h: h * 0.55,
    fontSize: 36, bold: true, color: accent, align: "center", valign: "middle", margin: 0
  });
  slide.addText(label, {
    x, y: y + h * 0.58, w, h: h * 0.38,
    fontSize: 9, color: GRAY, align: "center", valign: "top", margin: 0
  });
}

// ── Presentation Setup ────────────────────────────────────────────────────────
const pres = new pptxgen();
pres.layout  = "LAYOUT_16x9";
pres.title   = "PPE Detection & Segmentation — Final Project";
pres.author  = "Chapman University";
pres.subject = "AI/ML Final Project";

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  // Dark red background
  s.background = { color: DARK_RED };

  // Gold accent bar left
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.18, h: 5.625,
    fill: { color: GOLD }, line: { color: GOLD }
  });

  // Chapman name at top
  s.addText("CHAPMAN UNIVERSITY", {
    x: 0.4, y: 0.3, w: 9.2, h: 0.35,
    fontSize: 12, bold: true, color: GOLD, charSpacing: 4, margin: 0
  });
  s.addText("Fowler School of Engineering", {
    x: 0.4, y: 0.62, w: 9.2, h: 0.28,
    fontSize: 10, color: "E8CCCC", italic: true, margin: 0
  });

  // Divider
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.05, w: 5.5, h: 0.04,
    fill: { color: GOLD }, line: { color: GOLD }
  });

  // Main title
  s.addText("PPE Detection &\nSegmentation", {
    x: 0.4, y: 1.2, w: 9.2, h: 1.9,
    fontSize: 50, bold: true, color: WHITE, valign: "middle", margin: 0
  });

  // Subtitle
  s.addText("A Multi-Stage Computer Vision Pipeline\nfor Industrial Safety Compliance", {
    x: 0.4, y: 3.2, w: 7, h: 0.85,
    fontSize: 16, color: "F0CCCC", italic: true, margin: 0
  });

  // Gold divider
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 4.15, w: 5.5, h: 0.04,
    fill: { color: GOLD }, line: { color: GOLD }
  });

  // Date and course
  s.addText("AI / Machine Learning  |  Final Project Presentation  |  2024", {
    x: 0.4, y: 4.3, w: 9.2, h: 0.28,
    fontSize: 10, color: "DDBBBB", margin: 0
  });

  // Helmet icon approximation (red shape)
  s.addShape(pres.shapes.OVAL, {
    x: 7.6, y: 1.5, w: 2.0, h: 2.0,
    fill: { color: "A00000", transparency: 30 }, line: { color: GOLD, width: 2 }
  });
  s.addText("🦺", { x: 7.6, y: 1.5, w: 2.0, h: 2.0, fontSize: 64, align: "center", valign: "middle" });
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — Agenda
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "Presentation Agenda", "15-minute final project overview");

  // Two-column grid of agenda items
  const items = [
    ["01", "The Problem",           "PPE compliance in industrial settings"],
    ["02", "State of the Art",      "Existing approaches & benchmarks"],
    ["03", "Why It Matters",        "Safety impact & automation value"],
    ["04", "Datasets",              "MinhNKB (A2) and keremberke (A3)"],
    ["05", "Exploratory Analysis",  "Class distributions & data insights"],
    ["06", "Model Framework",       "Four evaluation categories (a)–(d)"],
    ["07", "Model Architectures",   "CNN, UNet, ViT, DeepLabV3+, YOLO"],
    ["08", "Results",               "Accuracy, mIoU, mAP, Grad-CAM"],
    ["09", "Meaningful Findings",   "Fine-tuning impact & task transfer"],
    ["10", "Future Work & Refs",    "Next steps and citations"],
  ];

  const colX = [0.35, 5.2];
  items.forEach(([num, title, desc], i) => {
    const col = i < 5 ? 0 : 1;
    const row = i % 5;
    const x = colX[col];
    const y = 1.05 + row * 0.82;

    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.6, h: 0.68,
      fill: { color: WHITE }, line: { color: MGRAY, width: 1 }, shadow: mkShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.42, h: 0.68,
      fill: { color: RED }, line: { color: RED }
    });
    s.addText(num, {
      x, y, w: 0.42, h: 0.68,
      fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
    });
    s.addText(title, {
      x: x + 0.5, y: y + 0.04, w: 4.0, h: 0.28,
      fontSize: 11, bold: true, color: DARK, margin: 0
    });
    s.addText(desc, {
      x: x + 0.5, y: y + 0.32, w: 4.0, h: 0.28,
      fontSize: 9, color: GRAY, margin: 0
    });
  });

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — The Problem
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "The Problem", "PPE compliance in high-risk industrial environments");

  // Problem statement
  s.addText("Personal Protective Equipment (PPE) is the last line of defence against workplace injury. In sectors such as construction, manufacturing, and chemical processing, non-compliance routinely causes preventable fatalities.", {
    x: 0.35, y: 0.95, w: 9.3, h: 0.7,
    fontSize: 12.5, color: DARK, margin: 0
  });

  // Two-column: challenges + target pipeline
  const challenges = [
    "Manual spot-checks are infrequent and error-prone",
    "Large sites make continuous human monitoring impossible",
    "Workers may self-report or be coached before inspections",
    "Existing CCTV footage is underutilised for compliance",
    "No granular per-body-part PPE feedback at scale",
  ];

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 1.72, w: 4.55, h: 3.4,
    fill: { color: LGRAY }, line: { color: MGRAY, width: 1 }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 1.72, w: 4.55, h: 0.36,
    fill: { color: RED }, line: { color: RED }
  });
  s.addText("Challenges with Manual Monitoring", {
    x: 0.35, y: 1.72, w: 4.55, h: 0.36,
    fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
  });

  challenges.forEach((c, i) => {
    s.addShape(pres.shapes.OVAL, {
      x: 0.55, y: 2.2 + i * 0.54, w: 0.22, h: 0.22,
      fill: { color: RED }, line: { color: RED }
    });
    s.addText(c, {
      x: 0.85, y: 2.15 + i * 0.54, w: 3.85, h: 0.44,
      fontSize: 10, color: DARK, valign: "middle", margin: 0
    });
  });

  // Right: our solution
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 1.72, w: 4.55, h: 3.4,
    fill: { color: WHITE }, line: { color: GOLD, width: 2 }, shadow: mkShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 1.72, w: 4.55, h: 0.36,
    fill: { color: GOLD }, line: { color: GOLD }
  });
  s.addText("Our Two-Stage Pipeline", {
    x: 5.1, y: 1.72, w: 4.55, h: 0.36,
    fontSize: 11, bold: true, color: DARK, align: "center", valign: "middle", margin: 0
  });

  const stages = [
    ["Stage 1 — Detect", "YOLOv8n fine-tuned on worker images identifies persons in each frame"],
    ["Stage 2 — Classify", "CNN crops + ensemble ML (SVM/RF/ET/GBM) classify PPE per person"],
    ["Stage 3 — Segment", "DeepLabV3+ & YOLOv8n-seg map exact PPE pixels / instances"],
    ["Output", "Per-frame compliance report: helmet, gloves, mask, goggles, shoes"],
  ];
  stages.forEach(([title, desc], i) => {
    s.addText(title, {
      x: 5.25, y: 2.2 + i * 0.7, w: 4.2, h: 0.26,
      fontSize: 10.5, bold: true, color: RED, margin: 0
    });
    s.addText(desc, {
      x: 5.25, y: 2.44 + i * 0.7, w: 4.2, h: 0.36,
      fontSize: 9.5, color: GRAY, margin: 0
    });
  });

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — Why It Matters
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "Why It Matters", "The economic and human cost of PPE non-compliance");

  // Big stat callouts
  const stats = [
    ["340M+",  "Occupational accidents\nper year globally\n(ILO)"],
    ["$170B",  "Annual economic cost\nof workplace injuries\nin the U.S. (NSC)"],
    ["29%",    "Of fatal injuries involve\nfailure to wear proper\nPPE (OSHA)"],
    ["<2 min", "Time to inspect 30+\nworkers with automated\nvideo analytics"],
  ];
  const xPositions = [0.3, 2.78, 5.26, 7.74];
  const accentColors = [RED, DARK_RED, GOLD, DARK_GOLD];
  stats.forEach(([val, lbl], i) => {
    statBox(s, pres, xPositions[i], 0.95, 2.2, 1.75, val, lbl, accentColors[i]);
  });

  // Why CV matters
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 2.9, w: 9.4, h: 2.2,
    fill: { color: "FFF8E8" }, line: { color: GOLD, width: 1.5 }, shadow: mkShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 2.9, w: 0.1, h: 2.2,
    fill: { color: GOLD }, line: { color: GOLD }
  });
  s.addText("Why Computer Vision?", {
    x: 0.55, y: 2.95, w: 9.0, h: 0.34,
    fontSize: 13, bold: true, color: DARK_RED, margin: 0
  });

  const reasons = [
    "Runs continuously — no shift changes, no fatigue, no missed hours",
    "Consistent — the same image gets the same result every time, regardless of who is watching",
    "Scales without extra cost — one model covers hundreds of cameras simultaneously",
    "Flags violations the moment they appear, not at the next safety audit",
    "Reads individual PPE items — helmet, gloves, mask, goggles, shoes — separately per worker",
  ];
  reasons.forEach((r, i) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.55, y: 3.35 + i * 0.35, w: 0.18, h: 0.18,
      fill: { color: GOLD }, line: { color: GOLD }
    });
    s.addText(r, {
      x: 0.82, y: 3.3 + i * 0.35, w: 8.7, h: 0.32,
      fontSize: 10.5, color: DARK, valign: "middle", margin: 0
    });
  });

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — State of the Art
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "State of the Art", "Key techniques in PPE detection and visual compliance monitoring");

  const cats = [
    {
      title: "Object Detection",
      color: "1D4ED8",
      items: [
        "YOLO family (v5–v11): single-stage, real-time; mAP50 ≈ 0.85 on PPE benchmarks",
        "Faster R-CNN: two-stage, higher precision at lower speed",
        "RT-DETR: transformer-based detection, SOTA on COCO",
      ]
    },
    {
      title: "Semantic Segmentation",
      color: "15803D",
      items: [
        "DeepLabV3+ (Chen 2018): ASPP + encoder-decoder; standard reference architecture",
        "UNet (Ronneberger 2015): skip connections; widely used in medical & safety imaging",
        "SegFormer: transformer backbone, SOTA on ADE20K (51.0 mIoU)",
      ]
    },
    {
      title: "Instance Segmentation",
      color: "6B21A8",
      items: [
        "Mask R-CNN (He 2017): pioneered instance segmentation; heavy compute",
        "YOLOv8n-seg: lightweight, joint box + mask head; near real-time",
        "SAM / SAM2 (Kirillov 2023, Ravi 2024): promptable, zero-shot mask generation",
      ]
    },
    {
      title: "Classification & Transfer",
      color: "B45309",
      items: [
        "ViT-B/16 (Dosovitskiy 2021): attention-based; strong with fine-tuning on small sets",
        "EfficientNet / MobileNet: mobile-friendly CNNs; accuracy/speed balance",
        "Custom CNNs (PPENet): task-specific, lightweight (<500K params)",
      ]
    },
  ];

  const xPos = [0.3, 5.1, 0.3, 5.1];
  const yPos = [1.05, 1.05, 3.05, 3.05];

  cats.forEach((cat, i) => {
    const x = xPos[i], y = yPos[i];
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.55, h: 1.82,
      fill: { color: WHITE }, line: { color: MGRAY, width: 1 }, shadow: mkShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.55, h: 0.32,
      fill: { color: cat.color }, line: { color: cat.color }
    });
    s.addText(cat.title, {
      x, y, w: 4.55, h: 0.32,
      fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
    });
    cat.items.forEach((item, j) => {
      s.addShape(pres.shapes.OVAL, {
        x: x + 0.15, y: y + 0.42 + j * 0.45, w: 0.14, h: 0.14,
        fill: { color: cat.color }, line: { color: cat.color }
      });
      s.addText(item, {
        x: x + 0.37, y: y + 0.36 + j * 0.45, w: 4.0, h: 0.42,
        fontSize: 9, color: DARK, valign: "middle", margin: 0
      });
    });
  });

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — Datasets
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "Datasets", "Two complementary PPE datasets for classification and segmentation");

  // Dataset A — MinhNKB
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 1.0, w: 4.55, h: 4.15,
    fill: { color: WHITE }, line: { color: MGRAY, width: 1 }, shadow: mkShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 1.0, w: 4.55, h: 0.38,
    fill: { color: RED }, line: { color: RED }
  });
  s.addText("Assignment 2 — MinhNKB Classification", {
    x: 0.3, y: 1.0, w: 4.55, h: 0.38,
    fontSize: 10.5, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
  });

  const a2info = [
    ["Task", "Image-level classification"],
    ["Source", "Kaggle / MinhNKB"],
    ["Classes", "helmet, safety_vest, full_ppe,\npartial_ppe, no_ppe  (5)"],
    ["Format", "JPEG crops, no spatial labels"],
    ["Split", "~3,000 train / ~750 val / ~750 test"],
    ["Limitation", "No pixel or bounding box annotations\n→ insufficient for segmentation"],
  ];
  a2info.forEach(([k, v], i) => {
    s.addText(k + ":", {
      x: 0.45, y: 1.5 + i * 0.56, w: 1.2, h: 0.4,
      fontSize: 9.5, bold: true, color: RED, valign: "top", margin: 0
    });
    s.addText(v, {
      x: 1.65, y: 1.5 + i * 0.56, w: 3.0, h: 0.4,
      fontSize: 9.5, color: DARK, valign: "top", margin: 0
    });
  });

  // Dataset B — keremberke
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 1.0, w: 4.55, h: 4.15,
    fill: { color: WHITE }, line: { color: GOLD, width: 2 }, shadow: mkShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 1.0, w: 4.55, h: 0.38,
    fill: { color: GOLD }, line: { color: GOLD }
  });
  s.addText("Assignment 3 — keremberke Segmentation", {
    x: 5.1, y: 1.0, w: 4.55, h: 0.38,
    fontSize: 10.5, bold: true, color: DARK, align: "center", valign: "middle", margin: 0
  });

  const a3info = [
    ["Task", "Semantic & instance segmentation"],
    ["Source", "Roboflow / keremberke"],
    ["Classes", "background, helmet, no_helmet,\nglove, no_glove, goggles, no_goggles,\nmask, no_mask, shoes, no_shoes  (11)"],
    ["Format", "JPEG + PNG pixel masks (YOLO-seg)"],
    ["Split", "600 train / 600 val / 600 test"],
    ["Advantage", "Both absence & presence classes\nenables granular PPE gap analysis"],
  ];
  a3info.forEach(([k, v], i) => {
    s.addText(k + ":", {
      x: 5.25, y: 1.5 + i * 0.56, w: 1.3, h: 0.4,
      fontSize: 9.5, bold: true, color: DARK_GOLD, valign: "top", margin: 0
    });
    s.addText(v, {
      x: 6.55, y: 1.5 + i * 0.56, w: 2.9, h: 0.4,
      fontSize: 9.5, color: DARK, valign: "top", margin: 0
    });
  });

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — Dataset Journey (Narrative)
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "Finding the Right Dataset", "Why we had to move beyond Assignment 2 data for segmentation");

  s.addText("Assignment 2 used the MinhNKB dataset, which provides only image-level labels — no bounding boxes, no pixel masks. For pixel-wise and instance segmentation, spatial annotations are essential. This required finding a new dataset entirely.", {
    x: 0.35, y: 0.9, w: 9.3, h: 0.65,
    fontSize: 11.5, color: DARK, margin: 0
  });

  // Timeline of decisions
  const steps = [
    {
      n: "1",
      title: "MinhNKB Ruled Out",
      desc: "5 coarse classes, image-level labels only. Cannot train a semantic segmenter without pixel annotations.",
      color: "B91C1C",
    },
    {
      n: "2",
      title: "COCO-PPE Filtered",
      desc: "General objects, no body-part PPE class distinctions. Insufficient for our granular compliance schema.",
      color: "B45309",
    },
    {
      n: "3",
      title: "keremberke Selected",
      desc: "11 PPE classes incl. absence variants. Roboflow-hosted. 600-image val set with polygon masks. Ideal fit.",
      color: "15803D",
    },
    {
      n: "4",
      title: "SAM2 Pseudo-Masks",
      desc: "For additional diversity: bounding boxes from keremberke annotations → SAM2 box prompts → pixel masks.",
      color: "1D4ED8",
    },
  ];

  steps.forEach((step, i) => {
    const x = 0.35 + i * 2.35;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.75, w: 2.15, h: 3.15,
      fill: { color: WHITE }, line: { color: step.color, width: 2 }, shadow: mkShadow()
    });
    s.addShape(pres.shapes.OVAL, {
      x: x + 0.78, y: 1.55, w: 0.6, h: 0.6,
      fill: { color: step.color }, line: { color: step.color }
    });
    s.addText(step.n, {
      x: x + 0.78, y: 1.55, w: 0.6, h: 0.6,
      fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.75, w: 2.15, h: 0.38,
      fill: { color: step.color }, line: { color: step.color }
    });
    s.addText(step.title, {
      x, y: 1.75, w: 2.15, h: 0.38,
      fontSize: 9.5, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
    });
    s.addText(step.desc, {
      x: x + 0.1, y: 2.22, w: 1.95, h: 2.5,
      fontSize: 9, color: DARK, margin: 0
    });

    // Arrow between steps
    if (i < 3) {
      s.addShape(pres.shapes.RECTANGLE, {
        x: x + 2.17, y: 3.26, w: 0.16, h: 0.08,
        fill: { color: MGRAY }, line: { color: MGRAY }
      });
    }
  });

  // SAM2 note
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 5.0, w: 9.3, h: 0.28,
    fill: { color: "EFF6FF" }, line: { color: "BFDBFE", width: 1 }
  });
  s.addText("SAM2 (Meta 2024): Segment Anything Model v2 — prompted with bounding boxes to auto-generate high-quality pixel masks for augmenting the training set.", {
    x: 0.45, y: 5.02, w: 9.1, h: 0.24,
    fontSize: 8.5, color: "1D4ED8", valign: "middle", italic: true, margin: 0
  });

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Exploratory Data Analysis
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "Exploratory Data Analysis", "Class distributions, pixel imbalance, and dataset characteristics");

  const eda = img(PLOTS, "01_eda_analysis.png");
  if (eda) {
    s.addImage({ path: eda, x: 0.3, y: 0.88, w: 5.4, h: 3.5 });
  }

  // Key EDA findings
  const findings = [
    ["Class Imbalance", "Absence classes (no_mask, no_helmet) are 3–5× under-represented relative to presence classes — requires focal loss or oversampling."],
    ["Pixel Distribution", "Background dominates pixel counts (>70% of pixels), causing naive models to bias toward background prediction."],
    ["Spatial Patterns", "PPE items cluster in the upper half of frames (helmets, goggles, masks) while shoes appear only in bottom 20%."],
    ["Box Area (IQR)", "Absence-class bounding boxes have higher IQR variance — subjects often partially occluded when PPE is absent."],
  ];

  findings.forEach(([ title, desc ], i) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.85, y: 0.88 + i * 0.88, w: 3.8, h: 0.78,
      fill: { color: WHITE }, line: { color: MGRAY, width: 1 }, shadow: mkShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.85, y: 0.88 + i * 0.88, w: 0.08, h: 0.78,
      fill: { color: i % 2 === 0 ? RED : GOLD }, line: { color: i % 2 === 0 ? RED : GOLD }
    });
    s.addText(title, {
      x: 6.02, y: 0.9 + i * 0.88, w: 3.55, h: 0.26,
      fontSize: 10, bold: true, color: DARK, margin: 0
    });
    s.addText(desc, {
      x: 6.02, y: 1.14 + i * 0.88, w: 3.55, h: 0.44,
      fontSize: 8.5, color: GRAY, margin: 0
    });
  });

  s.addText("Note: Most PPE datasets only annotate items that are present. keremberke annotates both presence and absence (no_helmet, no_mask, etc.) — which is what a real compliance system actually needs to detect.", {
    x: 0.3, y: 4.55, w: 9.4, h: 0.58,
    fontSize: 9.5, color: "15803D", italic: true, margin: 0,
    fill: { color: "F0FDF4" }, line: { color: "BBF7D0", width: 1 }
  });

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — Four-Category Evaluation Framework
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "Evaluation Framework", "Four required categories — one model from each");

  s.addText("The course requires one model from each category. That constraint turns out to be useful: it forces a direct comparison of architecture design, training strategy, and task transfer on the same dataset.", {
    x: 0.35, y: 0.88, w: 9.3, h: 0.5,
    fontSize: 11, color: DARK, margin: 0
  });

  const categories = [
    {
      letter: "a", color: CAT_A,
      title: "(a)  Custom Architecture",
      subtitle: "Built and trained from scratch, designed for this task",
      model: "PPENet CNN",
      task: "Image Classification",
      result: "87.3% Accuracy",
      details: "3 conv blocks (32→64→128 ch), AdaptiveAvgPool, FC 512→256→5. 226K params. OneCycleLR, label smoothing. 32×32 crops.",
    },
    {
      letter: "b", color: CAT_B,
      title: "(b)  Pre-existing Architecture, Trained from Scratch",
      subtitle: "Established architecture, weights randomly initialised for our data",
      model: "UNet",
      task: "Semantic Segmentation",
      result: "56.2% mIoU",
      details: "Ronneberger 2015 encoder-decoder with skip connections. Trained on MinhNKB segmentation from random init. 11-class output head.",
    },
    {
      letter: "c", color: CAT_C,
      title: "(c)  Pre-existing + Pretrained Weights, NOT Fine-tuned",
      subtitle: "COCO pretrained weights used directly — zero-shot inference",
      model: "DeepLabV3+ ResNet-50 (zero-shot)",
      task: "Semantic Segmentation",
      result: "8.2% mIoU  (all PPE = 0%)",
      details: "21 COCO class output head. COCO classes (aeroplane, bicycle…) do not match our 10 PPE classes. Background index 0 coincidentally overlaps.",
    },
    {
      letter: "d", color: CAT_D,
      title: "(d)  Pre-existing + Pretrained Weights, Fine-tuned",
      subtitle: "Transfer learning: pretrained backbone + task-specific fine-tuning",
      model: "ViT-B/16 · DeepLabV3+ · YOLOv8n-seg",
      task: "Classification / Seg / Instance Seg",
      result: "93.9% · 63.5% mIoU · 87.1% mAP50",
      details: "ImageNet pretrained ViT fine-tuned for crop classification. COCO-pretrained DeepLabV3+ fine-tuned for 11-class PPE segmentation. YOLOv8n-seg fine-tuned for instance masks.",
    },
  ];

  categories.forEach((cat, i) => {
    const y = 1.5 + i * 0.98;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.3, y, w: 9.4, h: 0.88,
      fill: { color: WHITE }, line: { color: MGRAY, width: 1 }, shadow: mkShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.3, y, w: 0.42, h: 0.88,
      fill: { color: cat.color }, line: { color: cat.color }
    });
    s.addText(`(${cat.letter})`, {
      x: 0.3, y, w: 0.42, h: 0.88,
      fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
    });
    s.addText(cat.title, {
      x: 0.82, y: y + 0.04, w: 5.5, h: 0.3,
      fontSize: 10.5, bold: true, color: DARK, margin: 0
    });
    s.addText(cat.subtitle, {
      x: 0.82, y: y + 0.33, w: 5.5, h: 0.22,
      fontSize: 8.5, color: GRAY, italic: true, margin: 0
    });
    s.addText(cat.details, {
      x: 0.82, y: y + 0.53, w: 5.5, h: 0.28,
      fontSize: 8, color: GRAY, margin: 0
    });
    // Result badge
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.5, y: y + 0.08, w: 3.1, h: 0.72,
      fill: { color: LGRAY }, line: { color: MGRAY, width: 1 }
    });
    s.addText(cat.model, {
      x: 6.6, y: y + 0.1, w: 2.9, h: 0.24,
      fontSize: 8.5, bold: true, color: cat.color, margin: 0
    });
    s.addText(cat.task, {
      x: 6.6, y: y + 0.32, w: 2.9, h: 0.18,
      fontSize: 8, color: GRAY, italic: true, margin: 0
    });
    s.addText(cat.result, {
      x: 6.6, y: y + 0.5, w: 2.9, h: 0.22,
      fontSize: 9, bold: true, color: RED, margin: 0
    });
  });

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — Model Architectures
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "Model Architectures", "Design decisions, backbone choices, and training configuration");

  const models = [
    {
      name: "PPENetFast (CNN)",
      cat: "a", catColor: CAT_A,
      arch: "3× [Conv→BN→ReLU→Pool] · AdaptiveAvgPool(2,2) · FC(512→256→5)",
      params: "226K",
      input: "32×32 RGB crop",
      training: "100 epochs · OneCycleLR (1e-3) · Label smoothing 0.05 · Batch 256",
      task: "Image Classification",
    },
    {
      name: "UNet",
      cat: "b", catColor: CAT_B,
      arch: "Encoder: 4× [Conv→BN→ReLU→MaxPool] · Bottleneck · Decoder: bilinear up + skip",
      params: "~31M",
      input: "512×512 RGB",
      training: "50 epochs · Adam · Cross-entropy · Random init · 5-class output",
      task: "Semantic Segmentation",
    },
    {
      name: "ViT-B/16",
      cat: "d", catColor: CAT_D,
      arch: "12 transformer layers · 12 attention heads · 768-dim · [CLS] token classification",
      params: "86M (pretrained) + fine-tuned head",
      input: "224×224 RGB",
      training: "10 epochs · AdamW (1e-4) · ImageNet pretrained → MinhNKB fine-tune",
      task: "Image Classification",
    },
    {
      name: "DeepLabV3+ ResNet-50",
      cat: "d", catColor: CAT_D,
      arch: "ResNet-50 encoder · ASPP (atrous rates 6,12,18) · Decoder with low-level features",
      params: "42.0M",
      input: "512×512 RGB",
      training: "50 epochs · SGD momentum · COCO pretrained → keremberke fine-tune · 11 classes",
      task: "Semantic Segmentation",
    },
    {
      name: "YOLOv8n-seg",
      cat: "d", catColor: CAT_D,
      arch: "CSPDarknet backbone · C2f neck · Dual-head: bounding box + 32-coeff mask prototype",
      params: "~3.4M",
      input: "640×640 RGB",
      training: "50 epochs · SGD · COCO pretrained → keremberke fine-tune · 10 PPE classes",
      task: "Instance Segmentation",
    },
  ];

  const colW = [1.6, 3.4, 0.6, 1.0, 3.0];
  const headers = ["Model", "Architecture", "Cat", "Params", "Task + Training"];
  let tableData = [];

  // Header row
  tableData.push(headers.map((h, ci) => ({
    text: h,
    options: { bold: true, color: WHITE, fill: { color: RED }, fontSize: 9, align: "center", valign: "middle" }
  })));

  models.forEach((m, i) => {
    tableData.push([
      { text: m.name, options: { bold: true, fontSize: 8.5, color: DARK, fill: { color: i % 2 === 0 ? WHITE : LGRAY } } },
      { text: m.arch, options: { fontSize: 7.5, color: GRAY, fill: { color: i % 2 === 0 ? WHITE : LGRAY } } },
      { text: `(${m.cat})`, options: { bold: true, fontSize: 9, color: WHITE, fill: { color: m.catColor }, align: "center" } },
      { text: m.params, options: { fontSize: 8, color: DARK, fill: { color: i % 2 === 0 ? WHITE : LGRAY }, align: "center" } },
      { text: m.task + "\n" + m.training, options: { fontSize: 7.5, color: GRAY, fill: { color: i % 2 === 0 ? WHITE : LGRAY } } },
    ]);
  });

  s.addTable(tableData, {
    x: 0.3, y: 1.0, w: 9.4, h: 4.1,
    colW: colW.map(w => w),
    border: { pt: 0.5, color: MGRAY },
    autoPage: false,
  });

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 11 — Classification Results (A2)
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "Classification Results — Assignment 2", "Crop-level PPE classification across custom and transfer-learned models");

  // Big stat row
  const classStats = [
    ["76.3%", "SVM (PCA+RBF)\nBaseline", GRAY,    "(b)"],
    ["87.3%", "PPENetFast CNN\nCustom Design", RED, "(a)"],
    ["93.9%", "ViT-B/16\nFine-tuned", CAT_D,     "(d)"],
  ];
  const xPos2 = [0.4, 3.65, 6.9];
  classStats.forEach(([val, lbl, col, cat], i) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: xPos2[i], y: 0.95, w: 2.85, h: 1.65,
      fill: { color: WHITE }, line: { color: col, width: 2 }, shadow: mkShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: xPos2[i], y: 0.95, w: 2.85, h: 0.07,
      fill: { color: col }, line: { color: col }
    });
    s.addText(val, {
      x: xPos2[i], y: 1.0, w: 2.85, h: 0.9,
      fontSize: 46, bold: true, color: col, align: "center", valign: "middle", margin: 0
    });
    s.addText(lbl, {
      x: xPos2[i], y: 1.88, w: 2.85, h: 0.6,
      fontSize: 9, color: GRAY, align: "center", valign: "top", margin: 0
    });
    addCatBadge(s, xPos2[i] + 2.48, 0.97, cat.slice(1, 2), cat === "(a)" ? CAT_A : (cat === "(b)" ? CAT_B : CAT_D));
  });

  // Per-class F1 table
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 2.75, w: 4.6, h: 2.5,
    fill: { color: WHITE }, line: { color: MGRAY, width: 1 }, shadow: mkShadow()
  });
  s.addText("Per-Class F1 Scores — PPENet CNN", {
    x: 0.3, y: 2.75, w: 4.6, h: 0.32,
    fontSize: 10, bold: true, color: WHITE,
    fill: { color: RED }, valign: "middle", align: "center", margin: 0
  });

  const f1data = [
    ["Class", "F1", "Precision", "Recall"],
    ["helmet", "0.89", "0.91", "0.87"],
    ["safety_vest", "0.88", "0.87", "0.89"],
    ["full_ppe", "0.77", "0.80", "0.74"],
    ["partial_ppe", "0.76", "0.73", "0.79"],
    ["no_ppe", "0.86", "0.88", "0.84"],
  ];
  s.addTable(
    f1data.map((row, ri) => row.map((cell, ci) => ({
      text: cell,
      options: {
        bold: ri === 0 || ci === 0,
        fontSize: 9,
        color: ri === 0 ? WHITE : (ci === 1 && parseFloat(cell) < 0.8 ? "B45309" : DARK),
        fill: { color: ri === 0 ? DARK_RED : (ri % 2 === 0 ? LGRAY : WHITE) },
        align: ci === 0 ? "left" : "center",
        valign: "middle",
      }
    }))),
    { x: 0.3, y: 3.07, w: 4.6, h: 2.12, border: { pt: 0.5, color: MGRAY } }
  );

  // Right: Training curve image
  const trainImg = img(MODELS, "prod_cnn_training.png");
  if (trainImg) {
    s.addImage({ path: trainImg, x: 5.1, y: 2.75, w: 4.55, h: 2.5 });
    s.addText("PPENet Training Curves (loss & accuracy)", {
      x: 5.1, y: 5.2, w: 4.55, h: 0.2,
      fontSize: 8, color: GRAY, align: "center", italic: true, margin: 0
    });
  }

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 12 — Segmentation Results (A3)
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "Segmentation Results — Assignment 3", "Semantic and instance segmentation on the keremberke dataset");

  // Four model results
  const segResults = [
    { model: "UNet (from scratch)", cat: "b", catColor: CAT_B, metric: "56.2%", metricLabel: "mIoU", sub: "Pixel Acc: 91.4%" },
    { model: "DeepLabV3+ Zero-shot", cat: "c", catColor: CAT_C, metric: "8.2%",  metricLabel: "mIoU", sub: "All PPE classes = 0%" },
    { model: "DeepLabV3+ Fine-tuned", cat: "d", catColor: CAT_D, metric: "63.5%", metricLabel: "mIoU", sub: "Pixel Acc: 99.6%" },
    { model: "YOLOv8n-seg", cat: "d", catColor: CAT_D, metric: "87.1%", metricLabel: "Mask mAP50", sub: "Box mAP50: 90.0%" },
  ];

  const xPoss = [0.3, 2.63, 4.96, 7.29];
  segResults.forEach((r, i) => {
    const x = xPoss[i];
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 0.92, w: 2.2, h: 1.9,
      fill: { color: WHITE }, line: { color: r.catColor, width: 2 }, shadow: mkShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 0.92, w: 2.2, h: 0.06,
      fill: { color: r.catColor }, line: { color: r.catColor }
    });
    s.addText(r.metric, {
      x, y: 0.98, w: 2.2, h: 0.95,
      fontSize: 34, bold: true, color: r.catColor, align: "center", valign: "middle", margin: 0
    });
    s.addText(r.metricLabel, {
      x, y: 1.88, w: 2.2, h: 0.26,
      fontSize: 9, color: GRAY, align: "center", margin: 0
    });
    s.addText(r.model, {
      x, y: 2.13, w: 2.2, h: 0.28,
      fontSize: 8.5, bold: true, color: DARK, align: "center", margin: 0
    });
    s.addText(r.sub, {
      x, y: 2.4, w: 2.2, h: 0.28,
      fontSize: 8, color: GRAY, align: "center", italic: true, margin: 0
    });
    addCatBadge(s, x + 0.02, 0.95, r.cat, r.catColor);
  });

  // Per-class IoU comparison (DeepLab fine-tuned)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 2.88, w: 4.6, h: 2.35,
    fill: { color: WHITE }, line: { color: MGRAY, width: 1 }, shadow: mkShadow()
  });
  s.addText("DeepLabV3+ Fine-tuned — Per-Class IoU", {
    x: 0.3, y: 2.88, w: 4.6, h: 0.32,
    fontSize: 10, bold: true, color: WHITE,
    fill: { color: CAT_D }, valign: "middle", align: "center", margin: 0
  });

  const perClassData = [
    ["background",   "0.9953"], ["helmet",    "0.9261"],
    ["no_helmet",    "0.5550"], ["glove",     "0.8817"],
    ["no_glove",     "0.7680"], ["goggles",   "0.8427"],
    ["no_goggles",   "0.7920"], ["mask",      "0.8040"],
    ["no_mask",      "0.5170"], ["shoes",     "0.6830"],
    ["no_shoes",     "0.6630"],
  ];
  const rows = perClassData.map(([cls, iou]) => [
    { text: cls, options: { fontSize: 8, color: DARK, fill: { color: WHITE } } },
    { text: iou, options: { fontSize: 8, bold: true, color: parseFloat(iou) < 0.65 ? "B45309" : "15803D", fill: { color: WHITE }, align: "center" } },
    { text: "", options: { fill: { color: WHITE } } },
  ]);
  // Add bar-like visual with shapes
  s.addTable(perClassData.map(([cls, iou], ri) => [
    { text: cls, options: { fontSize: 7.5, color: DARK, fill: { color: ri % 2 === 0 ? WHITE : LGRAY }, valign: "middle" } },
    { text: iou, options: { fontSize: 7.5, bold: true, color: parseFloat(iou) < 0.65 ? "B45309" : "15803D", fill: { color: ri % 2 === 0 ? WHITE : LGRAY }, align: "center", valign: "middle" } },
  ]), { x: 0.3, y: 3.2, w: 4.6, h: 1.95, border: { pt: 0.5, color: MGRAY } });

  // YOLOv8n results image
  const yoloRes = img(MODELS, "yolo_seg_results_plot.png");
  if (yoloRes) {
    s.addImage({ path: yoloRes, x: 5.1, y: 2.88, w: 4.55, h: 2.35 });
    s.addText("YOLOv8n-seg training curves", {
      x: 5.1, y: 5.2, w: 4.55, h: 0.2,
      fontSize: 8, color: GRAY, align: "center", italic: true, margin: 0
    });
  }

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 13 — Zero-Shot vs Fine-Tuned: The Value of Transfer Learning
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "Category (c) vs (d): The Value of Fine-tuning", "Why pretrained weights alone are not sufficient for specialised tasks");

  // Left: zero-shot failure
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 1.02, w: 4.5, h: 4.1,
    fill: { color: "FFF5F5" }, line: { color: "FCA5A5", width: 2 }, shadow: mkShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 1.02, w: 4.5, h: 0.38,
    fill: { color: CAT_C }, line: { color: CAT_C }
  });
  s.addText("(c)  COCO Zero-Shot — 8.2% mIoU", {
    x: 0.3, y: 1.02, w: 4.5, h: 0.38,
    fontSize: 10.5, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
  });

  s.addText("DeepLabV3+ ResNet-50\n21 COCO output classes", {
    x: 0.45, y: 1.5, w: 4.2, h: 0.52,
    fontSize: 10, bold: true, color: "991B1B", margin: 0
  });

  const zeroClasses = [
    ["background", "90.5%", "Coincidental overlap (index 0)"],
    ["helmet",     " 0.0%", "No COCO class maps to helmet"],
    ["no_helmet",  " 0.0%", "Absence classes don't exist in COCO"],
    ["glove",      " 0.0%", "—"],
    ["mask",       " 0.0%", "—"],
    ["shoes",      " 0.0%", "—"],
    ["…other",     " 0.0%", "—"],
  ];
  zeroClasses.forEach(([cls, iou, note], i) => {
    s.addText(cls, {
      x: 0.45, y: 2.1 + i * 0.37, w: 1.3, h: 0.32,
      fontSize: 8.5, color: DARK, margin: 0
    });
    s.addText(iou, {
      x: 1.75, y: 2.1 + i * 0.37, w: 0.7, h: 0.32,
      fontSize: 8.5, bold: true, color: parseFloat(iou) > 0.5 ? "15803D" : "B91C1C", align: "center", margin: 0
    });
    s.addText(note, {
      x: 2.5, y: 2.1 + i * 0.37, w: 2.15, h: 0.32,
      fontSize: 7.5, color: GRAY, italic: true, margin: 0
    });
  });

  s.addText("Conclusion: COCO class vocabulary has zero overlap with our 10 PPE classes. The model cannot predict what it was never trained to predict.", {
    x: 0.45, y: 4.65, w: 4.15, h: 0.38,
    fontSize: 8.5, color: "991B1B", italic: true, margin: 0
  });

  // Right: fine-tuned success
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.02, w: 4.5, h: 4.1,
    fill: { color: "F0FDF4" }, line: { color: "86EFAC", width: 2 }, shadow: mkShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.02, w: 4.5, h: 0.38,
    fill: { color: CAT_D }, line: { color: CAT_D }
  });
  s.addText("(d)  Fine-tuned — 63.5% mIoU", {
    x: 5.2, y: 1.02, w: 4.5, h: 0.38,
    fontSize: 10.5, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
  });

  s.addText("Same backbone (ResNet-50)\nFine-tuned 50 epochs on keremberke", {
    x: 5.35, y: 1.5, w: 4.2, h: 0.52,
    fontSize: 10, bold: true, color: "14532D", margin: 0
  });

  const ftClasses = [
    ["background", "99.5%", "✓"],
    ["helmet",     "92.6%", "✓"],
    ["no_helmet",  "55.5%", "Challenging — minority class"],
    ["glove",      "88.2%", "✓"],
    ["mask",       "80.4%", "✓"],
    ["shoes",      "68.3%", "✓"],
    ["no_mask",    "51.7%", "Hardest — small & occluded"],
  ];
  ftClasses.forEach(([cls, iou, note], i) => {
    s.addText(cls, {
      x: 5.35, y: 2.1 + i * 0.37, w: 1.3, h: 0.32,
      fontSize: 8.5, color: DARK, margin: 0
    });
    s.addText(iou, {
      x: 6.65, y: 2.1 + i * 0.37, w: 0.7, h: 0.32,
      fontSize: 8.5, bold: true, color: parseFloat(iou) > 0.65 ? "15803D" : "B45309", align: "center", margin: 0
    });
    s.addText(note, {
      x: 7.4, y: 2.1 + i * 0.37, w: 2.15, h: 0.32,
      fontSize: 7.5, color: GRAY, italic: true, margin: 0
    });
  });

  s.addText("Fine-tuning lifted mIoU from 8.2% → 63.5% — a +55 percentage-point improvement from the same architecture and pretrained weights.", {
    x: 5.35, y: 4.65, w: 4.15, h: 0.38,
    fontSize: 8.5, color: "14532D", bold: true, italic: true, margin: 0
  });

  // Middle gap arrow
  s.addShape(pres.shapes.RECTANGLE, {
    x: 4.8, y: 2.85, w: 0.4, h: 0.08,
    fill: { color: GOLD }, line: { color: GOLD }
  });
  s.addText("+55pp", {
    x: 4.72, y: 2.65, w: 0.56, h: 0.25,
    fontSize: 9, bold: true, color: GOLD, align: "center", margin: 0
  });

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 14 — Visual Results
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "Visual Results", null);

  // Column headers — below the red bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 0.68, w: 4.5, h: 0.28,
    fill: { color: CAT_D }, line: { color: CAT_D }
  });
  s.addText("DeepLabV3+  Fine-tuned  (Semantic Segmentation)", {
    x: 0.3, y: 0.68, w: 4.5, h: 0.28,
    fontSize: 8.5, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 0.68, w: 4.5, h: 0.28,
    fill: { color: CAT_D }, line: { color: CAT_D }
  });
  s.addText("YOLOv8n-seg  Fine-tuned  (Instance Segmentation)", {
    x: 5.2, y: 0.68, w: 4.5, h: 0.28,
    fontSize: 8.5, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
  });

  const dlabBest = img(MODELS, "deeplab_best3.png");
  const yoloBest = img(MODELS, "yolo_best3.png");
  const dlabWorst = img(MODELS, "deeplab_worst3.png");
  const yoloWorst = img(MODELS, "yolo_worst3.png");

  if (dlabBest) {
    s.addImage({ path: dlabBest, x: 0.3, y: 0.98, w: 4.5, h: 2.05 });
    s.addText("▲ Best 3 predictions (highest IoU)", {
      x: 0.3, y: 3.03, w: 4.5, h: 0.2, fontSize: 7.5, color: "15803D", align: "center", italic: true, margin: 0
    });
  }
  if (yoloBest) {
    s.addImage({ path: yoloBest, x: 5.2, y: 0.98, w: 4.5, h: 2.05 });
    s.addText("▲ Best 3 predictions (highest mAP)", {
      x: 5.2, y: 3.03, w: 4.5, h: 0.2, fontSize: 7.5, color: "15803D", align: "center", italic: true, margin: 0
    });
  }
  if (dlabWorst) {
    s.addImage({ path: dlabWorst, x: 0.3, y: 3.25, w: 4.5, h: 1.9 });
    s.addText("▼ Worst 3 predictions (no_mask / no_helmet — hardest absence classes)", {
      x: 0.3, y: 5.15, w: 4.5, h: 0.2, fontSize: 7.5, color: "B91C1C", align: "center", italic: true, margin: 0
    });
  }
  if (yoloWorst) {
    s.addImage({ path: yoloWorst, x: 5.2, y: 3.25, w: 4.5, h: 1.9 });
    s.addText("▼ Worst 3 predictions (occluded / crowded scenes)", {
      x: 5.2, y: 5.15, w: 4.5, h: 0.2, fontSize: 7.5, color: "B91C1C", align: "center", italic: true, margin: 0
    });
  }

  // Fallback if no images
  if (!dlabBest && !yoloBest) {
    s.addText("Prediction visualisations generated during evaluation.\nSee results/models/deeplab_best3.png, yolo_best3.png", {
      x: 0.5, y: 2.0, w: 9.0, h: 1.0,
      fontSize: 14, color: GRAY, align: "center", valign: "middle", italic: true
    });
  }

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 15 — Why Results Are Meaningful
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "What the Numbers Actually Mean", "Three results that tell a coherent story");

  // Summary comparison chart (drawn as bar shapes)
  const barData = [
    { label: "SVM Baseline\n(b) Classification", pct: 76.3, color: CAT_B },
    { label: "PPENet CNN\n(a) Classification",   pct: 87.3, color: CAT_A },
    { label: "UNet Seg\n(b) from scratch",        pct: 56.2, color: CAT_B },
    { label: "Zero-shot DL\n(c) No fine-tune",    pct: 8.2,  color: CAT_C },
    { label: "DeepLabV3+\n(d) Fine-tuned mIoU",  pct: 63.5, color: CAT_D },
    { label: "ViT-B/16\n(d) Classification",      pct: 93.9, color: CAT_D },
    { label: "YOLOv8n-seg\n(d) Mask mAP50",       pct: 87.1, color: CAT_D },
  ];

  const chartY = 1.05;
  const chartH = 2.2;
  const barW = 1.22;
  const gap = 0.1;
  const chartX = 0.35;

  // Axis
  s.addShape(pres.shapes.LINE, {
    x: chartX, y: chartY, w: 0, h: chartH,
    line: { color: MGRAY, width: 1.5 }
  });
  s.addShape(pres.shapes.LINE, {
    x: chartX, y: chartY + chartH, w: 9.0, h: 0,
    line: { color: MGRAY, width: 1.5 }
  });

  // Grid lines
  [25, 50, 75, 100].forEach(v => {
    const gy = chartY + chartH - (v / 100) * chartH;
    s.addShape(pres.shapes.LINE, {
      x: chartX, y: gy, w: 9.0, h: 0,
      line: { color: "EEEEEE", width: 0.5 }
    });
    s.addText(`${v}%`, {
      x: chartX - 0.38, y: gy - 0.12, w: 0.36, h: 0.24,
      fontSize: 7, color: GRAY, align: "right", margin: 0
    });
  });

  barData.forEach((bar, i) => {
    const bx = chartX + 0.15 + i * (barW + gap);
    const bh = (bar.pct / 100) * chartH;
    const by = chartY + chartH - bh;
    s.addShape(pres.shapes.RECTANGLE, {
      x: bx, y: by, w: barW, h: bh,
      fill: { color: bar.color }, line: { color: bar.color }
    });
    s.addText(`${bar.pct}%`, {
      x: bx, y: by - 0.26, w: barW, h: 0.24,
      fontSize: 8, bold: true, color: bar.color, align: "center", margin: 0
    });
    s.addText(bar.label, {
      x: bx - 0.05, y: chartY + chartH + 0.06, w: barW + 0.1, h: 0.52,
      fontSize: 7, color: DARK, align: "center", margin: 0
    });
  });

  // Key insights
  const insights = [
    ["Same model, 55 more points",
     "Zero-shot DeepLabV3+ (COCO weights, no fine-tuning): 8.2% mIoU. The same architecture after 50 epochs on our data: 63.5%. The model does not change — what changes is whether it has seen the task."],
    ["380× fewer parameters, 6.6 pp behind",
     "PPENet CNN (226K params) reaches 87.3%. ViT-B/16 (86M params) reaches 93.9%. The gap is real but the compute difference is not. A small purpose-built model holds its own here."],
    ["Per-worker vs per-pixel",
     "YOLOv8n-seg at 87.1% mask mAP50 can say which worker is missing a mask. Semantic segmentation can only say that somewhere in the frame, a mask is absent. For a real site, the first answer is more useful."],
  ];
  insights.forEach(([title, body], i) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.3, y: 3.88 + i * 0.48, w: 9.4, h: 0.42,
      fill: { color: i % 2 === 0 ? "FFF8E8" : WHITE }, line: { color: GOLD, width: 1 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.3, y: 3.88 + i * 0.48, w: 0.08, h: 0.42,
      fill: { color: GOLD }, line: { color: GOLD }
    });
    s.addText(title + ":  ", {
      x: 0.46, y: 3.90 + i * 0.48, w: 1.7, h: 0.36,
      fontSize: 8.5, bold: true, color: DARK_RED, valign: "middle", margin: 0
    });
    s.addText(body, {
      x: 2.05, y: 3.90 + i * 0.48, w: 7.5, h: 0.36,
      fontSize: 8.5, color: DARK, valign: "middle", margin: 0
    });
  });

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 16 — Future Work
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "Future Work", "Next steps for deployment and performance improvement");

  const futureItems = [
    {
      num: "01",
      title: "More YOLO Fine-tuning Epochs",
      desc: "Current person detector at mAP50=0.679 after 4 epochs. Target >0.85 with 50–100 epochs and data augmentation (mosaic, mixup, copy-paste).",
      priority: "High",
      color: RED,
    },
    {
      num: "02",
      title: "EfficientNet / MobileNetV2 Backbone",
      desc: "Swap ResNet-50 in DeepLabV3+ for EfficientNet-B4 — better accuracy/param ratio. MobileNetV3 enables edge-device deployment (Jetson Nano, RPi).",
      priority: "High",
      color: RED,
    },
    {
      num: "03",
      title: "Focal Loss + Class-Weighted Training",
      desc: "Absence classes (no_mask 51.7%, no_helmet 55.5%) suffer most. Focal loss (γ=2) and inverse-frequency class weights expected to boost tail-class IoU by 5–10 pp.",
      priority: "Medium",
      color: GOLD,
    },
    {
      num: "04",
      title: "SegFormer / Mask2Former Architecture",
      desc: "Transformer-based segmenters achieve 50+ mIoU on ADE20K. Replacing DeepLabV3+ with Mask2Former could yield +5–10% mIoU with similar compute.",
      priority: "Medium",
      color: GOLD,
    },
    {
      num: "05",
      title: "End-to-End YOLO PPE Detection",
      desc: "Single-stage YOLOv8-seg trained directly on raw scene images (not crops) for simultaneous person + PPE detection, eliminating the two-stage pipeline latency.",
      priority: "Medium",
      color: GOLD,
    },
    {
      num: "06",
      title: "Real-Time CCTV Deployment",
      desc: "Port to ONNX / TensorRT on RTX 5070. Target <30 ms/frame at 1080p for live compliance alerting on construction-site video feeds.",
      priority: "Low",
      color: GRAY,
    },
  ];

  const xCols = [0.3, 5.1];
  futureItems.forEach((item, i) => {
    const col = i < 3 ? 0 : 1;
    const row = i % 3;
    const x = xCols[col];
    const y = 1.02 + row * 1.42;

    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.6, h: 1.3,
      fill: { color: WHITE }, line: { color: MGRAY, width: 1 }, shadow: mkShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.5, h: 1.3,
      fill: { color: item.color }, line: { color: item.color }
    });
    s.addText(item.num, {
      x, y, w: 0.5, h: 0.72,
      fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
    });
    s.addText(item.priority, {
      x, y: y + 0.72, w: 0.5, h: 0.58,
      fontSize: 7, color: WHITE, align: "center", valign: "middle", margin: 0
    });
    s.addText(item.title, {
      x: x + 0.58, y: y + 0.06, w: 3.9, h: 0.32,
      fontSize: 10, bold: true, color: DARK, margin: 0
    });
    s.addText(item.desc, {
      x: x + 0.58, y: y + 0.38, w: 3.9, h: 0.84,
      fontSize: 8.5, color: GRAY, margin: 0
    });
  });

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 17 — References
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BGWHITE };
  addRedHeader(s, pres, "References", "Key papers, datasets, and frameworks used in this project");

  const refs = [
    { num: "[1]", cite: "Redmon et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. CVPR 2016." },
    { num: "[2]", cite: "Jocher et al. (2023). Ultralytics YOLOv8. GitHub: ultralytics/ultralytics." },
    { num: "[3]", cite: "Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015." },
    { num: "[4]", cite: "Chen et al. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabV3+). ECCV 2018." },
    { num: "[5]", cite: "Dosovitskiy et al. (2021). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. ICLR 2021." },
    { num: "[6]", cite: "He et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016." },
    { num: "[7]", cite: "Kirillov et al. (2023). Segment Anything. ICCV 2023." },
    { num: "[8]", cite: "Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos. arXiv 2408.00714." },
    { num: "[9]", cite: "keremberke (2023). PPE Detection Dataset (v4). Roboflow Universe. roboflow.com/keremberke/ppe-detection." },
    { num: "[10]", cite: "MinhNKB (2022). PPE Classification Dataset. Kaggle. kaggle.com/datasets/minhnkb/ppe." },
    { num: "[11]", cite: "Lin et al. (2014). Microsoft COCO: Common Objects in Context. ECCV 2014." },
    { num: "[12]", cite: "International Labour Organization (2022). Safety and Health at Work: A Vision for Sustainable Development. ILO Geneva." },
  ];

  refs.forEach((ref, i) => {
    const col = i < 6 ? 0 : 1;
    const row = i % 6;
    const x = col === 0 ? 0.3 : 5.1;
    const y = 0.95 + row * 0.73;

    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.6, h: 0.62,
      fill: { color: i % 2 === 0 ? WHITE : LGRAY }, line: { color: MGRAY, width: 0.5 }
    });
    s.addText(ref.num, {
      x: x + 0.06, y, w: 0.42, h: 0.62,
      fontSize: 9, bold: true, color: RED, align: "center", valign: "middle", margin: 0
    });
    s.addText(ref.cite, {
      x: x + 0.52, y: y + 0.06, w: 4.0, h: 0.52,
      fontSize: 7.5, color: DARK, valign: "middle", margin: 0
    });
  });

  addFooter(s, pres);
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 18 — Closing / Thank You
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: DARK_RED };

  // Gold accent bar left
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.18, h: 5.625,
    fill: { color: GOLD }, line: { color: GOLD }
  });

  s.addText("Thank You", {
    x: 0.4, y: 0.8, w: 9.2, h: 1.2,
    fontSize: 60, bold: true, color: WHITE, align: "left", valign: "middle", margin: 0
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 2.1, w: 5.5, h: 0.04,
    fill: { color: GOLD }, line: { color: GOLD }
  });

  s.addText("PPE Detection & Segmentation", {
    x: 0.4, y: 2.2, w: 9.2, h: 0.45,
    fontSize: 20, color: "F0CCCC", italic: true, margin: 0
  });

  // Summary box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 2.8, w: 5.8, h: 2.3,
    fill: { color: "7A0000", transparency: 20 }, line: { color: GOLD, width: 1 }
  });
  const summaryLines = [
    ["PPENet CNN (a):",      "87.3% accuracy  —  226K params"],
    ["UNet (b):",            "56.2% mIoU  —  from scratch"],
    ["DeepLabV3+ (c):",      "8.2% mIoU  —  zero-shot baseline"],
    ["DeepLabV3+ (d):",      "63.5% mIoU  —  fine-tuned"],
    ["ViT-B/16 (d):",        "93.9% accuracy  —  fine-tuned"],
    ["YOLOv8n-seg (d):",     "87.1% Mask mAP50  —  instance seg"],
  ];
  summaryLines.forEach(([k, v], i) => {
    s.addText(k, {
      x: 0.55, y: 2.88 + i * 0.36, w: 1.9, h: 0.32,
      fontSize: 9, bold: true, color: GOLD, valign: "middle", margin: 0
    });
    s.addText(v, {
      x: 2.45, y: 2.88 + i * 0.36, w: 3.6, h: 0.32,
      fontSize: 9, color: WHITE, valign: "middle", margin: 0
    });
  });

  s.addText("Questions?", {
    x: 6.6, y: 2.85, w: 3.1, h: 0.65,
    fontSize: 26, bold: true, color: GOLD, align: "center", margin: 0
  });
  s.addText("Chapman University\nFowler School of Engineering\nAI / Machine Learning — 2024", {
    x: 6.6, y: 3.55, w: 3.1, h: 1.4,
    fontSize: 11, color: "DDBBBB", align: "center", margin: 0
  });
}

// ── Write File ────────────────────────────────────────────────────────────────
const outPath = path.join(__dirname, "PPE_Final_Presentation.pptx");
pres.writeFile({ fileName: outPath })
  .then(() => console.log("Saved -> " + outPath))
  .catch(err => { console.error("Error:", err); process.exit(1); });
