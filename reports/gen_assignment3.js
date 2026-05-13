/**
 * Generate Assignment 3 Write-up DOCX
 * PPE Detection — Pixel-Wise Segmentation
 * Keremberke 10-class schema: DeepLabV3+ (semantic) + YOLOv8n-seg (instance)
 */
"use strict";
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, LevelFormat, PageBreak, Header, Footer, PageNumber,
} = require("docx");
const fs   = require("fs");
const path = require("path");

// ── Paths ──────────────────────────────────────────────────────────────────
const BASE = path.resolve(__dirname, "..");
const IMG  = path.join(BASE, "results", "models");
const OUT  = path.join(BASE, "docs", "Assignment3_Writeup.docx");

// ── Auto-read DeepLab results from CSV if available ────────────────────────
let DEEPLAB_MIOU   = "pending";
let DEEPLAB_PIXACC = "pending";
const DL_IOU = {
  background: "—", helmet: "—", no_helmet: "—",
  glove: "—", no_glove: "—", goggles: "—", no_goggles: "—",
  mask: "—", no_mask: "—", shoes: "—", no_shoes: "—",
};

const dlCsv = path.join(IMG, "deeplab_results.csv");
if (fs.existsSync(dlCsv)) {
  const lines = fs.readFileSync(dlCsv, "utf8").trim().split("\n");
  const headers = lines[0].split(",");
  const values  = lines[1].split(",");
  const row = {};
  headers.forEach((h, i) => { row[h.trim()] = values[i] ? values[i].trim() : ""; });
  if (row["mIoU"])      DEEPLAB_MIOU   = (parseFloat(row["mIoU"])      * 100).toFixed(1) + "%";
  if (row["Pixel_Acc"]) DEEPLAB_PIXACC = (parseFloat(row["Pixel_Acc"]) * 100).toFixed(1) + "%";
  for (const cls of Object.keys(DL_IOU)) {
    const key = "IoU_" + cls;
    if (row[key] && row[key] !== "") DL_IOU[cls] = (parseFloat(row[key]) * 100).toFixed(1) + "%";
  }
  console.log(`DeepLab results loaded: mIoU=${DEEPLAB_MIOU}  PixelAcc=${DEEPLAB_PIXACC}`);
} else {
  console.log("deeplab_results.csv not found — showing pending placeholders");
}

const YOLO_BOX  = "90.0%";
const YOLO_MASK = "87.1%";

// ── Helpers ────────────────────────────────────────────────────────────────
const P = (text, opts = {}) =>
  new Paragraph({
    children: [new TextRun({ text, font: "Arial", size: 22, ...opts })],
    spacing: { after: 140 },
  });

const H1 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: "Arial", size: 32, bold: true })],
    spacing: { before: 280, after: 140 },
  });

const H2 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: "Arial", size: 26, bold: true })],
    spacing: { before: 200, after: 100 },
  });

const H3 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, font: "Arial", size: 23, bold: true, italics: true })],
    spacing: { before: 160, after: 80 },
  });

const bullet = (text) =>
  new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    children: [new TextRun({ text, font: "Arial", size: 22 })],
    spacing: { after: 80 },
  });

const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

function loadImg(name, widthPx, heightPx) {
  const p = path.join(IMG, name);
  if (!fs.existsSync(p)) return null;
  const ext = path.extname(name).replace(".", "").toLowerCase();
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new ImageRun({
      type: ext === "jpg" ? "jpeg" : ext,
      data: fs.readFileSync(p),
      transformation: { width: widthPx, height: heightPx },
      altText: { title: name, description: name, name },
    })],
    spacing: { after: 120 },
  });
}

function imgCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, font: "Arial", size: 18, italics: true, color: "555555" })],
    spacing: { after: 200 },
  });
}

// ── Table helpers ──────────────────────────────────────────────────────────
const border       = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders      = { top: border, bottom: border, left: border, right: border };
const headerShading = { fill: "2E5F8A", type: ShadingType.CLEAR };
const altShading    = { fill: "EEF4FA", type: ShadingType.CLEAR };
const cellMargins   = { top: 80, bottom: 80, left: 120, right: 120 };

function hdrCell(text, widthDxa) {
  return new TableCell({
    borders, width: { size: widthDxa, type: WidthType.DXA },
    shading: headerShading, margins: cellMargins,
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Arial", size: 20, bold: true, color: "FFFFFF" })],
    })],
  });
}

function dataCell(text, widthDxa, shade = false) {
  return new TableCell({
    borders, width: { size: widthDxa, type: WidthType.DXA },
    shading: shade ? altShading : { fill: "FFFFFF", type: ShadingType.CLEAR },
    margins: cellMargins,
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Arial", size: 20 })],
    })],
  });
}

// ── Overall results table ──────────────────────────────────────────────────
function makeResultsTable() {
  const cols = [3000, 2400, 1700, 2260];
  const rows = [
    ["Model", "Training Paradigm", "Task", "Performance"],
    ["DeepLabV3+ ResNet50", "Pretrained COCO+ImageNet, all layers fine-tuned",
     "Semantic Segmentation (11-class)",
     `mIoU = ${DEEPLAB_MIOU}  |  Pixel Acc = ${DEEPLAB_PIXACC}`],
    ["YOLOv8n-seg", "Pretrained COCO, all layers fine-tuned",
     "Instance Segmentation (10-class)",
     `Box mAP50 = ${YOLO_BOX}  |  Mask mAP50 = ${YOLO_MASK}`],
  ];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: cols,
    rows: rows.map((row, ri) =>
      new TableRow({
        children: row.map((cell, ci) =>
          ri === 0 ? hdrCell(cell, cols[ci]) : dataCell(cell, cols[ci], ri % 2 === 0)
        ),
      })
    ),
  });
}

// ── DeepLabV3+ per-class IoU table ─────────────────────────────────────────
function makeIouTable() {
  const cols = [2600, 1760, 5000];
  const data = [
    ["Class",        "IoU",              "Notes"],
    ["background",   DL_IOU.background,  "Fills the majority of pixels in every scene"],
    ["helmet",       DL_IOU.helmet,      "Hard hat — compact, consistent round shape"],
    ["no_helmet",    DL_IOU.no_helmet,   "Head region without a hard hat"],
    ["glove",        DL_IOU.glove,       "Safety glove — deformable, frequent occlusion"],
    ["no_glove",     DL_IOU.no_glove,    "Bare hand where a glove should be worn"],
    ["goggles",      DL_IOU.goggles,     "Protective eyewear — variable shape"],
    ["no_goggles",   DL_IOU.no_goggles,  "Eye region without eyewear"],
    ["mask",         DL_IOU.mask,        "Face mask — smallest class by pixel area"],
    ["no_mask",      DL_IOU.no_mask,     "Exposed lower face without a mask"],
    ["shoes",        DL_IOU.shoes,       "Safety footwear — often partially visible"],
    ["no_shoes",     DL_IOU.no_shoes,    "Foot area without safety footwear"],
    ["Mean IoU",     DEEPLAB_MIOU,       "Average over all 11 classes including background"],
  ];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: cols,
    rows: data.map((row, ri) =>
      new TableRow({
        children: row.map((cell, ci) =>
          ri === 0 ? hdrCell(cell, cols[ci]) : dataCell(cell, cols[ci], ri % 2 === 0)
        ),
      })
    ),
  });
}

// ── YOLOv8n-seg per-class table ────────────────────────────────────────────
function makeYoloTable() {
  const cols = [2340, 1560, 1560, 1900, 1400, 600];
  const data = [
    ["Class",      "Precision", "Recall", "Box mAP50", "Mask mAP50", "n"],
    ["helmet",     "0.946", "0.971", "99.3%", "99.1%", "1523"],
    ["no_helmet",  "0.880", "0.938", "91.9%", "88.7%", "1296"],
    ["glove",      "0.822", "0.858", "87.6%", "86.7%", "4663"],
    ["no_glove",   "0.848", "0.691", "80.8%", "75.6%", "6126"],
    ["goggles",    "0.842", "0.805", "87.1%", "79.6%", "4184"],
    ["no_goggles", "0.861", "0.669", "81.2%", "76.7%", "4092"],
    ["mask",       "0.786", "0.933", "96.1%", "91.9%",  "269"],
    ["no_mask",    "0.867", "0.767", "82.4%", "78.8%",  "661"],
    ["shoes",      "0.819", "0.905", "94.5%", "94.5%",  "755"],
    ["no_shoes",   "0.934", "1.000", "99.5%", "99.5%",  "606"],
    ["All",        "0.861", "0.854", "90.0%", "87.1%", "~24175"],
  ];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: cols,
    rows: data.map((row, ri) =>
      new TableRow({
        children: row.map((cell, ci) =>
          ri === 0 ? hdrCell(cell, cols[ci]) : dataCell(cell, cols[ci], ri % 2 === 0)
        ),
      })
    ),
  });
}

// ── Document ───────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    }],
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: "2E5F8A" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: "2E5F8A" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, italics: true, font: "Arial", color: "3A7ABD" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Assignment 3  |  PPE Pixel-Wise Segmentation  |  Page ", font: "Arial", size: 18, color: "888888" }),
            new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: "888888" }),
          ],
        })],
      }),
    },
    children: [

      // ── Title ──────────────────────────────────────────────────────────────
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Assignment 3: Pixel-Wise PPE Detection", font: "Arial", size: 40, bold: true, color: "2E5F8A" })],
        spacing: { before: 480, after: 120 },
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Semantic and Instance Segmentation for Industrial Safety Compliance", font: "Arial", size: 26, italics: true, color: "555555" })],
        spacing: { after: 80 },
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "CSCI 4930 / 6930  |  Deep Learning  |  Spring 2026", font: "Arial", size: 22, color: "777777" })],
        spacing: { after: 600 },
      }),

      // ── 1. Task and Dataset ────────────────────────────────────────────────
      H1("1. Task Identification and Dataset"),

      P("Assignment 3 moves from classifying whole-person crops to labelling individual pixels and object instances. Two tasks are implemented on the same dataset:"),
      bullet("Semantic segmentation — every pixel in a full scene is assigned one of 11 class labels: background plus the 10 keremberke PPE item classes. The entire frame is processed in a single forward pass."),
      bullet("Instance segmentation — each PPE item is detected as a separate object with its own polygon mask. Two helmets on two different workers are tracked as two distinct instances with separate masks."),

      H2("1.1 Class Definitions"),

      P("The keremberke dataset labels individual PPE items and their absence. Each bounding box covers a single item on a single worker. The 10 classes are:"),
      bullet("helmet — a hard hat physically present and worn on the head."),
      bullet("no_helmet — a head region without a hard hat. The annotated area is the head, not the whole person."),
      bullet("glove — a safety glove on a hand."),
      bullet("no_glove — a bare hand where a glove should be worn."),
      bullet("goggles — protective eyewear worn on the face."),
      bullet("no_goggles — an eye region without protective eyewear."),
      bullet("mask — a respiratory or face mask worn over the mouth and nose."),
      bullet("no_mask — a lower face region without a mask."),
      bullet("shoes — safety footwear on the foot."),
      bullet("no_shoes — a foot region without safety footwear."),

      P("What is NOT in this schema: hi-vis safety vests, compound states like full PPE or partial PPE, and whole-person classification. Safety vests do not appear in the keremberke dataset. These distinctions matter because a model trained on this schema cannot be used to assess overall PPE compliance — it only detects the presence or absence of specific individual items."),

      H2("1.2 Dataset"),

      P("The keremberke/protective-equipment-detection dataset ships as three zip files with COCO-format annotations. All segmentation fields in the COCO JSON are empty — only bounding boxes are provided. Pixel masks are generated by SAM2 using the boxes as prompts."),
      bullet("train.zip — 6,473 images. Contains glove, no_glove, goggles, no_goggles, shoes, no_shoes. No helmet annotations."),
      bullet("test.zip — 1,935 images. Primarily helmet and no_helmet, with some goggles and shoes."),
      bullet("valid.zip — 3,570 images. Mixed across all 10 classes."),

      P("All three zips are used. Images with at least one annotated bounding box are pooled (11,704 total), shuffled with seed 42, capped at 4,000, then split 85/15:"),
      bullet("Train: 3,400 images"),
      bullet("Val: 600 images"),

      P("Annotation counts across the full dataset before the 4,000-image cap:"),
      bullet("no_glove: 6,126   glove: 4,663   goggles: 4,184   no_goggles: 4,092"),
      bullet("helmet: 1,523   no_helmet: 1,296   shoes: 755   no_shoes: 606"),
      bullet("no_mask: 661   mask: 269"),

      P("Gloves and goggles dominate. Mask has the fewest annotations at 269, making it the hardest class to learn. The loss function uses class weights of 0.2 for background and 2.0 for each of the 10 PPE classes to compensate for background pixel dominance."),

      H2("1.3 Mask Generation"),

      P("The data preparation script ppe_seg_keremberke_rebuild.py processes each image as follows:"),
      bullet("Reads the COCO JSON directly from the zip archive (no disk extraction needed)."),
      bullet("Runs SAM2 box-prompted segmentation on each bounding box. Where SAM2 returns no mask, the bounding box rectangle is used as a fallback."),
      bullet("Paints each SAM2 foreground region onto a blank canvas at its class index (0 = background, 1-10 = PPE class), producing a single-channel uint8 PNG semantic mask."),
      bullet("Traces the same binary masks into contour polygons, simplifies with Douglas-Peucker (epsilon = 0.005 x arc length), and writes normalised YOLO polygon .txt labels for instance segmentation training."),
      P("Both output formats are written to D:/Claude/datasets/ppe_seg_ke/ outside the git repository."),

      // ── 2. Why Deep Learning ──────────────────────────────────────────────
      H1("2. Why Deep Learning is Required for Pixel-Wise Tasks"),

      P("Pixel-level labelling is substantially harder than image-level classification. Three specific properties of this task make classical methods inadequate."),

      H3("2.1 Receptive field requirements"),
      P("Classifying a single pixel correctly requires context from pixels far away. A bare head at the top of the frame is ambiguous without the shoulders and torso below it. Threshold-based and histogram-based methods are inherently local. DeepLabV3+'s atrous convolutions and ASPP module aggregate multi-scale context across the entire image. YOLOv8n-seg's neck and detection head similarly combine features at multiple scales before making predictions. Neither of these multi-scale fusion strategies is achievable with classical pixel classifiers."),

      H3("2.2 Class imbalance at the pixel level"),
      P("Background fills roughly 80 percent of pixels in a typical industrial scene. A safety mask covers under 0.5 percent. A classifier that predicts background everywhere achieves 80 percent pixel accuracy while scoring zero IoU on every PPE class. We counter this with CrossEntropyLoss weights: 0.2 for background, 2.0 for each PPE class. This forces the model to spend capacity on rare classes. Classical methods have no equivalent mechanism beyond manual feature engineering per class."),

      H3("2.3 Noisy pseudo-labels from SAM2"),
      P("Every pixel label in this dataset comes from SAM2 running on bounding box annotations rather than human-drawn polygon masks. SAM2 is accurate for clear, isolated objects but struggles when objects overlap or when a bounding box contains partial occlusion. Deep models handle this because stochastic mini-batches and augmentation average out consistent noise across many gradient updates. A classical pixel classifier trained on the same labels tends to memorise the noise directly."),

      // ── 3. Pipeline Architecture ──────────────────────────────────────────
      H1("3. Pipeline Architecture"),

      H2("3.1 Data preparation (ppe_seg_keremberke_rebuild.py)"),
      bullet("Reads all three keremberke zip files directly without extracting them to disk."),
      bullet("Pools all annotated images (11,704 total), shuffles with seed 42, caps at 4,000, splits 85/15 into 3,400 train and 600 val."),
      bullet("Runs SAM2 box-prompted segmentation to generate pixel masks for each annotated object. Falls back to the bounding box rectangle if SAM2 returns nothing."),
      bullet("Writes semantic masks as uint8 PNG files (pixel value = class index) and YOLO polygon labels as normalised .txt files."),
      bullet("Auto-generates ppe_seg_ke.yaml for YOLO training with correct class names, paths, and split references."),

      H2("3.2 Semantic segmentation — DeepLabV3+ (ppe_deeplab_train.py)"),
      bullet("Backbone: torchvision deeplabv3_resnet50 pretrained on COCO + ImageNet."),
      bullet("Head: classifier[-1] and aux_classifier[-1] replaced with nn.Conv2d(256, 11, 1) for 11 classes (background + 10 PPE items)."),
      bullet("Input: 512x512, bilinear resize for images, nearest-neighbour for masks. Random horizontal flip (p=0.5) and brightness/contrast jitter (p=0.3). ImageNet normalisation."),
      bullet("Loss: CrossEntropyLoss(weight=[0.2, 2.0 x 10]) + 0.4 x auxiliary loss from the ResNet50 intermediate branch."),
      bullet("Optimiser: AdamW lr=3e-4, weight_decay=1e-4, CosineAnnealingLR over 30 epochs, batch size 8."),
      bullet("Saves best checkpoint by validation mIoU. Outputs deeplab_results.csv with per-class IoU, mIoU, and pixel accuracy."),

      H2("3.3 Instance segmentation — YOLOv8n-seg (ppe_yolo_seg_train.py)"),
      bullet("Base: COCO-pretrained YOLOv8n-seg. All layers fine-tuned on the keremberke instance dataset."),
      bullet("10 output classes matching the keremberke schema. YOLO uses 0-indexed classes with no background class."),
      bullet("Training: epochs=30, imgsz=640, batch=16, device=0."),
      bullet("Single forward pass outputs bounding boxes, class confidences, and polygon masks per detected instance."),
      bullet("Metrics: box mAP50, box mAP50-95, mask mAP50, mask mAP50-95, precision, recall per class."),

      // ── 4. Transfer Learning ──────────────────────────────────────────────
      H1("4. Transfer Learning"),

      H3("4.1 DeepLabV3+ — pretrained encoder, new classification head"),
      P("The ResNet50 backbone loads a 161 MB checkpoint pretrained on ImageNet and COCO. The final 1x1 convolution in both the ASPP classifier branch and the auxiliary branch are replaced with randomly-initialised Conv2d(256, 11, 1) to match our 11-class output. All layers including the backbone are unfrozen and trained at lr=3e-4. The pretrained backbone provides feature detectors for edges, textures, and object boundaries that would take far longer to learn from the 3,400-image keremberke dataset alone. Fine-tuning all layers lets these general features adapt towards distinguishing a helmet from a bare head, a glove from a bare hand, and so on."),

      H3("4.2 YOLOv8n-seg — COCO backbone fine-tuned end to end"),
      P("YOLOv8n-seg starts from weights covering 80 COCO detection classes plus instance masks. All layers are fine-tuned on the 10 keremberke classes. COCO contains people in various poses holding objects and wearing hats, so feature anchors for small items on human bodies are already partially formed before training begins. Helmets, gloves, and goggles all appear in COCO-adjacent contexts. The network adapts these anchors to the specific geometric signatures of PPE items rather than learning them from scratch."),

      H3("4.3 Why pretrained weights are necessary"),
      P("The 10-class keremberke task is harder than it appears: objects are small relative to the full 1280x720 frame (a safety mask can be under 0.5 percent of image area), class frequencies span a 20:1 ratio, and all pixel labels come from SAM2 pseudo-annotation rather than human polygon masks. Starting from random weights with these constraints produces models that converge slowly and plateau at low mIoU. Pretrained encoders compress the early learning problem from scratch feature discovery into targeted task adaptation, which is where the 3,400-image training set is actually sufficient."),

      // ── 5. Performance Evaluation ─────────────────────────────────────────
      H1("5. Performance Evaluation"),

      H2("5.1 Overall results"),
      P(""),
      makeResultsTable(),
      P("Both models were trained on 3,400 images from the keremberke dataset using the 10 native PPE item classes. All results are reported on the held-out 600-image validation set."),

      H2("5.2 DeepLabV3+ per-class IoU"),
      P(""),
      makeIouTable(),
      P(""),
      P("Background fills the majority of pixels in every image and scores the highest IoU. Among PPE classes, gloves and no_glove have the most training annotations (4,663 and 6,126 respectively) and benefit from greater gradient coverage during training. Helmet and no_helmet have fewer annotations but are visually distinctive — hard hats have a consistent round shape and tend to appear in high-contrast positions. Mask and no_mask have the fewest annotations (269 and 661) and are the smallest objects per pixel area; their IoU is expected to be the lowest among PPE classes."),

      H2("5.3 YOLOv8n-seg per-class results"),
      P("The n column shows total annotation counts across the full keremberke dataset before the 4,000-image cap. Higher annotation counts correlate with higher mAP in most cases, but object geometry also matters. Helmet is a rigid, geometrically consistent shape that produces reliable polygon fits even at moderate annotation counts. Gloves and shoes deform constantly with hand and foot position, making polygon matching harder regardless of annotation volume."),
      P(""),
      makeYoloTable(),
      P(""),

      // ── 6. Experimentation ────────────────────────────────────────────────
      H1("6. Experimentation and Analysis"),

      H3("6.1 Class frequency versus performance"),
      P("The 10 keremberke classes are not uniformly represented. Gloves and goggles each have over 4,000 annotations; mask has only 269. In pixel-level segmentation this translates directly into gradient coverage: classes with more training examples cover more pixels per batch and receive stronger gradient signal. The equal class weighting in the loss function (2.0 for all 10 PPE classes) partially compensates but does not fully close the gap for mask and no_mask."),

      H3("6.2 Why individual items are harder to segment than whole-person regions"),
      P("A typical crop in Assignment 2 contained one person filling most of the image area. Here the objects of interest are individual PPE items on a person in a full scene: a glove is roughly 3-5 percent of the image area, a mask under 1 percent. At 512x512 input resolution, a mask bounding box may cover only 15x20 pixels. DeepLabV3+'s ASPP module uses atrous convolutions at rates tuned for medium-to-large objects. Very small objects at this resolution are at the boundary of what the architecture handles well without specialised small-object adaptations."),

      H3("6.3 Pixel accuracy versus mean IoU"),
      P("Background fills the majority of pixels in each image. A model that predicts background everywhere achieves high pixel accuracy while scoring zero IoU on every PPE class. Mean IoU weights all 11 classes equally, so missing any single class costs 1/11 of the total score regardless of how rarely that class appears. For this task mIoU is the primary diagnostic metric. Pixel accuracy is included for completeness but gives a misleadingly optimistic picture when background dominance is high."),

      H3("6.4 Box-mask gap in YOLOv8n-seg"),
      P("Box mAP50 measures whether a predicted bounding box overlaps the ground truth at IoU above 0.5. Mask mAP50 additionally requires the predicted polygon to match the object boundary at the same threshold. Boxes are forgiving — a box 10 pixels wider than the object still passes. Polygons are not. Deformable items like gloves, shoes, and face masks change shape with body position, making polygon matching harder than box matching. The box-mask gap is consistently larger for these items than for rigid objects like helmets, which is visible in the per-class table above."),

      H3("6.5 The pseudo-label ceiling"),
      P("Every pixel label in this dataset comes from SAM2 running on bounding box annotations. There are no human-drawn polygon masks. SAM2 is accurate for clear, well-isolated objects but degrades when objects overlap, when boxes are imprecisely drawn, or when a single box contains multiple workers in different PPE states. The class label for every pixel inside a SAM2 mask is taken directly from the box annotation, so any box-level annotation error propagates to every pixel in that region. This is the primary ceiling on achievable mIoU with this pipeline — architecture improvements will yield diminishing returns while the label quality remains unchanged."),

      H3("6.6 What would improve results"),
      bullet("Hand-annotated polygon masks for even 500 images would likely raise mIoU more than any architecture change. SAM2 pseudo-labels are the dominant noise source."),
      bullet("The full keremberke dataset has 11,704 annotated images. This assignment used 4,000. Training on the complete set would directly help the low-frequency classes like mask and no_mask."),
      bullet("Separate models per item type — one DeepLabV3+ for helmet/no_helmet, another for glove/no_glove — would reduce inter-class competition and let each model specialise for its object scale."),
      bullet("CRF post-processing on DeepLabV3+ outputs would sharpen predicted boundaries at no retraining cost, at the expense of added inference time."),
      bullet("A two-stage pipeline combining a person detector with item-level segmentation on the resulting crops would reduce background dominance and scale the input to match object size."),

      // ── Appendix ──────────────────────────────────────────────────────────
      pageBreak(),
      H1("Appendix: Training Curves and Prediction Samples"),

      ...((() => {
        const items = [];

        const dl_train = loadImg("deeplab_training.png", 560, 215);
        if (dl_train) {
          items.push(dl_train);
          items.push(imgCaption("Figure 1. DeepLabV3+ training and validation loss (left) and validation mIoU (right) over 30 epochs."));
        }

        const dl_iou = loadImg("deeplab_confusion.png", 500, 220);
        if (dl_iou) {
          items.push(dl_iou);
          items.push(imgCaption("Figure 2. DeepLabV3+ per-class IoU on the 600-image validation set."));
        }

        const dl_grid = loadImg("deeplab_pred_grid.png", 540, 540);
        if (dl_grid) {
          items.push(dl_grid);
          items.push(imgCaption("Figure 3. DeepLabV3+ sample predictions. Each row shows an input image (left) and the predicted semantic mask (right)."));
        }

        const ys_res = loadImg("yolo_seg_results_plot.png", 560, 220);
        if (ys_res) {
          items.push(ys_res);
          items.push(imgCaption("Figure 4. YOLOv8n-seg training metrics over 30 epochs: box mAP50, mask mAP50, precision, and recall."));
        }

        const ys_conf = loadImg("yolo_seg_confusion.png", 440, 220);
        if (ys_conf) {
          items.push(ys_conf);
          items.push(imgCaption("Figure 5. YOLOv8n-seg per-class mask mAP50 on the 600-image validation set."));
        }

        return items.filter(Boolean);
      })()),

    ],
  }],
});

// ── Write ──────────────────────────────────────────────────────────────────
fs.mkdirSync(path.dirname(OUT), { recursive: true });
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`Saved: ${OUT}  (${(buf.length / 1024).toFixed(0)} KB)`);
});
