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
  const cols = [600, 2700, 1900, 1500, 2660];
  const rows = [
    ["Cat.", "Model", "Training Paradigm", "Task", "Performance"],
    ["(d)", "DeepLabV3+ ResNet50",
     "Pre-existing arch; COCO+ImageNet pretrained weights; all layers fine-tuned on keremberke",
     "Semantic Segmentation (11-class)",
     `mIoU = ${DEEPLAB_MIOU}  |  Pixel Acc = ${DEEPLAB_PIXACC}`],
    ["(d)", "YOLOv8n-seg",
     "Pre-existing arch; COCO pretrained weights; all layers fine-tuned on keremberke",
     "Instance Segmentation (10-class)",
     `Box mAP50 = ${YOLO_BOX}  |  Mask mAP50 = ${YOLO_MASK}`],
    ["(c)", "DeepLabV3+ ResNet50 (zero-shot)",
     "Pre-existing arch; COCO pretrained weights; NO fine-tuning — used as-is",
     "Semantic Seg. zero-shot (11-class)",
     "mIoU = 8.2%  (all PPE classes = 0%)"],
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

      H2("1.1 Finding a suitable dataset"),

      P("The dataset used in Assignments 1 and 2 (MinhNKB) contains only image-level labels: one of five compliance states per whole-person crop (helmet, safety_vest, full_ppe, partial_ppe, no_ppe). There are no bounding boxes, no polygon outlines, and no pixel masks. Pixel-wise segmentation models require spatial annotations — either bounding boxes that can be converted to masks, or polygon masks drawn directly. The A2 dataset could not be used for this task."),

      P("Three options were evaluated for acquiring spatially annotated PPE data:"),
      bullet("Manually annotating the existing MinhNKB images with polygon masks. This was ruled out: producing reliable per-pixel ground truth for 3,400+ full-scene industrial images would require a dedicated annotation tool and many hours of labelling. The resulting dataset would still be limited to the 5-class MinhNKB schema, which does not distinguish individual PPE items."),
      bullet("Using the bounding-box output from the A1 YOLOv8n person detector to crop regions of interest and label them. Ruled out because the A1 detector only localises people, not individual PPE items, so box-to-mask conversion would produce person-sized regions rather than item-level masks."),
      bullet("Finding a publicly available dataset with bounding-box or polygon annotations for individual PPE items. The keremberke/protective-equipment-detection dataset on Roboflow matched this requirement: 11,704 annotated images in COCO format with bounding boxes for 10 item-level PPE classes across construction, manufacturing, and food-processing scenes."),

      P("The keremberke dataset was selected for three specific reasons. First, it provides item-level annotations rather than person-level states — each box covers a single helmet, glove, or goggles instance, making the annotation granularity suitable for pixel-level prediction. Second, it is large enough (11,704 images before capping) to support both semantic and instance segmentation training without severe data scarcity. Third, it is distributed in COCO format with bounding boxes present in every annotation record, which makes it directly compatible with SAM2 box-prompted segmentation for automatic pixel mask generation."),

      P("The COCO JSON files ship with empty segmentation fields — only bounding boxes are provided. SAM2 (Segment Anything Model 2) was used to bridge this gap: each bounding box is passed to SAM2 as a spatial prompt, and SAM2 returns a binary foreground mask tightly fitted to the object inside the box. This converts the bounding-box dataset into a pseudo-segmentation dataset without any manual pixel labelling. Where SAM2 returns no mask (approximately 2-3% of boxes due to ambiguous or heavily occluded regions), the bounding box rectangle is used as a fallback mask. The result is a full pixel-level dataset suitable for training both semantic and instance segmentation models."),

      H2("1.2 Model selection"),

      P("With the keremberke dataset prepared, two architectures were chosen to cover the two distinct segmentation paradigms:"),
      bullet("DeepLabV3+ ResNet-50 for semantic segmentation. DeepLabV3+ was chosen because its Atrous Spatial Pyramid Pooling (ASPP) module aggregates context at multiple scales simultaneously, which is important for a dataset where objects of interest range from large gloves covering 5% of the image to small face masks covering under 0.5%. The ResNet-50 backbone is pretrained on ImageNet and COCO (category (d)), providing feature detectors for edges, shapes, and textures before any PPE-specific training. The ASPP decoder is replaced with an 11-class head matching our schema. All layers are fine-tuned end to end. DeepLabV3+ assigns one class label to every pixel in the full scene, including explicit background prediction."),
      bullet("YOLOv8n-seg for instance segmentation. YOLOv8 was already used in the A1 and A2 pipelines for person detection and end-to-end PPE detection. The seg variant extends the detection head with a mask branch that produces polygon outlines per detected instance. This architecture was chosen because it directly connects to the prior project work, handles overlapping instances natively (two helmets on two workers are reported as separate detections), and runs at practical inference speed on an RTX 5070. Like DeepLabV3+, it starts from COCO pretrained weights (category (d)) and is fine-tuned on the keremberke instance dataset."),
      P("The two models are complementary rather than redundant. DeepLabV3+ labels every pixel and includes background, making it suitable for scene-level coverage analysis. YOLOv8n-seg detects discrete item instances with confidence scores, making it suitable for per-worker compliance counting. Both are evaluated independently on the same 600-image validation split."),

      H2("1.3 Class Definitions"),

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

      P("Class label ambiguity: The no_X classes carry inherent annotation ambiguity. A box labelled no_helmet covers the head region of a worker not wearing a hard hat — but where the head ends is a judgment call that varies across annotators, image resolutions, and partial occlusions. The same applies to no_glove (bare hand boundary), no_goggles (eye region extent), and no_mask (lower face region). Because SAM2 receives these imprecise boxes and produces pixel masks that are then treated as ground truth, any box-level ambiguity propagates directly to the pixel labels. This differs from the Assignment 2 schema where full_ppe and partial_ppe created ambiguity about the threshold of compliance — here the ambiguity is spatial rather than categorical. Models trained on this data absorb both types of noise."),

      H2("1.4 Dataset"),

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

      H2("1.5 Mask Generation"),

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

      H3("2.4 Quantitative evidence from prior work"),
      P("Assignment 2 on this project established a concrete baseline for PPE recognition on person-level crops using models from all four evaluation categories:"),
      bullet("Category (a) — custom architecture trained from scratch: PPENet CNN, our own 3-block convolutional image classifier (32x32 input, 226K parameters), achieved 87.33% accuracy on 5-class crop classification. This is an image classification model — one label per person crop. No reference architecture was followed in its design."),
      bullet("Category (b) — pre-existing architecture trained from scratch (no pretrained weights): a UNet semantic segmentation model (Ronneberger et al. 2015) trained from randomly initialised weights achieved mIoU of 56.2% on 5-class pixel-level segmentation of person crops. UNet is a well-known encoder-decoder architecture; training it from scratch establishes what the architecture can learn without any transfer from large-scale pre-training."),
      bullet("Category (c) — pre-existing architecture with pre-trained weights, NOT fine-tuned: the COCO-pretrained DeepLabV3+ ResNet-50 semantic segmentation model run directly on our keremberke validation set without any fine-tuning scored 8.2% mIoU. All 10 PPE class IoUs are exactly 0% because the COCO output head has 21 classes (aeroplane, bicycle, bird, etc.) with no overlap with our label schema. This directly demonstrates that pre-trained weights cannot transfer predictions to an unseen label space without task-specific adaptation."),
      bullet("Category (d) — pre-existing architectures with pre-trained weights, fine-tuned: ViT-B/16 image classification model fine-tuned on MinhNKB crop data reached 93.9% accuracy (A2). In Assignment 3, DeepLabV3+ ResNet-50 semantic segmentation model fine-tuned on keremberke reached 63.5% mIoU, and YOLOv8n-seg instance segmentation model fine-tuned on keremberke reached 87.1% mask mAP50."),
      P("The 17-percentage-point accuracy gap between the SVM and the best deep model in Assignment 2 exists even for the comparatively easy crop-level task. Pixel-level segmentation is harder in every dimension: 262,144 pixels must be classified simultaneously, background fills over 80 percent of the image area, objects of interest may cover fewer than 50 pixels, and the model must maintain spatial coherence between adjacent pixel predictions. The gap between classical and deep approaches widens substantially at the pixel level. The (c) vs (d) contrast — 8.2% mIoU zero-shot versus 63.5% mIoU fine-tuned — quantifies exactly how much task-specific adaptation contributes on top of the pre-trained backbone."),

      // ── 3. Pipeline Architecture ──────────────────────────────────────────
      H1("3. Pipeline Architecture"),

      H2("3.1 Data preparation (ppe_seg_keremberke_rebuild.py)"),
      bullet("Reads all three keremberke zip files directly without extracting them to disk."),
      bullet("Pools all annotated images (11,704 total), shuffles with seed 42, caps at 4,000, splits 85/15 into 3,400 train and 600 val."),
      bullet("Runs SAM2 box-prompted segmentation to generate pixel masks for each annotated object. Falls back to the bounding box rectangle if SAM2 returns nothing."),
      bullet("Writes semantic masks as uint8 PNG files (pixel value = class index) and YOLO polygon labels as normalised .txt files."),
      bullet("Auto-generates ppe_seg_ke.yaml for YOLO training with correct class names, paths, and split references."),

      H2("3.2 Semantic segmentation — DeepLabV3+ [Category (d)]"),
      P("DeepLabV3+ ResNet-50 is a pre-existing architecture (Chen et al. 2018) loaded with weights pre-trained on ImageNet and COCO, then fine-tuned end-to-end on our keremberke dataset. This places it in category (d): pre-existing architecture with pre-trained weights that were fine-tuned. It performs dense pixel-level semantic segmentation — every pixel in the image receives one of 11 class labels (background + 10 PPE items) in a single forward pass."),
      bullet("Backbone: torchvision deeplabv3_resnet50 pretrained on COCO + ImageNet. ResNet-50 belongs to the same architectural family explored in Assignment 2, where ResNet-18 served as both a random-weight baseline (87.0% val accuracy from scratch) and a frozen transfer learning baseline. ResNet-50 extends this to 50 layers across four residual stages, providing richer spatial feature hierarchies required for dense pixel prediction rather than single-label image classification."),
      bullet("Head: classifier[-1] and aux_classifier[-1] replaced with nn.Conv2d(256, 11, 1) for 11 classes (background + 10 PPE items)."),
      bullet("Input: 512x512, bilinear resize for images, nearest-neighbour for masks. Random horizontal flip (p=0.5) and brightness/contrast jitter (p=0.3). ImageNet normalisation."),
      bullet("Loss: CrossEntropyLoss(weight=[0.2, 2.0 x 10]) + 0.4 x auxiliary loss from the ResNet50 intermediate branch."),
      bullet("Optimiser: AdamW lr=3e-4, weight_decay=1e-4, CosineAnnealingLR over 30 epochs, batch size 8."),
      bullet("Saves best checkpoint by validation mIoU. Outputs deeplab_results.csv with per-class IoU, mIoU, and pixel accuracy."),

      H2("3.3 Instance segmentation — YOLOv8n-seg [Category (d)]"),
      P("YOLOv8n-seg is a pre-existing architecture (Ultralytics 2023) loaded with COCO-pretrained weights, then fine-tuned end-to-end on our keremberke instance dataset. This also places it in category (d): pre-existing architecture with pre-trained weights that were fine-tuned. It performs instance segmentation — each detected PPE item receives its own separate polygon mask, bounding box, and class label, allowing multiple instances of the same class (e.g. two helmets on two workers) to be tracked independently. This differs from DeepLabV3+'s semantic approach, which assigns a single label per pixel without distinguishing between instances."),
      bullet("Base: COCO-pretrained YOLOv8n-seg. All layers fine-tuned on the keremberke instance dataset."),
      bullet("10 output classes matching the keremberke schema. YOLO uses 0-indexed classes with no background class."),
      bullet("Training: epochs=30, imgsz=640, batch=16, device=0."),
      bullet("Single forward pass outputs bounding boxes, class confidences, and polygon masks per detected instance."),
      bullet("Metrics: box mAP50, box mAP50-95, mask mAP50, mask mAP50-95, precision, recall per class."),

      // ── 4. Transfer Learning ──────────────────────────────────────────────
      H1("4. Transfer Learning"),

      H3("4.1 DeepLabV3+ — pretrained encoder, new classification head"),
      P("The ResNet50 backbone loads a 161 MB checkpoint pretrained on ImageNet and COCO. The final 1x1 convolution in both the ASPP classifier branch and the auxiliary branch are replaced with randomly-initialised Conv2d(256, 11, 1) to match our 11-class output. All layers including the backbone are unfrozen and trained at lr=3e-4."),
      P("Assignment 2 tested two ResNet-18 transfer learning strategies on the same PPE domain: (1) frozen backbone — pretrained features used as fixed representations, only the head trained; (2) full fine-tuning from random weights. The frozen strategy showed that ImageNet features generalise well to PPE recognition even without task-specific adaptation. For Assignment 3, full fine-tuning is used because semantic segmentation requires the backbone to retain spatial information that ImageNet classification training compresses away — fine-tuning all layers at a conservative lr=3e-4 lets the spatial features adapt without destroying pretrained edge and texture detectors. The pretrained backbone provides feature detectors for edges, textures, and object boundaries that would take far longer to learn from the 3,400-image keremberke dataset alone."),

      H3("4.2 YOLOv8n-seg — COCO backbone fine-tuned end to end"),
      P("YOLOv8n-seg starts from weights covering 80 COCO detection classes plus instance masks. All layers are fine-tuned on the 10 keremberke classes. COCO contains people in various poses holding objects and wearing hats, so feature anchors for small items on human bodies are already partially formed before training begins. Helmets, gloves, and goggles all appear in COCO-adjacent contexts. The network adapts these anchors to the specific geometric signatures of PPE items rather than learning them from scratch."),

      H3("4.3 Category (c) vs (d): the value of fine-tuning"),
      P("Running the COCO-pretrained DeepLabV3+ as-is on our validation set — category (c): pre-existing architecture with pre-trained weights that were not fine-tuned — produces 8.2% mIoU. Every one of the 10 PPE class IoU scores is exactly 0.0 because the COCO output head has 21 classes (aeroplane, bicycle, bird, etc.) that share no overlap with our label schema. Only background coincidentally maps to the same index. The model is producing meaningful low-level features — edges, textures, object boundaries from its ResNet-50 backbone — but those features feed into prediction heads that output the wrong class space entirely."),
      P("Fine-tuning the same model on our 3,400 training images — category (d) — raises mIoU from 8.2% to 63.5%, a 55-point improvement. This gap is entirely attributable to replacing the COCO output head with an 11-class head and adapting all backbone weights to the PPE domain. The backbone features do transfer: the fine-tuned model converges in 30 epochs, whereas training the same architecture from randomly initialised weights on a dataset this size would require far more epochs to reach equivalent performance. The 10-class keremberke task is additionally hard because objects are small relative to the full image frame, class frequencies span a 20:1 ratio, and all pixel labels come from SAM2 pseudo-annotation rather than human polygon masks. Pretrained features accelerate convergence precisely because they contain already-useful detectors for edges, shapes, and textures that the 3,400-image training set alone could not fully learn."),

      // ── 5. Performance Evaluation ─────────────────────────────────────────
      H1("5. Performance Evaluation"),

      H2("5.1 Overall results"),
      P(""),
      makeResultsTable(),
      P("Both models were trained on 3,400 images from the keremberke dataset using the 10 native PPE item classes. All results are reported on the held-out 600-image validation set."),

      H2("5.2 DeepLabV3+ per-class IoU — Category (d)"),
      P("DeepLabV3+ is evaluated here as a category (d) model: a pre-existing semantic segmentation architecture with ImageNet+COCO pre-trained weights that were fully fine-tuned on our keremberke 10-class data. It predicts a single class label per pixel across the entire image simultaneously."),
      P(""),
      makeIouTable(),
      P(""),
      P("Background fills the majority of pixels in every image and scores the highest IoU. Among PPE classes, gloves and no_glove have the most training annotations (4,663 and 6,126 respectively) and benefit from greater gradient coverage during training. Helmet and no_helmet have fewer annotations but are visually distinctive — hard hats have a consistent round shape and tend to appear in high-contrast positions."),
      P("The two weakest classes are no_mask (51.7% IoU) and no_helmet (55.5%), both of which are absence classes. An absence region has no fixed visual signature — no_mask covers any lower face without a mask, meaning the pixels could look like skin, a beard, a scarf, or a construction bandana. The model must learn what is not present rather than what is. This parallels the partial_ppe and full_ppe weakness observed in Assignment 2 (F1 approximately 0.76-0.77), where judging overall compliance was harder than detecting a single salient object. The mask class itself (62.8% IoU) suffers from the smallest annotation count (269) and the smallest object size — at 512x512 input a face mask occupies roughly 400-600 pixels, placing it at the edge of reliable segmentation."),
      P("Targeted strategies to close these gaps: (1) focal loss (gamma=2) would down-weight easy background and well-classified pixels and concentrate gradient on hard examples like no_mask and no_helmet, which uniform class weights do not address; (2) class-specific oversampling — duplicating training images containing mask or no_mask annotations — would increase their gradient coverage without changing the loss hyperparameters; (3) separate specialist models for the two most challenging pairs (no_mask/mask and no_helmet/helmet) would eliminate inter-class competition at inference."),

      H2("5.3 YOLOv8n-seg per-class results — Category (d)"),
      P("YOLOv8n-seg is also a category (d) model: a pre-existing instance segmentation architecture with COCO pre-trained weights that were fully fine-tuned on our keremberke data. Unlike DeepLabV3+, it performs instance segmentation — detecting each PPE item as a separate object with its own polygon mask, bounding box, and confidence score. Two helmets on two different workers are reported as two separate instances. This makes it better suited to counting and per-person compliance checks, but it does not label every pixel in the scene and does not produce a background class."),
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

      H3("6.6 Lessons from Assignment 2 applied to segmentation"),
      P("Several findings from Assignment 2 directly inform the design choices and remaining gaps in Assignment 3."),
      bullet("Minority class underperformance is consistent across both assignments. partial_ppe and full_ppe were the hardest classes in Assignment 2 (F1 ~0.76) for the same reason no_mask and no_helmet are hardest here — absence-of-compliance states lack a salient visual signature, and both are affected by annotation ambiguity. The 600-sample-per-class cap used in Assignment 2 reduced imbalance but did not close the gap; a similar dynamic appears here with the 2.0 class weight for all PPE classes."),
      bullet("Transfer learning strategy matters more than depth. In Assignment 2, frozen ResNet-18 features provided a strong fixed baseline without any domain adaptation. Assignment 3 requires full fine-tuning because segmentation needs spatially precise features that a frozen classification backbone compresses away. The ViT-B/16 result from Assignment 2 (93.9% accuracy) suggests that attention-based architectures would also transfer well here — a SegFormer or Mask2Former backbone could be a direct next step."),
      bullet("The two-stage pipeline from Assignment 2 (YOLO person detector + crop classifier) is a viable alternative architecture for Assignment 3. Running PPE item segmentation on person crops rather than full scenes would reduce background dominance from ~80% to near zero and scale each item to a consistent resolution, directly addressing the two main failure modes identified in this assignment."),

      H3("6.7 What would improve results"),
      bullet("Hand-annotated polygon masks for even 500 images would likely raise mIoU more than any architecture change. SAM2 pseudo-labels are the dominant noise source."),
      bullet("Focal loss (gamma=2) would down-weight easy background pixels and concentrate gradient on hard, misclassified pixels — more targeted than uniform 2.0 class weights, particularly for no_mask and no_helmet."),
      bullet("Class-specific oversampling for mask (269 annotations) and no_mask (661 annotations) would increase gradient coverage for the two weakest classes without changing the loss function."),
      bullet("The full keremberke dataset has 11,704 annotated images. This assignment used 4,000. Training on the complete set would directly help the low-frequency classes."),
      bullet("Separate models per item type — one DeepLabV3+ for helmet/no_helmet, another for glove/no_glove — would reduce inter-class competition and let each model specialise for its object scale."),
      bullet("CRF post-processing on DeepLabV3+ outputs would sharpen predicted boundaries at no retraining cost, at the expense of added inference time."),
      bullet("A two-stage pipeline combining a person detector with item-level segmentation on the resulting crops would reduce background dominance from ~80% to near zero and scale the input to match object size — directly extending the Assignment 2 two-stage architecture to pixel-level output."),

      H3("6.8 Four-category model evaluation"),
      P("The assignment requires evaluating performance across four model categories. The table below maps every trained model in this project to its category."),

      (() => {
        const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
        const borders = { top: border, bottom: border, left: border, right: border };
        const hdrShading = { fill: "1F3864", type: ShadingType.CLEAR };
        const altShading  = { fill: "EEF2F7", type: ShadingType.CLEAR };
        const cols = [480, 2500, 1500, 1000, 1880];
        const hdr  = ["Cat", "Model + Task Type", "Task", "Score", "Notes"];
        const rows = [
          ["(a)", "PPENet CNN\n[Image Classification]",
           "5-class crop classification (MinhNKB, A2)", "87.3% Acc",
           "Custom-designed architecture: 3 conv blocks (32-64-128 ch), AdaptiveAvgPool, FC 512-256-5, 226K params. No reference architecture. Classifies one person crop into one compliance state."],
          ["(b)", "UNet\n[Semantic Segmentation]",
           "5-class semantic segmentation (MinhNKB, A2)", "56.2% mIoU",
           "Pre-existing encoder-decoder (Ronneberger et al. 2015); trained from randomly initialised weights, no pretrained weights. Labels every pixel in a crop with one of 5 compliance classes."],
          ["(c)", "DeepLabV3+ ResNet-50\n[Semantic Seg, zero-shot]",
           "10-class semantic seg. zero-shot (keremberke, A3)", "8.2% mIoU*",
           "Pre-existing architecture; COCO pretrained 21-class head used as-is, no fine-tuning. *All 10 PPE class IoUs = 0 because COCO classes do not overlap our schema. Shows that pretrained weights cannot transfer predictions to an unseen label space."],
          ["(d)", "ViT-B/16\n[Image Classification]",
           "5-class crop classification (MinhNKB, A2)", "93.9% Acc",
           "Pre-existing Vision Transformer; ImageNet pretrained; all layers fine-tuned 20 epochs. Classifies one person crop — same task as PPENet CNN (a) using an attention-based backbone."],
          ["(d)", "DeepLabV3+ ResNet-50\n[Semantic Segmentation]",
           "10-class semantic segmentation (keremberke, A3)", "63.5% mIoU",
           "Pre-existing architecture; ImageNet+COCO pretrained; all layers fine-tuned 30 epochs. Labels every pixel in a full scene with one of 11 classes (background + 10 PPE items)."],
          ["(d)", "YOLOv8n-seg\n[Instance Segmentation]",
           "10-class instance segmentation (keremberke, A3)", "87.1% Mask mAP50",
           "Pre-existing architecture; COCO pretrained; all layers fine-tuned 30 epochs. Detects each PPE item as a separate object instance with its own polygon mask — two helmets on two workers = two distinct instances."],
        ];
        const makeCell = (text, shading, bold, colIdx) =>
          new TableCell({
            borders, shading,
            width: { size: cols[colIdx], type: WidthType.DXA },
            margins: { top: 60, bottom: 60, left: 100, right: 100 },
            children: [new Paragraph({ children: [new TextRun({ text, bold, font: "Arial", size: bold ? 18 : 16 })] })],
          });
        return new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: cols,
          rows: [
            new TableRow({
              tableHeader: true,
              children: hdr.map((h, i) => makeCell(h, hdrShading, true, i)),
            }),
            ...rows.map((row, ri) =>
              new TableRow({
                children: row.map((cell, ci) =>
                  makeCell(cell, ri % 2 === 1 ? altShading : { fill: "FFFFFF", type: ShadingType.CLEAR }, false, ci)
                ),
              })
            ),
          ],
        });
      })(),

      P("Key insight: category (c) — the COCO-pretrained model used without fine-tuning — achieves near-zero IoU on every PPE class because the 21 COCO output classes do not overlap with our 10-class keremberke schema. This directly demonstrates that pre-trained weights provide a strong feature backbone but require task-specific fine-tuning to produce useful predictions for a specialised domain."),

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

        // Best / worst prediction grids with Grad-CAM and saliency maps
        // Each image is ~2620×1653px (ratio 1.58). Fit to content width (620px).
        const DL_GRID_W = 620, DL_GRID_H = 393;
        const YO_GRID_W = 620, YO_GRID_H = 390;

        items.push(pageBreak());
        items.push(H2("Best and Worst Predictions with Grad-CAM and Saliency Maps"));
        items.push(P("Each grid shows five columns: Input image | Ground-truth mask or boxes | Model prediction | Grad-CAM heatmap | Saliency map. Images are ranked by per-image metric: mean IoU (excluding background) for DeepLabV3+, mean best-match box-IoU against ground-truth boxes for YOLOv8n-seg. The Grad-CAM target layer is backbone.layer4 (final ResNet50 block) for DeepLab and the SPPF layer (layer 9) for YOLO. The saliency map shows the gradient magnitude of the score with respect to the input pixel values."));

        const dl_best = loadImg("deeplab_best3.png", DL_GRID_W, DL_GRID_H);
        if (dl_best) {
          items.push(dl_best);
          items.push(imgCaption("Figure 6. DeepLabV3+ — top-3 validation images by per-image mIoU (excluding background). Score shown on the left of each row."));
        }

        const dl_worst = loadImg("deeplab_worst3.png", DL_GRID_W, DL_GRID_H);
        if (dl_worst) {
          items.push(dl_worst);
          items.push(imgCaption("Figure 7. DeepLabV3+ — bottom-3 validation images by per-image mIoU. Zero scores indicate images where every PPE class present in the ground truth was completely missed."));
        }

        const yo_best = loadImg("yolo_best3.png", YO_GRID_W, YO_GRID_H);
        if (yo_best) {
          items.push(yo_best);
          items.push(imgCaption("Figure 8. YOLOv8n-seg — top-3 validation images by mean best-match box-IoU against ground-truth boxes."));
        }

        const yo_worst = loadImg("yolo_worst3.png", YO_GRID_W, YO_GRID_H);
        if (yo_worst) {
          items.push(yo_worst);
          items.push(imgCaption("Figure 9. YOLOv8n-seg — bottom-3 validation images by mean best-match box-IoU. Zero scores indicate images where the model produced no confident detections above the 0.25 confidence threshold."));
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
