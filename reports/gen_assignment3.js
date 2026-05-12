/**
 * Generate Assignment 3 Write-up DOCX
 * PPE Detection — Pixel-Wise Segmentation
 */
"use strict";
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, LevelFormat, PageBreak, Header, Footer, PageNumber,
} = require("docx");
const fs = require("fs");
const path = require("path");

// ── Paths ──────────────────────────────────────────────────────────────────
const BASE = path.resolve(__dirname, "..");
const IMG  = path.join(BASE, "results", "models");
const OUT  = path.join(BASE, "docs", "Assignment3_Writeup.docx");

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
const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };
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

// ── Results table ──────────────────────────────────────────────────────────
// NOTE: fill DEEPLAB_MIOU, DEEPLAB_PIXACC, YOLO_BOX, YOLO_MASK after training
const DEEPLAB_MIOU   = "see Table 2";
const DEEPLAB_PIXACC = "see Table 2";
const YOLO_BOX       = "see Table 3";
const YOLO_MASK      = "see Table 3";

function makeResultsTable() {
  const cols = [3000, 2400, 1700, 2260]; // 9360 total
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

// ── Per-class IoU table (DeepLabV3+) ──────────────────────────────────────
// Fill IoU values after training completes
function makeIouTable() {
  const cols = [2600, 1760, 5000]; // 9360 total
  const data = [
    ["Class",        "IoU",  "Notes"],
    ["background",   "—",    "Dominant class — fills most pixels"],
    ["helmet",       "—",    "Small, round object; consistent shape"],
    ["no_helmet",    "—",    "Head region without hard hat"],
    ["glove",        "—",    "Small hand-area object; often occluded"],
    ["no_glove",     "—",    "Bare hand region"],
    ["goggles",      "—",    "Face-area object; variable shape"],
    ["no_goggles",   "—",    "Eye region without eyewear"],
    ["mask",         "—",    "Lower face region; small area"],
    ["no_mask",      "—",    "Exposed lower face"],
    ["shoes",        "—",    "Foot region; often partially visible"],
    ["no_shoes",     "—",    "Foot area without safety footwear"],
    ["Mean IoU",     "—",    "Average over all 11 classes including background"],
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

// ── YOLOv8-seg per-class table ─────────────────────────────────────────────
// Fill values after training completes
function makeYoloTable() {
  const cols = [2340, 1560, 1560, 1900, 1400, 600]; // 9360
  const data = [
    ["Class",      "Precision", "Recall", "Box mAP50", "Mask mAP50", "n"],
    ["helmet",     "—", "—", "—", "—", "1523"],
    ["no_helmet",  "—", "—", "—", "—", "1296"],
    ["glove",      "—", "—", "—", "—", "4663"],
    ["no_glove",   "—", "—", "—", "—", "6126"],
    ["goggles",    "—", "—", "—", "—", "4184"],
    ["no_goggles", "—", "—", "—", "—", "4092"],
    ["mask",       "—", "—", "—", "—", "269"],
    ["no_mask",    "—", "—", "—", "—", "661"],
    ["shoes",      "—", "—", "—", "—", "755"],
    ["no_shoes",   "—", "—", "—", "—", "606"],
    ["All",        "—", "—", "—", "—", "~24175"],
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

      // ── Title ────────────────────────────────────────────────────────────
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Assignment 3: Pixel-Wise PPE Detection", font: "Arial", size: 40, bold: true, color: "2E5F8A" })],
        spacing: { before: 480, after: 120 },
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Semantic Segmentation and Instance Segmentation for Industrial Safety Compliance", font: "Arial", size: 26, italics: true, color: "555555" })],
        spacing: { after: 80 },
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "CSCI 4930 / 6930  |  Deep Learning  |  Spring 2026", font: "Arial", size: 22, color: "777777" })],
        spacing: { after: 600 },
      }),

      // ── 1. Task Identification ────────────────────────────────────────────
      H1("1. Task Identification and Dataset"),

      P("Assignment 2 classified entire person crops into five compound PPE categories. Assignment 3 goes a level deeper: instead of one label per crop, we assign a label to every pixel. It also switches to a more granular schema — individual PPE item types rather than whole-person compound states. Two tasks:"),

      bullet("Semantic segmentation — label each pixel in a full scene as one of 11 classes: background plus the 10 keremberke PPE item classes. No person detector runs first; the whole frame is processed directly."),
      bullet("Instance segmentation — find each PPE item as a separate object with its own polygon mask, so a helmet on one worker and a helmet on another are tracked as two distinct instances."),

      P(""),
      H2("1.1 Class Definitions"),

      P("The keremberke dataset labels individual PPE items and their absence. Each bounding box covers a single item on a single worker — not a whole-person state. The 10 classes are:"),

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

      P("What is NOT in this schema: hi-vis safety vests, compound states like 'full PPE' or 'partial PPE', and whole-person classification. Those were Assignment 2 categories from the MinhNKB dataset. This assignment uses a different dataset with different, more specific labels. Safety vests do not appear in keremberke at all."),

      P(""),
      H2("1.2 Dataset"),

      P("The keremberke/protective-equipment-detection dataset (Roboflow, COCO format) ships as three zip files. All annotations are bounding boxes — segmentation fields are empty, so SAM2 generates pixel masks from the boxes."),

      bullet("train.zip — 6473 images. Contains glove, no_glove, goggles, no_goggles, shoes, no_shoes only. No helmets."),
      bullet("test.zip — 1935 images. Contains primarily helmet and no_helmet, plus goggles, shoes."),
      bullet("valid.zip — 3570 images. Mixed across all 10 classes."),

      P("All three zips are used. Images with at least one annotated box are pooled (11,704 total), shuffled with seed 42, capped at 4,000, then split 85/15 train/val:"),
      bullet("Train: 3,400 images"),
      bullet("Val: 600 images"),

      P("Annotation counts across the full dataset (all zips before capping):"),
      bullet("no_glove: 6,126   glove: 4,663   goggles: 4,184   no_goggles: 4,092"),
      bullet("helmet: 1,523   no_helmet: 1,296   no_shoes: 606   shoes: 755"),
      bullet("no_mask: 661   mask: 269"),

      P("Gloves and goggles dominate. Mask has the fewest annotations at 269. The class weights in the loss function ([0.2 for background, 2.0 for each PPE class]) treat all 10 PPE classes equally regardless of frequency."),

      P(""),
      H2("1.3 Mask generation"),

      P("The data preparation script (ppe_seg_keremberke_rebuild.py) does the following for each image:"),
      bullet("Reads the COCO JSON from the zip file to get bounding boxes and class names."),
      bullet("Runs SAM2 box-prompted segmentation on each bounding box. If SAM2 returns no mask, the bounding box rectangle is used as a fallback."),
      bullet("Paints each SAM2 foreground region onto a blank canvas at its class index (0 = background, 1–10 = PPE class), producing a single-channel uint8 PNG mask."),
      bullet("Traces the same binary masks into contour polygons, simplifies with Douglas-Peucker (epsilon = 0.005 × arc length), writes as normalised YOLO polygon .txt labels."),
      P(""),

      P("Both formats are written to D:/Claude/datasets/ppe_seg_ke/ outside the git repository."),

      // ── 2. Why DL ────────────────────────────────────────────────────────
      H1("2. Why Deep Learning is Required for Pixel-Wise Tasks"),

      P("The classification problem in Assignment 2 already ruled out classical methods. Segmentation makes things harder in three specific ways."),

      H3("2.1 Receptive field requirements"),
      P("Classifying a single pixel correctly often requires context from hundreds of pixels away. A hard-hat at the top of the frame is ambiguous without the shoulders and torso below it. Threshold-based and histogram-based methods are local by construction. DeepLabV3+'s atrous convolutions and ASPP module build multi-scale context across the full image, which is not something you can bolt onto a classical method."),

      H3("2.2 Class imbalance at the pixel level"),
      P("Background makes up roughly 80 percent of pixels in a typical industrial scene. Helmets can be under 0.5 percent. A naive pixel classifier learns to predict background for everything and still gets 80 percent accuracy. We use CrossEntropyLoss with weights [0.2, 2.0, 2.0, 2.0, 2.0, 2.0] (background vs each PPE class) to force the model to spend capacity on the rare classes. Classical methods have no equivalent mechanism."),

      H3("2.3 Noisy pseudo-labels"),
      P("Our masks come from SAM2, not human annotators. They are imperfect: SAM2 sometimes over- or under-segments a box region, and a single class label is assigned to the whole mask even when a box contains workers in different PPE states. Deep models handle this reasonably because stochastic mini-batches and augmentation average out consistent noise across many gradient updates. A classical pixel classifier trained on the same labels tends to fit the noise directly."),

      // ── 3. Pipeline ──────────────────────────────────────────────────────
      H1("3. Pipeline Architecture"),

      P("The Assignment 3 pipeline is designed to reuse the maximum amount of infrastructure from Assignment 2. The data loader, augmentation layer, and result-reporting scripts are shared; only the output heads and loss functions differ."),

      H2("3.1 Data preparation (ppe_seg_keremberke_rebuild.py)"),
      bullet("Reads all three keremberke zip files (train.zip, test.zip, valid.zip) directly from disk without extracting them."),
      bullet("Pools all images with at least one annotated box (11,704 total), shuffles with seed 42, caps at 4,000, splits 85/15."),
      bullet("For each image, runs SAM2 box-prompted segmentation on every bounding box to generate a pixel mask. Falls back to the bounding box rectangle if SAM2 returns nothing."),
      bullet("Writes semantic masks (uint8 PNG, class index = pixel value) and YOLO polygon labels (.txt, normalised 0-1 coordinates)."),
      bullet("Auto-generates ppe_seg_ke.yaml for YOLO training. Final dataset: 3,400 train / 600 val."),

      H2("3.2 Semantic segmentation — DeepLabV3+ (ppe_deeplab_train.py)"),
      bullet("Backbone: torchvision deeplabv3_resnet50, pretrained on COCO + ImageNet."),
      bullet("Head: model.classifier[-1] and model.aux_classifier[-1] replaced with nn.Conv2d(256, 11, 1) for 11 classes (background + 10 PPE items)."),
      bullet("Input: 512x512, bilinear resize for images, NEAREST for masks. Random horizontal flip (p=0.5), random brightness/contrast jitter (p=0.3 each). ImageNet normalisation."),
      bullet("Loss: CrossEntropyLoss(weight=[0.2, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) + 0.4 × aux loss."),
      bullet("Optimiser: AdamW lr=3e-4, weight_decay=1e-4, CosineAnnealingLR, 30 epochs, batch 8."),
      bullet("Metrics: per-class IoU (11 classes), mean IoU, pixel accuracy."),

      H2("3.3 Instance segmentation — YOLOv8n-seg (ppe_yolo_seg_train.py)"),
      bullet("COCO-pretrained YOLOv8n-seg fine-tuned on the keremberke instance dataset."),
      bullet("10 output classes matching the keremberke schema (YOLO 0-indexed, no background class)."),
      bullet("Training: epochs=30, imgsz=640, batch=16, device=0."),
      bullet("Single forward pass outputs boxes, class scores, and polygon masks per instance."),
      bullet("Metrics: box mAP50/50-95, mask mAP50/50-95, precision, recall per class."),

      H2("3.4 What was reused from Assignment 2"),
      P("The comparison and reporting infrastructure carries over with minor extensions:"),
      bullet("ppe_experiment_comparison.py — updated CSV loaders for the new keremberke results."),
      bullet("generate_assignment_report.py — table rows and pixel-wise section updated for the new schema."),
      bullet("results/models/ naming convention, git worktree workflow, and .gitignore rules for large checkpoints all unchanged."),

      // ── 4. Transfer Learning ─────────────────────────────────────────────
      H1("4. Transfer Learning"),

      P("Both models start from pretrained weights. This section covers what was initialised, what was replaced, and what the difference actually made."),

      H3("4.1 DeepLabV3+ — pretrained encoder, new head"),
      P("The ResNet50 backbone loads a 161 MB checkpoint trained on ImageNet and COCO PASCAL VOC. The last 1x1 convolution in the ASPP classifier and in the auxiliary branch are replaced with randomly-initialised Conv2d(256, 11, 1) for our 11 classes (background + 10 PPE items). All backbone layers are left unfrozen and trained at lr=3e-4. The backbone already knows edges, textures, and object boundaries — we direct those features towards distinguishing a helmet from a bare head, a glove from a bare hand, and so on."),

      H3("4.2 What happens without pretrained weights"),
      P("On a prior run with the smaller MinhNKB-only 1368-image dataset, training DeepLabV3+ from random initialisation kept mIoU below 0.15 for all 30 epochs. The pretrained version reached 0.483 on the same data. The keremberke dataset is larger (3400 training images) but the 10-class task is harder — more classes, smaller objects per class — so pretrained weights are equally essential here."),

      H3("4.3 YOLOv8n-seg — COCO backbone fine-tuned"),
      P("YOLOv8n-seg starts from COCO weights covering 80 detection classes plus instance masks. All layers are fine-tuned on the 10 keremberke classes. COCO contains people holding objects, wearing hats, and in various poses, so the feature anchors for small items on human bodies are already partially formed. Helmets, gloves, and goggles all appear in COCO-adjacent contexts even if not under the same names."),

      // ── 5. Performance Evaluation ─────────────────────────────────────────
      H1("5. Performance Evaluation"),

      H2("5.1 Overall results"),
      P(""),
      makeResultsTable(),
      P("Both models were trained on 3,400 images from the keremberke dataset using the 10 native PPE item classes. Results above are from the 600-image validation set."),

      H2("5.2 DeepLabV3+ per-class IoU"),
      P(""),
      makeIouTable(),
      P(""),

      P("Background makes up the majority of pixels in every image and should score the highest IoU. Among PPE item classes, glove and no_glove have the most training annotations (4,663 and 6,126 respectively) and are expected to score well. Helmet and no_helmet have fewer annotations (1,523 and 1,296) but are visually distinctive — hard hats have a consistent shape and colour. Mask and no_mask have the fewest annotations (269 and 661) and are the smallest objects; their IoU is expected to be lowest."),

      H2("5.3 YOLOv8n-seg per-class results"),
      P("The 'n' column shows total annotation counts across the full keremberke dataset before the 4,000-image cap. Higher annotation counts generally correlate with higher mAP, but object size and visual consistency also matter — a helmet is a more geometrically stable shape than a glove or shoe, so it tends to get a better polygon fit even with fewer examples."),
      P(""),
      makeYoloTable(),
      P(""),

      // ── 6. Experimentation ───────────────────────────────────────────────
      H1("6. Experimentation and Analysis"),

      H3("6.1 Class frequency vs performance"),
      P("The 10 keremberke classes are not uniformly represented. Gloves and goggles (positive and negative) each have over 4,000 annotations in the full dataset; mask has only 269. In pixel-level tasks this translates directly: classes with more training examples cover more pixels during training and produce stronger gradients. Helmet and no_helmet sit in the middle at ~1,300-1,500 annotations each — enough to be learnable but not dominant."),

      H3("6.2 Why individual items are harder to segment than whole-person crops"),
      P("Assignment 2 drew a bounding box around an entire person. The person filled most of the crop. Here the bounding boxes are around individual items — a glove is roughly 5 percent of the image area, a mask even less. At 512x512 input resolution, a mask box might cover 20x15 pixels. DeepLabV3+'s ASPP module uses atrous rates tuned for medium-to-large objects. Small items at the limit of the input resolution are the hardest case."),

      H3("6.3 Pixel accuracy vs mean IoU"),
      P("Background fills most pixels in each image. A model that predicts background everywhere achieves high pixel accuracy while scoring 0 IoU on every PPE class. Mean IoU weights all 11 classes equally, so ignoring any one class costs roughly 1/11 of the score. For this task, mIoU is the primary metric. Pixel accuracy is included for completeness but carries little diagnostic value on its own."),

      H3("6.4 Box-mask gap in YOLOv8n-seg"),
      P("Box mAP50 measures whether the predicted bounding box overlaps the ground truth at IoU > 0.5. Mask mAP50 requires the predicted polygon to also match the object boundary at the same threshold. Boxes are forgiving — a box 10 pixels too wide still scores well. Polygon masks are not. Gloves, shoes, and masks in particular change shape constantly with hand and body position. The expected gap between box and mask mAP is larger for these deformable items than for rigid ones like helmets."),

      H3("6.5 The pseudo-label problem"),
      P("Every pixel label in this dataset comes from SAM2 running on a keremberke bounding box annotation — there are no human-drawn polygon masks anywhere. SAM2 is accurate for clear, isolated objects. It struggles when objects overlap (one glove in front of another) or when the bounding box contains partial occlusion. The class label for every pixel inside a SAM2 mask comes from the box annotation, so if a box is slightly misdrawn, every mislabelled pixel contributes to training noise. This is the primary ceiling on achievable mIoU with this data pipeline."),

      H3("6.6 What would actually improve these results"),
      bullet("Hand-annotated polygon masks for even 500 images would likely raise mIoU more than any architecture change. SAM2 pseudo-labels are the primary noise source."),
      bullet("The keremberke dataset has 11,704 images with relevant annotations; this assignment used only 4,000. Training on the full set (requiring ~5 hours of SAM2 processing) would directly help the lower-frequency classes like mask and no_mask."),
      bullet("Separate models per item type — one DeepLabV3+ for helmet/no_helmet, another for glove/no_glove — would let each model specialise without competing for capacity across very different object scales."),
      bullet("CRF post-processing on DeepLab outputs would sharpen predicted boundaries at no retraining cost, at the expense of added inference time per image."),
      bullet("A two-stage pipeline (person detector first, then item-level segmentation on the crop) would reduce background dominance and let the segmentation model focus on the human body region where PPE items actually appear."),

      // ── Appendix ─────────────────────────────────────────────────────────
      pageBreak(),
      H1("Appendix: Training Curves and Prediction Samples"),

      ...((() => {
        const items = [];
        const dl_train = loadImg("deeplab_training.png", 560, 215);
        if (dl_train) { items.push(dl_train); items.push(imgCaption("Figure 1. DeepLabV3+ training loss (left) and validation mIoU (right) over 30 epochs.")); }

        const dl_iou = loadImg("deeplab_confusion.png", 500, 220);
        if (dl_iou) { items.push(dl_iou); items.push(imgCaption("Figure 2. DeepLabV3+ per-class IoU on the validation set. Background dominates; helmet is the hardest PPE class.")); }

        const dl_grid = loadImg("deeplab_pred_grid.png", 540, 540);
        if (dl_grid) { items.push(dl_grid); items.push(imgCaption("Figure 3. DeepLabV3+ sample predictions (left column: input image; right column: predicted segmentation mask).")); }

        const ys_res = loadImg("yolo_seg_results_plot.png", 560, 220);
        if (ys_res) { items.push(ys_res); items.push(imgCaption("Figure 4. YOLOv8n-seg training metrics over 30 epochs including box and mask mAP.")); }

        const ys_conf = loadImg("yolo_seg_confusion.png", 440, 380);
        if (ys_conf) { items.push(ys_conf); items.push(imgCaption("Figure 5. YOLOv8n-seg confusion matrix on the validation set.")); }

        const exp_comp = loadImg("experiment_comparison.png", 560, 460);
        if (exp_comp) { items.push(exp_comp); items.push(imgCaption("Figure 6. Full 19-model comparison chart spanning classification, detection, semantic segmentation, and instance segmentation tasks.")); }

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
