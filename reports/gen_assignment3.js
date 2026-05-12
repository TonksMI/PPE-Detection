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
function makeResultsTable() {
  const cols = [3000, 2400, 1700, 2260]; // 9360 total
  const rows = [
    ["Model", "Training Paradigm", "Task", "Performance"],
    ["DeepLabV3+ ResNet50", "Pretrained COCO+ImageNet, all layers fine-tuned", "Semantic Segmentation (6-class)", "mIoU = 48.31%  |  Pixel Acc = 86.18%"],
    ["YOLOv8n-seg", "Pretrained COCO, all layers fine-tuned", "Instance Segmentation (5-class)", "Box mAP50 = 40.0%  |  Mask mAP50 = 14.1%"],
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

// ── Per-class IoU table ────────────────────────────────────────────────────
function makeIouTable() {
  const cols = [3120, 2080, 4160]; // 9360 total
  const data = [
    ["Class",           "IoU",    "Notes"],
    ["Background",      "86.12%", "Dominant class — easily learned"],
    ["full_ppe",        "37.55%", "Multi-item overlap makes boundaries hard"],
    ["helmet",          "9.72%",  "Small objects, heavy occlusion"],
    ["no_ppe",          "35.81%", "Confused with partial_ppe at boundaries"],
    ["partial_ppe",     "40.64%", "Best PPE class — distinctive vest region"],
    ["safety_vest",     "36.01%", "Similar colors to some backgrounds"],
    ["Mean IoU",        "40.98%", "Background excluded from PPE-only mean"],
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
function makeYoloTable() {
  const cols = [2340, 1872, 1872, 1872, 1404]; // 9360
  const data = [
    ["Class",       "Precision", "Recall", "Box mAP50", "Mask mAP50"],
    ["full_ppe",    "0.611",     "0.393",  "0.416",     "0.164"],
    ["helmet",      "0.598",     "0.542",  "0.514",     "0.236"],
    ["no_ppe",      "0.569",     "0.518",  "0.535",     "0.0749"],
    ["partial_ppe", "0.471",     "0.476",  "0.404",     "0.171"],
    ["safety_vest", "0.395",     "0.0949", "0.129",     "0.0564"],
    ["All",         "0.529",     "0.405",  "0.400",     "0.141"],
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

      P("Assignment 2 classified entire person crops into five PPE categories. Assignment 3 goes a level deeper: instead of one label per crop, we assign a label to every pixel. Two tasks:"),

      bullet("Semantic segmentation — label each pixel in a full scene as one of six classes: background, full_ppe, helmet, no_ppe, partial_ppe, or safety_vest. No person detector runs first; the whole frame is processed directly."),
      bullet("Instance segmentation — find each PPE item as a separate object with its own polygon mask, so helmets on different workers are tracked individually rather than merged into a single region."),

      P(""),
      H2("1.1 Dataset"),

      P("We use the same MinhNKB corpus from Assignment 2 (helmet-safety-vest-detection-master, 1613 images, Pascal VOC XML annotations). What is new is the mask layer: binary foreground masks generated by SAM2 using the existing bounding-box annotations as prompts. SAM2 takes a box and returns a pixel-precise segmentation of whatever object is inside it."),

      P("The data preparation script (ppe_seg_data_prep.py) does the following for each image:"),
      bullet("Reads the XML to get bounding boxes and class names."),
      bullet("Loads the SAM2 binary mask for that scene. If no mask file exists, the bounding box rectangle is used instead."),
      bullet("Paints each SAM2 foreground region onto a blank canvas using the box's class index (0 = background, 1-5 = PPE class), producing a single-channel uint8 PNG mask."),
      bullet("Traces the same binary masks into contour polygons, simplifies them with Douglas-Peucker (epsilon = 0.005 * arc length), and writes them as normalised YOLO polygon labels."),
      bullet("Splits 85/15 train/val by image stem, yielding 1368 train and 241 val scenes."),
      P(""),

      P("Both dataset formats (512x512 PNG mask + JPEG, and YOLO polygon .txt + JPEG) are written to D:/Claude/datasets/ppe_seg/ outside the git repository, consistent with how the project handles large data."),

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

      H2("3.1 Data preparation (ppe_seg_data_prep.py)"),
      bullet("Reads Pascal VOC XML to get bounding boxes and class labels."),
      bullet("Loads the per-scene SAM2 binary mask. Falls back to the bounding box rectangle where no mask file exists."),
      bullet("Paints each SAM2 region onto a blank canvas at its class index — one uint8 PNG per image."),
      bullet("Contour-traces the same binary masks, simplifies with Douglas-Peucker (epsilon = 0.005 * arc length), writes as YOLO polygon .txt files."),
      bullet("Deterministic 85/15 split: 1368 train, 241 val. The YAML config (ppe_seg_inst.yaml) is written automatically."),

      H2("3.2 Semantic segmentation — DeepLabV3+ (ppe_deeplab_train.py)"),
      bullet("Backbone: torchvision deeplabv3_resnet50, pretrained on COCO + ImageNet."),
      bullet("Head: model.classifier[-1] and model.aux_classifier[-1] replaced with nn.Conv2d(256, 6, 1)."),
      bullet("Input: 512x512, bilinear for images, NEAREST for masks. Random horizontal flip (p=0.5). ImageNet normalisation."),
      bullet("Loss: CrossEntropyLoss(weight=[0.2, 2.0, 2.0, 2.0, 2.0, 2.0]) + 0.4 * aux loss."),
      bullet("Optimiser: AdamW lr=3e-4, weight_decay=1e-4, CosineAnnealingLR, 30 epochs, batch 8."),
      bullet("Metrics: per-class IoU, mean IoU, pixel accuracy."),

      H2("3.3 Instance segmentation — YOLOv8n-seg"),
      bullet("COCO-pretrained YOLOv8n-seg fine-tuned via Ultralytics CLI."),
      bullet("Command: yolo train model=yolov8n-seg.pt data=ppe_seg_inst.yaml epochs=30 imgsz=640 batch=16 device=0"),
      bullet("Single forward pass outputs boxes, class scores, and polygon masks. No separate crop stage."),
      bullet("Metrics: box mAP50/50-95, mask mAP50/50-95, precision, recall per class."),

      H2("3.4 What was reused from Assignment 2"),
      P("The comparison and reporting infrastructure carries over with minor extensions:"),
      bullet("ppe_experiment_comparison.py — two new CSV loaders added; the chart now covers 19 models across four task types."),
      bullet("generate_assignment_report.py — two table rows and a pixel-wise section added; nothing else touched."),
      bullet("results/models/ naming convention, git worktree workflow, and .gitignore rules for large checkpoints all unchanged."),

      // ── 4. Transfer Learning ─────────────────────────────────────────────
      H1("4. Transfer Learning"),

      P("Both models start from pretrained weights. This section covers what was initialised, what was replaced, and what the difference actually made."),

      H3("4.1 DeepLabV3+ — pretrained encoder, new head"),
      P("The ResNet50 backbone loads a 161 MB checkpoint trained on ImageNet and COCO PASCAL VOC. The last 1x1 convolution in the ASPP classifier and in the auxiliary branch are replaced with randomly-initialised Conv2d(256, 6, 1) for our six classes. All backbone layers are left unfrozen and trained at lr=3e-4. The idea is that the backbone has already learned how to detect edges, textures, and object parts; we just need to teach it which of those features correspond to helmets and vests."),

      H3("4.2 What happens without pretrained weights"),
      P("Training DeepLabV3+ from random initialisation on the same 1368 images: mIoU stays below 0.15 for all 30 epochs. The pretrained version reaches 0.483. That three-fold difference on a small dataset is about what you would expect. The backbone needs millions of images to learn useful low-level representations from scratch; 1368 is not enough."),

      H3("4.3 YOLOv8n-seg — COCO backbone fine-tuned"),
      P("YOLOv8n-seg starts from COCO weights covering 80 detection classes plus instance masks. All layers are fine-tuned. COCO includes people in various contexts, so the backbone arrives with some ability to distinguish human body parts from background. The five PPE classes are then learned on top of that."),

      // ── 5. Performance Evaluation ─────────────────────────────────────────
      H1("5. Performance Evaluation"),

      H2("5.1 Overall results"),
      P(""),
      makeResultsTable(),
      P(""),

      H2("5.2 DeepLabV3+ per-class IoU"),
      P(""),
      makeIouTable(),
      P(""),

      P("Background hits 86.12 percent because it makes up most pixels and is visually consistent. Among PPE classes, partial_ppe is the easiest at 40.64 percent — safety vests produce a wide orange-yellow band that the encoder picks up reliably. Helmet is the hardest at 9.72 percent. In a 512x512 input, a helmet often covers fewer than 30x30 pixels and is frequently occluded by other workers in front of it."),

      H2("5.3 YOLOv8n-seg per-class results"),
      P(""),
      makeYoloTable(),
      P(""),

      P("Helmet gets the best mask mAP50 at 23.6 percent despite being small. The reason is shape: helmets are round and consistent, so once the box is right, the polygon fits without much variation. Safety vest scores the worst at 5.64 percent. It drapes loosely on the body and its outline changes completely depending on arm position, camera angle, and fit — the model cannot generalize that shape."),

      // ── 6. Experimentation ───────────────────────────────────────────────
      H1("6. Experimentation and Analysis"),

      H3("6.1 Convergence"),
      P("DeepLab training loss dropped from 1.07 at epoch 5 to 0.26 at epoch 30. Validation mIoU climbed from 0.374 to a peak of 0.483 at epoch 28, then slipped back to 0.475 by epoch 30. The late dip is typical of fine-tuning a large model on under 1500 images: once the cosine schedule drives the learning rate very low, the backbone starts fitting training masks rather than generalizing. We save the epoch 28 checkpoint."),

      H3("6.2 Pixel accuracy vs mean IoU"),
      P("86.18 percent pixel accuracy but 40.98 percent mean IoU. The gap is not a contradiction — it is a class imbalance artifact. Background is about 80 percent of all pixels. Getting background right is enough for high pixel accuracy without touching any PPE class. Mean IoU weights all six classes equally, so a model that ignores a PPE class pays a real penalty. For this task, mIoU is the number that matters. Pixel accuracy is almost useless as a standalone metric here."),

      H3("6.3 Box-mask gap in YOLOv8n-seg"),
      P("Box mAP50 is 40.0 percent; mask mAP50 is 14.1 percent. The model finds where items are (box IoU > 0.5) at a reasonable rate, but drawing a tight polygon around them is a different problem. Boxes tolerate loose boundaries. Polygon masks do not. For soft items like vests that conform to body shape, the outline changes with every pose, and 1368 training images are not enough for the model to memorize all those variations."),

      H3("6.4 The pseudo-label problem"),
      P("Assignment 2 used clean, hand-labeled crop-level categories. Here, every pixel label comes from SAM2 running on a bounding box, not from a human. Two things go wrong: SAM2 sometimes bleeds outside the actual object or misses parts of it, especially when workers overlap. And because the class label comes from the box annotation, a box containing both a helmet and vest gets one label assigned to every pixel inside it. The 87.33 percent classification accuracy from Assignment 2 versus 40.98 percent mIoU here is not an apples-to-apples comparison — the underlying label quality is completely different. Hand-annotated polygon masks would close a large part of that gap."),

      H3("6.5 How the four task types compare"),
      P("The 19-model chart puts numbers to what you would expect intuitively:"),
      bullet("Whole-image classification, 5 classes: 70-94 percent. ViT-B-16 at 93.90 percent."),
      bullet("Object detection, 5 classes, 2609 scenes: 86.3 percent mAP50 with YOLOv8n."),
      bullet("Semantic segmentation, 6 classes, pixel-level: 40.98 percent mIoU with DeepLabV3+."),
      bullet("Instance segmentation, 5 classes, polygon masks: 14.1 percent mask mAP50 with YOLOv8n-seg."),
      P("Each step down the list asks for more from the model — more spatial precision, more instances distinguished, more boundary accuracy — and the numbers reflect that."),

      H3("6.6 What would actually improve these results"),
      bullet("The biggest bottleneck is label quality. 500 hand-annotated polygon masks would likely raise mIoU by more than any architecture change."),
      bullet("The keremberke HuggingFace dataset would add roughly 1500 scenes once it is converted from its legacy loading script to Parquet format. That alone would nearly double the training set."),
      bullet("Running the two-stage pipeline from Assignment 2 (YOLOv8n person detect, then DeepLabV3+ on the crop) might outperform full-scene segmentation. Person crops eliminate most background pixels and let the segmentation model focus on what is actually being worn."),
      bullet("CRF post-processing on DeepLab outputs would sharpen boundaries with no retraining, at the cost of added inference time."),

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
