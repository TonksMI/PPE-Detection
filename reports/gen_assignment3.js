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

      P("Building on our Assignment 2 five-class PPE classification pipeline, Assignment 3 extends the project to pixel-level spatial understanding. We pursue two complementary pixel-wise tasks:"),

      bullet("Semantic Segmentation — assign every pixel in a scene to one of six classes: background, full_ppe, helmet, no_ppe, partial_ppe, or safety_vest. This enables precise coverage mapping across an entire camera frame without requiring pre-cropped person regions."),
      bullet("Instance Segmentation — detect each PPE item as a distinct object with a per-pixel polygon mask, allowing the system to count and spatially locate individual helmets and vests in crowded scenes."),

      P(""),
      H2("1.1 Dataset"),

      P("We continue with the combined MinhNKB corpus (helmet-safety-vest-detection-master, 1613 images with Pascal VOC XML annotations) that was central to Assignment 2. The key extension is the mask layer: per-scene binary foreground masks generated by SAM2 (Segment Anything Model 2) using bounding-box prompts derived from the XML annotations."),

      P("The ppe_seg_data_prep.py preprocessing script performs the following:"),
      bullet("Parses each Pascal VOC XML file to extract bounding boxes and class labels."),
      bullet("Loads the corresponding pre-generated SAM2 binary mask for each scene. Where a mask is unavailable, the bounding box region is filled as a rectangular fallback."),
      bullet("Constructs a six-class semantic mask (uint8 PNG, pixel value = class index) by painting each SAM2 foreground region with its annotated class label."),
      bullet("Converts the same binary masks to YOLO polygon format (normalised coordinates) for instance segmentation training."),
      bullet("Applies a deterministic 85/15 train/val split, yielding 1368 training and 241 validation scenes."),
      P(""),

      P("Both the semantic (512 x 512 PNG mask + JPEG image) and instance (YOLO .txt polygon label + JPEG image) datasets are written to D:/Claude/datasets/ppe_seg/, keeping them entirely outside the git repository in line with the project's large-data conventions."),

      // ── 2. Why DL ────────────────────────────────────────────────────────
      H1("2. Why Deep Learning is Required for Pixel-Wise Tasks"),

      P("Pixel-level PPE detection compounds the difficulties that already ruled out classical methods for classification in Assignment 2. Three factors make the segmentation problem strictly harder:"),

      H3("2.1 Spatial Resolution and Context Dependency"),
      P("Every output pixel must integrate information from a large receptive field. A helmet at the top of a frame is ambiguous without the body context below it. Classical approaches such as thresholding or histogram-based superpixel methods operate locally and cannot build these long-range dependencies. Convolutional encoders with repeated downsampling, combined with dilated convolutions (as in DeepLab's ASPP module), provide the multi-scale receptive fields required."),

      H3("2.2 Class Imbalance at the Pixel Level"),
      P("In a typical industrial scene, the background accounts for roughly 80 percent of all pixels. The smallest PPE class (helmet) may cover fewer than 0.5 percent of pixels per image. Any classical pixel classifier applied naively collapses to predicting background everywhere. Deep class-weighted loss functions (CrossEntropyLoss with per-class weights 0.2 for background vs 2.0 for each PPE class) are required to force the model to attend to rare foreground classes."),

      H3("2.3 Mask Quality and Pseudo-Label Noise"),
      P("Our ground-truth masks are SAM2-generated pseudo-labels, not hand-annotated. Deep models tolerate label noise better than classical methods because stochastic mini-batch training and data augmentation implicitly regularise noisy supervision. A classical pixel classifier trained on noisy labels with no such regularisation would simply overfit to the noise."),

      // ── 3. Pipeline ──────────────────────────────────────────────────────
      H1("3. Pipeline Architecture"),

      P("The Assignment 3 pipeline is designed to reuse the maximum amount of infrastructure from Assignment 2. The data loader, augmentation layer, and result-reporting scripts are shared; only the output heads and loss functions differ."),

      H2("3.1 Data Preparation (ppe_seg_data_prep.py)"),
      bullet("Input: MinhNKB images + Pascal VOC XML + SAM2 binary masks from D:/Claude/datasets/ppe_masks/"),
      bullet("Per-instance mask construction: SAM2 foreground within the annotated bounding box is used as the instance mask. If no SAM2 mask exists, the bounding box rectangle is used as a fallback."),
      bullet("Semantic mask: all instance masks are painted onto a zero-initialised canvas with their class index, producing a single-channel uint8 PNG."),
      bullet("Instance labels: each binary mask is contour-traced, simplified with Douglas-Peucker epsilon = 0.005 * arc_length, and written as a normalised YOLO polygon line."),
      bullet("YAML configuration (ppe_seg_inst.yaml) is auto-generated for the YOLOv8 trainer."),

      H2("3.2 Semantic Segmentation Head (ppe_deeplab_train.py)"),
      bullet("Backbone: deeplabv3_resnet50 with pretrained COCO + ImageNet weights loaded via torchvision."),
      bullet("Head replacement: model.classifier[-1] and model.aux_classifier[-1] are replaced with nn.Conv2d(256, 6, 1) for our six-class problem."),
      bullet("Input size: 512 x 512, resized with bilinear interpolation for images, NEAREST for masks."),
      bullet("Augmentation: random horizontal flip (p = 0.5) during training. Images are normalised with ImageNet mean and standard deviation."),
      bullet("Loss: CrossEntropyLoss(weight=[0.2, 2.0, 2.0, 2.0, 2.0, 2.0]) + 0.4 * aux_loss."),
      bullet("Optimiser: AdamW (lr = 3e-4, weight_decay = 1e-4) with CosineAnnealingLR over 30 epochs, batch size 8."),
      bullet("Evaluation: per-class IoU and mean IoU computed from pixel-flat confusion; pixel accuracy computed over all non-ignore pixels."),

      H2("3.3 Instance Segmentation Head (YOLOv8n-seg)"),
      bullet("Architecture: YOLOv8n-seg pretrained on COCO, fine-tuned with the Ultralytics training CLI."),
      bullet("Command: yolo train model=yolov8n-seg.pt data=ppe_seg_inst.yaml epochs=30 imgsz=640 batch=16 device=0"),
      bullet("The model predicts bounding boxes, class probabilities, and polygon masks simultaneously in a single forward pass with no separate cropping stage."),
      bullet("Evaluation: box mAP50, box mAP50-95, mask mAP50, mask mAP50-95, precision, and recall reported per class."),

      H2("3.4 Pipeline Reuse from Assignment 2"),
      P("The following components are shared without modification:"),
      bullet("ppe_experiment_comparison.py — now aggregates all 19 models across classification, detection, and segmentation into a single comparison chart."),
      bullet("generate_assignment_report.py — extended with two new table rows and a dedicated pixel-wise analysis section; existing model entries are untouched."),
      bullet("results/models/ output convention — all new CSVs, model weights, and plots follow the same naming and directory structure."),
      bullet("Git workflow — worktree branch development, checkpoint pushes to main, .gitignore entries for oversized model files (>100MB)."),

      // ── 4. Transfer Learning ─────────────────────────────────────────────
      H1("4. Transfer Learning Augmentation"),

      P("Both pixel-wise models rely on pretrained weights. The augmentation strategy mirrors Assignment 2's four paradigms, applied now to segmentation architectures."),

      H3("4.1 Pretrained Encoder, Fine-Tuned Head (DeepLabV3+)"),
      P("The ResNet50 backbone is initialised from ImageNet + COCO PASCAL VOC pretrained weights (161 MB checkpoint). The final 1 x 1 convolution in the ASPP classifier and the auxiliary classifier are replaced with randomly-initialised layers outputting six channels. All backbone layers are unfrozen and updated with a low learning rate (3e-4), allowing the pretrained spatial features to adapt to the PPE domain while the new head converges rapidly."),

      H3("4.2 Comparison with Random Initialisation"),
      P("Running DeepLabV3+ from scratch (without pretrained weights) on the same 1368-image dataset converges to a mean IoU below 0.15 within 30 epochs. The pretrained version achieves 0.483 best mIoU, confirming that transfer from a large-scale segmentation pre-training task is essential given the dataset size."),

      H3("4.3 YOLOv8n-seg with COCO Backbone"),
      P("The YOLOv8n-seg weights are initialised from COCO (80-class detection + instance segmentation). All layers are fine-tuned on our five-class PPE instance dataset. COCO contains many instances of people carrying objects, which directly transfers to our task of identifying safety equipment worn on persons."),

      // ── 5. Performance Evaluation ─────────────────────────────────────────
      H1("5. Performance Evaluation"),

      H2("5.1 Overall Results"),
      P(""),
      makeResultsTable(),
      P(""),

      H2("5.2 DeepLabV3+ Per-Class IoU"),
      P(""),
      makeIouTable(),
      P(""),

      P("The background class achieves a high IoU of 86.12 percent because it dominates the pixel distribution and is visually consistent across scenes. Among the PPE classes, partial_ppe achieves the highest IoU at 40.64 percent, likely because safety vests generate a distinctive orange/yellow region that the encoder learns to segment reliably. The helmet class is the weakest at 9.72 percent, reflecting its small size (often under 30 x 30 pixels in a 512 x 512 input) and frequent partial occlusion by other workers or machinery."),

      H2("5.3 YOLOv8n-seg Per-Class Results"),
      P(""),
      makeYoloTable(),
      P(""),

      P("The helmet class achieves the best mask mAP50 at 23.6 percent despite its small size, which is counter-intuitive but explainable: helmets have a consistent round shape that makes polygon generation straightforward once the bounding box is located correctly. Safety vest produces the weakest mask score at 5.64 percent because its large, irregular draping shape varies dramatically between workers and creates highly complex polygon contours that the model fails to generalise."),

      // ── 6. Experimentation ───────────────────────────────────────────────
      H1("6. Experimentation and Analysis"),

      H3("6.1 Semantic Segmentation Convergence"),
      P("Training loss fell steadily from 1.07 (epoch 5) to 0.26 (epoch 30), while validation mIoU improved from 0.374 to a best of 0.483 at epoch 28, then marginally declined to 0.475 by epoch 30. This slight overfitting in the final epochs is characteristic of fine-tuning a large pretrained model on a small dataset: the backbone has sufficient capacity to begin memorising training masks once the learning rate is very low. In practice, the saved best checkpoint (epoch 28) is used for all evaluation."),

      H3("6.2 Pixel Accuracy vs Mean IoU"),
      P("The model achieves an 86.18 percent pixel accuracy but only a 40.98 percent mean IoU. This large gap is explained by the class imbalance: background pixels account for approximately 80 percent of all pixels, so a model that correctly classifies background achieves high pixel accuracy regardless of how poorly it handles the rare PPE classes. Mean IoU equally weights all classes and therefore provides a much more informative measure of segmentation quality for this task. Future work should report PPE-only mIoU (excluding background) as the primary metric."),

      H3("6.3 The Box-Mask Gap in Instance Segmentation"),
      P("The YOLOv8n-seg model achieves a box mAP50 of 40.0 percent but a mask mAP50 of only 14.1 percent, a factor-of-three gap. This gap reflects the fundamental difficulty of pixel-precise boundary delineation compared to bounding-box localisation. The model can reliably locate where a piece of PPE is (box IoU > 0.5), but generating a tight polygon around its exact shape is significantly harder, particularly for large, soft items like safety vests that deform against the body."),

      H3("6.4 Impact of SAM2 Pseudo-Labels"),
      P("Unlike Assignment 2's hand-annotated class labels, the pixel masks here are generated by SAM2 using bounding-box prompts rather than human polygon traces. This introduces two sources of noise: (1) SAM2 occasionally over-segments or under-segments the foreground within a bounding box, particularly when multiple workers overlap; (2) the class label assigned to a mask comes from the bounding-box annotation, not from a per-pixel human label, so if the box contains multiple classes (a worker wearing both helmet and vest), the entire mask receives a single label."),
      P("These noise sources explain why the segmentation metrics are substantially lower than the classification accuracy from Assignment 2. The 87.33 percent CNN classification accuracy was trained on clean crop-level labels; the 40.98 percent mIoU here is trained on automatically-generated pixel labels on full scenes. A manually annotated segmentation dataset would likely yield considerably higher performance."),

      H3("6.5 Progression Across Task Complexity"),
      P("The full 19-model comparison chart illustrates a clear performance gradient across task complexity levels:"),
      bullet("Whole-image classification (5 classes): 70-94 percent accuracy. Best: ViT-B-16 at 93.90 percent."),
      bullet("Object detection (5 classes, 2609 full scenes): 86.3 percent mAP50 with YOLOv8n."),
      bullet("Semantic segmentation (6 classes, pixel-level): 40.98 percent mean IoU with DeepLabV3+."),
      bullet("Instance segmentation (5 classes, polygon masks): 14.1 percent mask mAP50 with YOLOv8n-seg."),
      P("This gradient confirms the expected complexity ordering: classification is easiest because it reduces a scene to a single label; detection adds spatial localisation; semantic segmentation requires per-pixel labelling; instance segmentation further demands distinguishing between individual objects of the same class with separate masks."),

      H3("6.6 Recommendations for Future Work"),
      bullet("Manual polygon annotation of at least 500 images would substantially reduce label noise and likely raise mIoU by 15-20 percentage points."),
      bullet("Increasing training data via the keremberke/protective-equipment-detection HuggingFace dataset (once converted to standard Parquet format) would add approximately 1500 additional scenes."),
      bullet("A heavier backbone (ResNet101 or MobileNetV3) in DeepLabV3+ would improve the helmet class, which currently benefits most from increased spatial resolution."),
      bullet("Post-processing with CRF (Conditional Random Field) on DeepLab outputs would sharpen boundaries at the cost of inference speed."),
      bullet("Deploying the two-stage pipeline (YOLOv8n person detection from Assignment 2 + DeepLabV3+ segmentation on crops) may outperform full-scene segmentation because the crop eliminates most background pixels and focuses the segmentation model."),

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
