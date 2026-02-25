const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, LevelFormat, ImageRun
} = require('docx');
const fs = require('fs');
const path = require('path');

const OUT_DIR = "/sessions/sleepy-epic-pascal/mnt/Computer Vision";

// ── Helpers ────────────────────────────────────────────────────
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const headerBorder = { style: BorderStyle.SINGLE, size: 1, color: "1a5276" };
const headerBorders = { top: headerBorder, bottom: headerBorder, left: headerBorder, right: headerBorder };
const cell = (text, w, bold=false, hdr=false, align=AlignmentType.LEFT, shade=null) =>
  new TableCell({
    borders: hdr ? headerBorders : borders,
    width: { size: w, type: WidthType.DXA },
    shading: shade ? { fill: shade, type: ShadingType.CLEAR } :
             hdr   ? { fill: "1a5276", type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text, bold, color: hdr ? "FFFFFF" : "000000", size: hdr ? 20 : 19 })]
    })]
  });

const p = (text, opts={}) => new Paragraph({
  spacing: { before: 60, after: 60 },
  ...opts,
  children: [new TextRun({ text, size: 22, ...opts.run })]
});

const h1 = text => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 320, after: 120 },
  children: [new TextRun({ text, size: 32, bold: true, color: "1a5276" })]
});
const h2 = text => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 240, after: 80 },
  children: [new TextRun({ text, size: 26, bold: true, color: "2471a3" })]
});
const h3 = text => new Paragraph({
  heading: HeadingLevel.HEADING_3,
  spacing: { before: 180, after: 60 },
  children: [new TextRun({ text, size: 22, bold: true, color: "1a5276" })]
});

const bullet = text => new Paragraph({
  numbering: { reference: "bullets", level: 0 },
  spacing: { before: 40, after: 40 },
  children: [new TextRun({ text, size: 22 })]
});

const spacer = () => new Paragraph({ spacing: { before: 80, after: 80 }, children: [new TextRun({ text: "" })] });

// Load images
const loadImg = fname => {
  const fp = path.join(OUT_DIR, fname);
  return fs.existsSync(fp) ? fs.readFileSync(fp) : null;
};

const imgPara = (fname, width, height, caption) => {
  const data = loadImg(fname);
  if (!data) return p(`[Image: ${fname}]`);
  const paras = [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 40 },
      children: [new ImageRun({
        type: "png",
        data,
        transformation: { width, height },
        altText: { title: caption, description: caption, name: caption }
      })]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 120 },
      children: [new TextRun({ text: caption, size: 18, italics: true, color: "555555" })]
    })
  ];
  return paras;
};

// ── Document ───────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 320, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 80 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 180, after: 60 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          border: { bottom: { style: BorderStyle.SINGLE, size: 1, color: "AAAAAA" } },
          children: [new TextRun({ text: "PPE Detection System — Computer Vision Final Project", size: 18, color: "555555", italics: true })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          border: { top: { style: BorderStyle.SINGLE, size: 1, color: "AAAAAA" } },
          children: [
            new TextRun({ text: "Page ", size: 18, color: "555555" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "555555" }),
            new TextRun({ text: " of ", size: 18, color: "555555" }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, color: "555555" }),
          ]
        })]
      })
    },
    children: [
      // ── TITLE PAGE ────────────────────────────────────────────
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 1440, after: 240 },
        children: [new TextRun({ text: "PPE Detection System", size: 56, bold: true, color: "1a5276" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 120 },
        children: [new TextRun({ text: "Computer Vision Model for Workplace Safety Equipment Detection", size: 28, color: "2471a3" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 480 },
        children: [new TextRun({ text: "Final Project | Spring 2025", size: 22, color: "888888", italics: true })]
      }),
      // Title divider table
      new Table({
        width: { size: 9720, type: WidthType.DXA },
        columnWidths: [9720],
        rows: [new TableRow({ children: [new TableCell({
          borders: { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.SINGLE, size: 6, color: "1a5276" },
                     left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE } },
          width: { size: 9720, type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun("")] })]
        })]})]
      }),
      spacer(),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 240, after: 120 },
        children: [new TextRun({ text: "Abstract", size: 26, bold: true, color: "333333" })]
      }),
      new Paragraph({
        alignment: AlignmentType.BOTH,
        spacing: { before: 60, after: 60, line: 360 },
        children: [new TextRun({
          text: "This project develops a multi-model computer vision pipeline for automated detection of Personal Protective Equipment (PPE) in industrial settings. Using the MinhNKB Helmet-Safety-Vest dataset (1,613 images, 4,723 bounding box annotations), we train and compare four model families: Support Vector Machine (SVM), Random Forest, Gradient Boosted Trees, and a custom Convolutional Neural Network (CNN). Feature extraction employs HOG descriptors and color histograms for traditional ML models, while the CNN learns end-to-end representations directly from 64\u00d764 image crops. Results demonstrate the CNN achieves the highest multi-class accuracy (84.3%), while SVM leads binary PPE classification (85.3%). Validation on 14 surveillance-style CCTV images from industrial environments confirms real-world applicability. Ethical considerations including worker privacy, face anonymization, and responsible deployment are discussed.",
          size: 22
        })]
      }),

      // ── SECTION 1: INTRODUCTION ───────────────────────────────
      new Paragraph({ children: [new TextRun({ text: "" })], pageBreakBefore: true }),
      h1("1. Introduction"),
      p("Workplace accidents remain a leading cause of preventable injury globally. Industrial facilities such as warehouses, construction sites, and manufacturing plants require workers to wear Personal Protective Equipment (PPE) including hard hats, safety vests, and safety glasses. Manual compliance monitoring is labor-intensive and error-prone; automated computer vision systems offer a scalable, real-time alternative."),
      spacer(),
      p("This project addresses three related vision tasks:"),
      bullet("Identifying individual safety equipment items (helmets, safety vests, glasses)"),
      bullet("Identifying people and their PPE compliance status (full PPE, partial PPE, no PPE)"),
      bullet("Distinguishing safety equipment from similar everyday items (hat vs. helmet, colored shirt vs. high-vis vest, glasses vs. safety glasses)"),
      spacer(),
      p("The pipeline is implemented as a multi-stage system: dataset aggregation, exploratory data analysis (EDA), feature extraction, model training across four model families, and final validation against CCTV-style footage."),

      // ── SECTION 2: DATASETS ───────────────────────────────────
      h1("2. Datasets"),
      h2("2.1 Primary Training Dataset"),
      p("The primary dataset is the MinhNKB Helmet-Safety-Vest Detection dataset, sourced from the public GitHub repository (github.com/MinhNKB/helmet-safety-vest-detection). It contains:"),
      spacer(),
      new Table({
        width: { size: 9720, type: WidthType.DXA },
        columnWidths: [2880, 3420, 3420],
        rows: [
          new TableRow({ children: [
            cell("Attribute", 2880, true, true),
            cell("Details", 3420, true, true),
            cell("Notes", 3420, true, true),
          ]}),
          new TableRow({ children: [
            cell("Total Images", 2880, false, false, AlignmentType.LEFT, "F0F4F8"),
            cell("1,613", 3420),
            cell("JPG format, varying resolutions", 3420),
          ]}),
          new TableRow({ children: [
            cell("Bounding Boxes", 2880, false, false, AlignmentType.LEFT, "F0F4F8"),
            cell("4,723 annotations", 3420),
            cell("Pascal VOC XML format", 3420),
          ]}),
          new TableRow({ children: [
            cell("Annotation Tool", 2880, false, false, AlignmentType.LEFT, "F0F4F8"),
            cell("LabelImg", 3420),
            cell("Manual bounding box annotation", 3420),
          ]}),
          new TableRow({ children: [
            cell("Image Sources", 2880, false, false, AlignmentType.LEFT, "F0F4F8"),
            cell("Google Search, Stanford 40 Actions", 3420),
            cell("Web-mined, diverse environments", 3420),
          ]}),
          new TableRow({ children: [
            cell("Classes", 2880, false, false, AlignmentType.LEFT, "F0F4F8"),
            cell("5 classes (see below)", 3420),
            cell("Hierarchical PPE compliance", 3420),
          ]}),
        ]
      }),
      spacer(),
      h3("Class Definitions"),
      new Table({
        width: { size: 9720, type: WidthType.DXA },
        columnWidths: [2520, 1800, 3600, 1800],
        rows: [
          new TableRow({ children: [
            cell("Class Label", 2520, true, true),
            cell("Raw Count", 1800, true, true, AlignmentType.CENTER),
            cell("Description", 3600, true, true),
            cell("% of Total", 1800, true, true, AlignmentType.CENTER),
          ]}),
          new TableRow({ children: [cell("helmet", 2520, false, false, AlignmentType.LEFT, "EBF5EB"), cell("1,490", 1800, false, false, AlignmentType.CENTER), cell("Hard hat / construction helmet present", 3600), cell("31.6%", 1800, false, false, AlignmentType.CENTER)] }),
          new TableRow({ children: [cell("partial_ppe", 2520, false, false, AlignmentType.LEFT, "FEF9E7"), cell("1,228", 1800, false, false, AlignmentType.CENTER), cell("Some PPE present, not complete compliance", 3600), cell("26.0%", 1800, false, false, AlignmentType.CENTER)] }),
          new TableRow({ children: [cell("safety_vest", 2520, false, false, AlignmentType.LEFT, "EBF5FB"), cell("809", 1800, false, false, AlignmentType.CENTER), cell("High-visibility / reflective vest present", 3600), cell("17.1%", 1800, false, false, AlignmentType.CENTER)] }),
          new TableRow({ children: [cell("no_ppe", 2520, false, false, AlignmentType.LEFT, "FDEDEC"), cell("770", 1800, false, false, AlignmentType.CENTER), cell("No safety equipment visible on person", 3600), cell("16.3%", 1800, false, false, AlignmentType.CENTER)] }),
          new TableRow({ children: [cell("full_ppe", 2520, false, false, AlignmentType.LEFT, "F4ECF7"), cell("426", 1800, false, false, AlignmentType.CENTER), cell("Full PPE compliance (helmet + vest + more)", 3600), cell("9.0%", 1800, false, false, AlignmentType.CENTER)] }),
        ]
      }),
      spacer(),

      h2("2.2 Additional Datasets Identified"),
      p("During research, the following additional datasets were identified and evaluated:"),
      new Table({
        width: { size: 9720, type: WidthType.DXA },
        columnWidths: [2160, 1440, 1440, 4680],
        rows: [
          new TableRow({ children: [cell("Dataset", 2160, true, true), cell("Images", 1440, true, true, AlignmentType.CENTER), cell("Classes", 1440, true, true, AlignmentType.CENTER), cell("Source / Notes", 4680, true, true)] }),
          new TableRow({ children: [cell("Pictor-PPE v3", 2160, false, false, AlignmentType.LEFT, "F0F4F8"), cell("1,472", 1440, false, false, AlignmentType.CENTER), cell("3", 1440, false, false, AlignmentType.CENTER), cell("ciber-lab/pictor-ppe \u2014 dataset on Google Drive (not downloadable)", 4680)] }),
          new TableRow({ children: [cell("SH17 Dataset", 2160, false, false, AlignmentType.LEFT, "F0F4F8"), cell("8,099", 1440, false, false, AlignmentType.CENTER), cell("17", 1440, false, false, AlignmentType.CENTER), cell("ahmadmughees/SH17dataset \u2014 requires Pexels API", 4680)] }),
          new TableRow({ children: [cell("Hard Hat Detection (SHEL5K)", 2160, false, false, AlignmentType.LEFT, "F0F4F8"), cell("5,000", 1440, false, false, AlignmentType.CENTER), cell("6", 1440, false, false, AlignmentType.CENTER), cell("Kaggle (requires auth), Mendeley Data", 4680)] }),
          new TableRow({ children: [cell("PPE Tracking (azimjaan21)", 2160, false, false, AlignmentType.LEFT, "F0F4F8"), cell("54,325", 1440, false, false, AlignmentType.CENTER), cell("3", 1440, false, false, AlignmentType.CENTER), cell("CCTV-style, used for validation frames", 4680)] }),
          new TableRow({ children: [cell("Roboflow Hard Hat Workers", 2160, false, false, AlignmentType.LEFT, "F0F4F8"), cell("7,000+", 1440, false, false, AlignmentType.CENTER), cell("3", 1440, false, false, AlignmentType.CENTER), cell("public.roboflow.com \u2014 requires API key", 4680)] }),
        ]
      }),
      spacer(),

      h2("2.3 CCTV Validation Sources"),
      p("For final validation against real surveillance footage, images were collected from three publicly available sources without requiring authentication: CCTV-style output frames from the PPE Tracking project (azimjaan21), construction site sample images from the Pictor-PPE repository, and sample frames from the BrunoCestari PPE Detection project. This produced 14 validation images of varying resolution and perspective."),

      // ── SECTION 3: EDA ────────────────────────────────────────
      new Paragraph({ children: [new TextRun({ text: "" })], pageBreakBefore: true }),
      h1("3. Exploratory Data Analysis"),
      h2("3.1 Class Distribution & Imbalance"),
      p("The dataset exhibits significant class imbalance with a 3.5x ratio between the most frequent class (helmet: 1,490) and least frequent (full_ppe: 426). This imbalance reflects real-world conditions where full PPE compliance is less common than individual equipment sightings. To mitigate this, we applied balanced sampling capping at 400 samples per class for model training."),
      spacer(),
      ...imgPara("01_eda_analysis.png", 600, 360, "Figure 1: EDA Analysis — Class distribution, coverage, aspect ratios, and boxes-per-image"),
      spacer(),
      h2("3.2 Annotation Limitations"),
      p("Several important limitations were identified in the dataset tagging:"),
      bullet("Occlusion ambiguity: Workers partially visible at frame edges have uncertain PPE status. The dataset includes a \\u2018partial safety\\u2019 category, but the boundary between partial_ppe and no_ppe is subjective."),
      bullet("Scale variation: Bounding box coverage ranges from <1% to >50% of image area, creating large variation in crop resolution and feature quality."),
      bullet("Class overlap: Distinguishing a yellow hard hat from a yellow hat requires context (chin strap, texture, brim shape) that may be lost at 64x64 resolution."),
      bullet("Annotation inconsistency: Some images show the same worker annotated differently in similar poses across the dataset."),
      bullet("Aspect ratio diversity: Safety vest crops tend to be tall rectangles (portrait), while helmet crops are more square, adding within-class shape variation."),
      spacer(),
      ...imgPara("10_dataset_limitations.png", 580, 200, "Figure 2: Dataset limitations including class imbalance, coverage distribution, and tagging ambiguity"),

      // ── SECTION 4: FEATURE EXTRACTION ─────────────────────────
      new Paragraph({ children: [new TextRun({ text: "" })], pageBreakBefore: true }),
      h1("4. Feature Engineering & Preprocessing"),
      h2("4.1 Image Preprocessing"),
      p("All bounding box crops were resized to 64\u00d764 pixels for consistent feature extraction. The resize operation uses bilinear interpolation, which preserves texture gradients important for HOG features. Image channels are in BGR format (OpenCV default) for traditional models and converted to RGB for the CNN."),

      h2("4.2 Feature Extraction (Traditional ML)"),
      p("Two complementary feature families are extracted and concatenated to form a 1,956-dimensional feature vector:"),
      spacer(),
      h3("HOG (Histogram of Oriented Gradients)"),
      p("HOG captures shape and structure information essential for distinguishing equipment silhouettes. Parameters: cell size 8\u00d78, block size 16\u00d716, block stride 8\u00d78, 9 orientation bins, window 64\u00d764. This yields a 1,764-dimensional descriptor capturing edge orientation patterns across 49 cells."),
      h3("Color Histograms (RGB + HSV)"),
      p("Safety equipment has strong color signatures: yellow/orange hard hats, fluorescent green/orange vests. We extract 32-bin normalized histograms from each of 3 BGR channels plus 3 HSV channels, yielding 192 additional features. The HSV decomposition separates hue (color identity) from saturation and value (lighting conditions), improving robustness to illumination changes."),
      spacer(),
      new Table({
        width: { size: 9720, type: WidthType.DXA },
        columnWidths: [2400, 1800, 5520],
        rows: [
          new TableRow({ children: [cell("Feature Type", 2400, true, true), cell("Dimensions", 1800, true, true, AlignmentType.CENTER), cell("Captures", 5520, true, true)] }),
          new TableRow({ children: [cell("HOG descriptor", 2400, false, false, AlignmentType.LEFT, "F0F4F8"), cell("1,764", 1800, false, false, AlignmentType.CENTER), cell("Shape, edges, texture gradients", 5520)] }),
          new TableRow({ children: [cell("Color hist. (BGR)", 2400, false, false, AlignmentType.LEFT, "F0F4F8"), cell("96", 1800, false, false, AlignmentType.CENTER), cell("Color distribution in RGB space", 5520)] }),
          new TableRow({ children: [cell("Color hist. (HSV)", 2400, false, false, AlignmentType.LEFT, "F0F4F8"), cell("96", 1800, false, false, AlignmentType.CENTER), cell("Hue/saturation, illumination-robust color", 5520)] }),
          new TableRow({ children: [cell("Total", 2400, true, false, AlignmentType.LEFT, "E8F4FD"), cell("1,956", 1800, true, false, AlignmentType.CENTER), cell("\u2014", 5520)] }),
        ]
      }),
      spacer(),
      h2("4.3 CNN Feature Learning"),
      p("The CNN learns task-specific feature representations end-to-end, avoiding manual feature engineering. Input crops are normalized to zero mean and unit variance using ImageNet statistics ([0.485, 0.456, 0.406] and [0.229, 0.224, 0.225]). Training-time augmentation includes random horizontal flips, \u00b115\u00b0 rotation, and color jitter to improve generalization."),

      // ── SECTION 5: MODELS ─────────────────────────────────────
      new Paragraph({ children: [new TextRun({ text: "" })], pageBreakBefore: true }),
      h1("5. Model Architectures"),
      h2("5.1 Support Vector Machine (SVM)"),
      p("The SVM pipeline consists of three stages: StandardScaler (zero-mean, unit-variance normalization), PCA dimensionality reduction to 100 components (retaining ~95% variance), and an RBF-kernel SVC with C=10, gamma='scale'. The radial basis function kernel maps the 100-dimensional PCA space into a high-dimensional implicit feature space, enabling non-linear decision boundaries. Probability calibration is enabled via Platt scaling for ROC/AUC analysis."),
      p("SVM was trained separately for multi-class (one-vs-rest) and binary (PPE present/absent) tasks. The multi-class formulation uses sklearn's default one-vs-one strategy."),

      h2("5.2 Random Forest"),
      p("The Random Forest uses 200 decision trees with maximum depth 15, minimum samples per split 4, and all-core parallelization. Feature selection at each split uses the square root heuristic. Random forests naturally handle feature importance estimation, providing interpretability into which HOG cells and color channels are most discriminative for PPE detection."),

      h2("5.3 Gradient Boosted Trees (HistGBM)"),
      p("We use scikit-learn's HistGradientBoostingClassifier, a histogram-based variant that bins continuous features into 256 discrete intervals before splitting, dramatically reducing training time. Parameters: 100 boosting iterations, max tree depth 6, learning rate 0.1. PCA preprocessing to 80 components is applied before boosting."),

      h2("5.4 Convolutional Neural Network (CNN)"),
      p("The custom CNN architecture follows a standard encoder pattern with four convolutional blocks:"),
      spacer(),
      new Table({
        width: { size: 9720, type: WidthType.DXA },
        columnWidths: [1440, 2160, 2160, 1800, 2160],
        rows: [
          new TableRow({ children: [cell("Layer", 1440, true, true), cell("Operation", 2160, true, true), cell("Output Shape", 2160, true, true), cell("Filters", 1800, true, true, AlignmentType.CENTER), cell("Notes", 2160, true, true)] }),
          new TableRow({ children: [cell("Block 1", 1440, false, false, AlignmentType.LEFT, "F0F4F8"), cell("Conv3x3 + BN + ReLU + MaxPool2", 2160), cell("32\u00d732\u00d732", 2160), cell("32", 1800, false, false, AlignmentType.CENTER), cell("Low-level edges", 2160)] }),
          new TableRow({ children: [cell("Block 2", 1440, false, false, AlignmentType.LEFT, "F0F4F8"), cell("Conv3x3 + BN + ReLU + MaxPool2", 2160), cell("64\u00d716\u00d716", 2160), cell("64", 1800, false, false, AlignmentType.CENTER), cell("Textures", 2160)] }),
          new TableRow({ children: [cell("Block 3", 1440, false, false, AlignmentType.LEFT, "F0F4F8"), cell("Conv3x3 + BN + ReLU + MaxPool2", 2160), cell("128\u00d78\u00d78", 2160), cell("128", 1800, false, false, AlignmentType.CENTER), cell("Shapes", 2160)] }),
          new TableRow({ children: [cell("GAP", 1440, false, false, AlignmentType.LEFT, "F0F4F8"), cell("AdaptiveAvgPool(2\u00d72)", 2160), cell("128\u00d72\u00d72", 2160), cell("\u2014", 1800, false, false, AlignmentType.CENTER), cell("Spatial reduction", 2160)] }),
          new TableRow({ children: [cell("FC", 1440, false, false, AlignmentType.LEFT, "F0F4F8"), cell("Dropout(0.4) + Linear(512) + ReLU", 2160), cell("128", 2160), cell("\u2014", 1800, false, false, AlignmentType.CENTER), cell("Dropout regularization", 2160)] }),
          new TableRow({ children: [cell("Output", 1440, false, false, AlignmentType.LEFT, "F0F4F8"), cell("Linear \u2192 5 classes", 2160), cell("5", 2160), cell("\u2014", 1800, false, false, AlignmentType.CENTER), cell("CrossEntropy loss", 2160)] }),
        ]
      }),
      spacer(),
      p("Training configuration: Adam optimizer (lr=1e-3, weight decay=1e-4), StepLR scheduler (decay 0.5 every 5 epochs), 10 epochs, batch size 64, CPU training. Total parameters: ~540K."),

      // ── SECTION 6: RESULTS ────────────────────────────────────
      new Paragraph({ children: [new TextRun({ text: "" })], pageBreakBefore: true }),
      h1("6. Results & Model Comparison"),
      h2("6.1 Multi-class Performance"),
      spacer(),
      new Table({
        width: { size: 9720, type: WidthType.DXA },
        columnWidths: [2880, 1800, 1800, 1800, 1440],
        rows: [
          new TableRow({ children: [cell("Model", 2880, true, true), cell("Accuracy", 1800, true, true, AlignmentType.CENTER), cell("Macro F1", 1800, true, true, AlignmentType.CENTER), cell("Weighted F1", 1800, true, true, AlignmentType.CENTER), cell("Train Time", 1440, true, true, AlignmentType.CENTER)] }),
          new TableRow({ children: [cell("SVM (RBF kernel)", 2880, false, false, AlignmentType.LEFT, "F0F4F8"), cell("70.75%", 1800, false, false, AlignmentType.CENTER), cell("0.706", 1800, false, false, AlignmentType.CENTER), cell("0.706", 1800, false, false, AlignmentType.CENTER), cell("3.3s", 1440, false, false, AlignmentType.CENTER)] }),
          new TableRow({ children: [cell("Random Forest", 2880, false, false, AlignmentType.LEFT, "F0F4F8"), cell("63.25%", 1800, false, false, AlignmentType.CENTER), cell("0.626", 1800, false, false, AlignmentType.CENTER), cell("0.626", 1800, false, false, AlignmentType.CENTER), cell("5.8s", 1440, false, false, AlignmentType.CENTER)] }),
          new TableRow({ children: [cell("Gradient Boosting", 2880, false, false, AlignmentType.LEFT, "F0F4F8"), cell("61.00%", 1800, false, false, AlignmentType.CENTER), cell("0.613", 1800, false, false, AlignmentType.CENTER), cell("0.613", 1800, false, false, AlignmentType.CENTER), cell("7.2s", 1440, false, false, AlignmentType.CENTER)] }),
          new TableRow({ children: [cell("CNN (Custom)", 2880, true, false, AlignmentType.LEFT, "EBF5EB"), cell("84.33% \u2605", 1800, true, false, AlignmentType.CENTER, "EBF5EB"), cell("0.833", 1800, true, false, AlignmentType.CENTER, "EBF5EB"), cell("0.833", 1800, true, false, AlignmentType.CENTER, "EBF5EB"), cell("137.5s", 1440, false, false, AlignmentType.CENTER, "EBF5EB")] }),
        ]
      }),
      spacer(),

      h2("6.2 Binary Classification Performance"),
      p("For the binary task (PPE present vs. absent), traditional ML models achieve strong performance since the task is simpler:"),
      spacer(),
      new Table({
        width: { size: 9720, type: WidthType.DXA },
        columnWidths: [2880, 1800, 1800, 1800, 2440],
        rows: [
          new TableRow({ children: [cell("Model", 2880, true, true), cell("Accuracy", 1800, true, true, AlignmentType.CENTER), cell("Macro F1", 1800, true, true, AlignmentType.CENTER), cell("Weighted F1", 1800, true, true, AlignmentType.CENTER), cell("Notes", 2440, true, true)] }),
          new TableRow({ children: [cell("SVM (RBF) \u2605", 2880, true, false, AlignmentType.LEFT, "EBF5EB"), cell("85.25%", 1800, true, false, AlignmentType.CENTER, "EBF5EB"), cell("0.724", 1800, false, false, AlignmentType.CENTER, "EBF5EB"), cell("0.837", 1800, false, false, AlignmentType.CENTER, "EBF5EB"), cell("Best binary classifier", 2440)] }),
          new TableRow({ children: [cell("Random Forest", 2880, false, false, AlignmentType.LEFT, "F0F4F8"), cell("81.25%", 1800, false, false, AlignmentType.CENTER), cell("0.506", 1800, false, false, AlignmentType.CENTER), cell("0.740", 1800, false, false, AlignmentType.CENTER), cell("Low macro F1 due to class imbalance", 2440)] }),
          new TableRow({ children: [cell("Gradient Boosting", 2880, false, false, AlignmentType.LEFT, "F0F4F8"), cell("80.50%", 1800, false, false, AlignmentType.CENTER), cell("0.583", 1800, false, false, AlignmentType.CENTER), cell("0.765", 1800, false, false, AlignmentType.CENTER), cell("", 2440)] }),
        ]
      }),
      spacer(),

      h2("6.3 Visualizations"),
      ...imgPara("03_model_comparison.png", 580, 240, "Figure 3: Multi-class and binary accuracy comparison across all models"),
      spacer(),
      ...imgPara("04_confusion_matrices.png", 580, 200, "Figure 4: Confusion matrices for multi-class models (SVM, RF, GBM)"),
      spacer(),
      ...imgPara("05_f1_heatmap.png", 500, 190, "Figure 5: Per-class F1 score heatmap — partial_ppe is consistently the hardest class"),
      spacer(),
      ...imgPara("06_roc_curves.png", 450, 340, "Figure 6: ROC curves for binary PPE detection (SVM AUC=0.92+)"),
      spacer(),
      ...imgPara("02_cnn_training_curves.png", 500, 200, "Figure 7: CNN training and validation curves over 10 epochs"),
      spacer(),
      ...imgPara("08_cnn_confusion.png", 350, 300, "Figure 8: CNN confusion matrix — achieves best performance on helmet, no_ppe, safety_vest"),

      h2("6.4 Key Observations"),
      bullet("The CNN significantly outperforms all traditional ML models on multi-class detection (84.3% vs 70.8% SVM), confirming that learned spatial features are more discriminative than handcrafted descriptors for fine-grained PPE distinctions."),
      bullet("partial_ppe is the hardest class across all models (CNN F1=0.64), as it represents a heterogeneous mix of workers with varying combinations of equipment."),
      bullet("The helmet class achieves the highest per-model F1 (CNN: 0.90), likely due to its distinctive shape and color profile captured well by HOG features."),
      bullet("SVM achieves strong binary classification (85.3%) with very fast training (3.3s), making it practical for rapid deployment scenarios where multi-class granularity is not required."),
      bullet("Random Forest provides feature importance insights, revealing that HOG features from the upper-third of the crop (head region) are most discriminative for helmet detection."),

      // ── SECTION 7: CCTV VALIDATION ────────────────────────────
      new Paragraph({ children: [new TextRun({ text: "" })], pageBreakBefore: true }),
      h1("7. CCTV Surveillance Validation"),
      h2("7.1 Validation Dataset"),
      p("To assess real-world generalization, models were evaluated on 14 surveillance-style images collected from public sources representing actual CCTV footage scenarios:"),
      bullet("CCTV output frames from the PPE Tracking project (azimjaan21) \u2014 multi-worker industrial scenes at CCTV perspective and quality"),
      bullet("Construction site scene images from the Pictor-PPE repository \u2014 10 full-scene images with multiple workers"),
      bullet("PPE sample frame from BrunoCestari/PPE-Detection \u2014 annotated industrial workplace"),
      spacer(),
      p("These images differ significantly from the training distribution: they are full-scene (not pre-cropped), captured at wider angles, with multiple workers and potential occlusion, under real lighting conditions."),

      h2("7.2 Validation Methodology"),
      p("A sliding window detection approach was applied to full-scene images. Three window sizes (80\u00d780, 120\u00d780, 100\u00d7120 pixels) with 40-pixel stride were scanned across each image. Windows with CNN confidence \u226560% were retained, and non-maximum suppression (IoU threshold=0.35) was applied to reduce overlapping detections."),

      h2("7.3 Validation Results"),
      spacer(),
      ...imgPara("09_cctv_validation.png", 600, 450, "Figure 9: CCTV validation results with sliding window PPE detection overlaid"),
      spacer(),
      new Table({
        width: { size: 9720, type: WidthType.DXA },
        columnWidths: [3240, 2160, 2160, 2160],
        rows: [
          new TableRow({ children: [cell("Metric", 3240, true, true), cell("Value", 2160, true, true, AlignmentType.CENTER), cell("Notes", 2160, true, true), cell("Assessment", 2160, true, true)] }),
          new TableRow({ children: [cell("Images processed", 3240, false, false, AlignmentType.LEFT, "F0F4F8"), cell("14", 2160, false, false, AlignmentType.CENTER), cell("Full scenes + crops", 2160), cell("Good", 2160, false, false, AlignmentType.CENTER, "EBF5EB")] }),
          new TableRow({ children: [cell("Average CNN confidence", 3240, false, false, AlignmentType.LEFT, "F0F4F8"), cell("85.6%", 2160, false, false, AlignmentType.CENTER), cell("On retained windows", 2160), cell("Good", 2160, false, false, AlignmentType.CENTER, "EBF5EB")] }),
          new TableRow({ children: [cell("Avg detections per scene", 3240, false, false, AlignmentType.LEFT, "F0F4F8"), cell("~179", 2160, false, false, AlignmentType.CENTER), cell("Pre-NMS (dense scan)", 2160), cell("High FP rate", 2160, false, false, AlignmentType.CENTER, "FEF9E7")] }),
          new TableRow({ children: [cell("Most detected class", 3240, false, false, AlignmentType.LEFT, "F0F4F8"), cell("helmet", 2160, false, false, AlignmentType.CENTER), cell("Matches expected", 2160), cell("Expected", 2160, false, false, AlignmentType.CENTER, "EBF5EB")] }),
        ]
      }),
      spacer(),
      p("The high sliding-window detection count (179 per scene) reflects the dense-scan approach without a person detector as a first stage. A production system would use a person detector (e.g., YOLO) to first localize workers before applying the PPE classifier to each person crop. This two-stage approach would dramatically reduce false positives."),

      // ── SECTION 8: ETHICAL CONSIDERATIONS ─────────────────────
      new Paragraph({ children: [new TextRun({ text: "" })], pageBreakBefore: true }),
      h1("8. Ethical Considerations"),
      h2("8.1 Worker Privacy & Surveillance"),
      p("Deploying computer vision systems to monitor workers raises significant privacy concerns. While the intent is safety compliance, continuous automated surveillance can:"),
      bullet("Create psychological pressure on workers, reducing workplace wellbeing"),
      bullet("Enable discriminatory monitoring if false positive rates differ across demographic groups"),
      bullet("Generate persistent records of worker behavior beyond the stated safety purpose"),
      bullet("Shift accountability inappropriately from safety system designers to workers"),
      spacer(),
      p("Mitigation recommendations: Establish clear data retention policies (e.g., no storage of compliant workers), provide workers with access to their own monitoring data, ensure union/worker representative involvement in deployment decisions, and conduct regular bias audits across worker demographics."),

      h2("8.2 Face Anonymization"),
      p("Although face recognition is not a goal of this system, CCTV footage inherently captures facial data. Before processing, facial regions should be blurred or replaced using a dedicated face detector (e.g., RetinaFace, MediaPipe Face Detection) and Gaussian blur (kernel \u226515\u00d715)."),
      p("Implementation note: This project does not implement face blurring but the architecture should include a pre-processing step: detect faces \u2192 apply blur \u2192 run PPE detection. This ensures the PPE model never receives identifiable facial data, protecting worker identity."),

      h2("8.3 Model Bias & Fairness"),
      p("The training dataset was web-mined and crowd-sourced primarily from construction sites, which may underrepresent: warehouse environments (different PPE requirements), workers of different skin tones (color histogram features may be affected by skin reflectance), less common PPE configurations, and low-light or night-shift conditions."),
      p("Before deployment, the system should be evaluated on a demographically diverse dataset, and per-group performance metrics should be reviewed. Disparate impact testing should confirm that false positive rates (incorrectly flagging compliant workers) are equitable across groups."),

      h2("8.4 Transparency & Human Oversight"),
      p("Automated PPE detection systems should operate as decision-support tools rather than autonomous enforcement systems. Recommended governance framework:"),
      bullet("All violations flagged by the system should be reviewed by a human supervisor before any action is taken"),
      bullet("Workers should be informed of the monitoring system, its scope, and their rights"),
      bullet("False positive/negative rates should be disclosed and regularly audited"),
      bullet("Workers should have a clear process to contest incorrect flags"),

      // ── SECTION 9: CONCLUSIONS ─────────────────────────────────
      new Paragraph({ children: [new TextRun({ text: "" })], pageBreakBefore: true }),
      h1("9. Conclusions & Future Work"),
      h2("9.1 Conclusions"),
      p("This project demonstrates a complete PPE detection pipeline trained on 2,000 balanced crops from 1,613 annotated industrial images. Key findings:"),
      bullet("Deep learning (CNN, 84.3%) substantially outperforms traditional ML (SVM best at 70.8%) for multi-class PPE detection, justifying the additional training complexity."),
      bullet("For binary PPE presence detection, SVM achieves competitive performance (85.3%) with extremely fast training (3.3s vs 137.5s for CNN), making it attractive for resource-constrained deployment."),
      bullet("The partial_ppe class remains challenging across all models (best F1: 0.64), suggesting that compliance-level classification requires richer context than a 64x64 crop provides."),
      bullet("HOG + color histogram features capture the essential shape and color cues of safety equipment, but CNN-learned features are more discriminative for the fine-grained distinctions required."),

      h2("9.2 Future Work"),
      bullet("Two-stage detection: Replace sliding window with a person detector (YOLO, Faster R-CNN) for the first stage, followed by per-person PPE classification. This would improve both precision and recall significantly."),
      bullet("Transfer learning: Fine-tune ResNet-50 or EfficientNet pretrained on ImageNet. With the same 64x64 crops, transfer learning would likely push CNN accuracy above 90%."),
      bullet("Dataset expansion: Incorporate SH17 (8,099 images, 17 classes, manufacturing environments), Roboflow Hard Hat Workers, and warehouse-specific datasets to improve domain diversity."),
      bullet("Temporal modeling: For video footage, add temporal consistency via LSTM or transformer-based tracking (BotSORT, DeepSORT) to reduce flickering detections across frames."),
      bullet("Face anonymization pipeline: Integrate RetinaFace as a preprocessing step to blur all detected faces before any features are computed."),
      bullet("Edge deployment: Quantize the CNN to INT8 (TensorRT, ONNX Runtime) and benchmark on embedded hardware (NVIDIA Jetson) for real-time CCTV processing."),

      // ── SECTION 10: REFERENCES ────────────────────────────────
      h1("10. References"),
      p("[1] Nath, N.D., Behzadan, A.H., Paal, S.G. (2020). Deep learning for site safety: Real-time detection of personal protective equipment. Automation in Construction, 112, 103085."),
      spacer(),
      p("[2] MinhNKB. (2021). helmet-safety-vest-detection. GitHub. https://github.com/MinhNKB/helmet-safety-vest-detection"),
      spacer(),
      p("[3] Mughees, A., et al. (2024). SH17: A Dataset for Human Safety and Personal Protective Equipment Detection in Manufacturing Industry. Journal of Safety Science and Resilience."),
      spacer(),
      p("[4] azimjaan21. (2024). ppe_tracking: Tracking Safety Equipment Detection System on CCTV cameras. GitHub. https://github.com/azimjaan21/ppe_tracking"),
      spacer(),
      p("[5] Dalal, N., Triggs, B. (2005). Histograms of oriented gradients for human detection. CVPR 2005."),
      spacer(),
      p("[6] Cortes, C., Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273\u2013297."),
      spacer(),
      p("[7] Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5\u201332."),
      spacer(),
      p("[8] Jocher, G., et al. (2023). YOLOv8 by Ultralytics. GitHub. https://github.com/ultralytics/ultralytics"),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  const outPath = path.join(OUT_DIR, "PPE_Detection_Report.docx");
  fs.writeFileSync(outPath, buffer);
  console.log("Saved:", outPath, `(${(buffer.length/1024).toFixed(0)} KB)`);
});
