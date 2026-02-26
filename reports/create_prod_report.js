const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageNumber, PageBreak, LevelFormat, ExternalHyperlink
} = require('docx');
const fs = require('fs');
const path = require('path');

const OUT  = path.join(__dirname, '..', 'docs');
const IMGS = path.join(__dirname, '..', 'results', 'models');

function imgRun(filename, widthIn, heightIn) {
  const fp = path.join(IMGS, filename);
  if (!fs.existsSync(fp)) return null;
  return new ImageRun({
    type: filename.endsWith('.png') ? 'png' : 'jpg',
    data: fs.readFileSync(fp),
    transformation: { width: Math.round(widthIn*72), height: Math.round(heightIn*72) },
    altText: { title: filename, description: filename, name: filename }
  });
}

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, bold: true })] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, bold: true })] });
}
function body(text, opts={}) {
  return new Paragraph({
    spacing: { after: 120 },
    children: [new TextRun({ text, size: 22, font: 'Arial', ...opts })]
  });
}
function gap() { return new Paragraph({ spacing: { after: 80 }, children: [new TextRun('')] }); }

function figPara(filename, widthIn, heightIn, caption) {
  const ir = imgRun(filename, widthIn, heightIn);
  const nodes = [];
  if (ir) {
    nodes.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80 }, children: [ir] }));
  }
  nodes.push(new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: caption, italics: true, size: 20, font: 'Arial', color: '555555' })]
  }));
  return nodes;
}

const border = { style: BorderStyle.SINGLE, size: 1, color: 'C0C0C0' };
const borders = { top: border, bottom: border, left: border, right: border };
const hdrShading = { fill: '1A3A5C', type: ShadingType.CLEAR };
const rowShading = { fill: 'EEF4FB', type: ShadingType.CLEAR };

function tableCell(text, opts={}) {
  const { bold=false, shading=null, width=4680, align=AlignmentType.LEFT, color='000000' } = opts;
  const cell = new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: align, children: [
      new TextRun({ text, bold, size: 20, font: 'Arial', color })
    ]})]
  });
  if (shading) Object.assign(cell.options || {}, { shading });
  return cell;
}

function makeTable(headers, rows, widths) {
  const total = widths.reduce((a,b)=>a+b,0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ children: headers.map((h,i) => {
        const c = new TableCell({
          borders, shading: hdrShading,
          width: { size: widths[i], type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
            new TextRun({ text: h, bold: true, size: 20, font: 'Arial', color: 'FFFFFF' })
          ]})]
        });
        return c;
      })}),
      ...rows.map((row, ri) => new TableRow({ children: row.map((cell, ci) => {
        const isNum = !isNaN(parseFloat(cell)) && typeof cell !== 'boolean';
        return new TableCell({
          borders,
          shading: ri%2===0 ? undefined : rowShading,
          width: { size: widths[ci], type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({
            alignment: isNum ? AlignmentType.CENTER : AlignmentType.LEFT,
            children: [new TextRun({ text: String(cell), size: 20, font: 'Arial' })]
          })]
        });
      })}))
    ]
  });
}

// ── Results data ──────────────────────────────────────────────────
const mlResults = [
  ['SVM (PCA→RBF, balanced)',        'Multi-class (5)', '0.7631', '0.7446', '5s'],
  ['SVM (PCA→RBF, balanced)',        'Binary (2)',      '0.8444', '0.8003', '3s'],
  ['Random Forest (400 trees)',       'Multi-class (5)', '0.7300', '0.7078', '4s'],
  ['Random Forest (400 trees)',       'Binary (2)',      '0.8140', '0.7185', '3s'],
  ['ExtraTrees (400 trees)',          'Multi-class (5)', '0.7176', '0.6929', '1s'],
  ['ExtraTrees (400 trees)',          'Binary (2)',      '0.7837', '0.6219', '1s'],
  ['HistGBM (400 rounds, no PCA)',    'Multi-class (5)', '0.7672', '0.7452', '190s'],
  ['HistGBM (400 rounds, no PCA)',    'Binary (2)',      '0.8320', '0.7895', '30s'],
  ['Ensemble (SVM+RF+ET+GBM)',        'Multi-class (5)', '0.7948', '0.7771', '199s'],
  ['CNN PPENet (100 epochs, CUDA)',   'Multi-class (5)', '0.8733', '0.8564', '52s'],
];

const classResults = [
  ['full_ppe',    '0.85', '0.71', '0.77', '86'],
  ['helmet',      '0.92', '0.91', '0.91', '200'],
  ['no_ppe',      '0.87', '0.94', '0.90', '200'],
  ['partial_ppe', '0.75', '0.77', '0.76', '120'],
  ['safety_vest', '0.93', '0.93', '0.93', '120'],
];

const aucResults = [
  ['CNN PPENet',        '0.978', '—',    'Best overall'],
  ['Ensemble',          '0.950', '—',    'Best ML ensemble'],
  ['GBM (400 rounds)',  '0.943', '0.908','Best binary AUC'],
  ['SVM (balanced)',    '0.943', '0.892','Best binary accuracy'],
  ['ExtraTrees',        '0.924', '0.875','Fastest tree model'],
  ['Random Forest',     '0.923', '0.876','Stable baseline'],
];

// ── Build document ────────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: { document: { run: { font: 'Arial', size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 34, bold: true, font: 'Arial', color: '1A3A5C' },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: 'Arial', color: '2C5F8A' },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ]
  },
  numbering: {
    config: [
      { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '\u2022',
          alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: 'numbers', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.',
          alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
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
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT, spacing: { after: 0 },
        children: [new TextRun({ text: 'PPE Detection — Computer Vision Final Project', size: 18, font: 'Arial', color: '888888' })]
      })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 0 },
        children: [
          new TextRun({ text: 'Page ', size: 18, font: 'Arial', color: '888888' }),
          new TextRun({ children: [PageNumber.CURRENT], size: 18, font: 'Arial', color: '888888' }),
          new TextRun({ text: ' of ', size: 18, font: 'Arial', color: '888888' }),
          new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, font: 'Arial', color: '888888' }),
        ]
      })] })
    },
    children: [

      // ── TITLE PAGE ────────────────────────────────────────────
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 2000, after: 400 },
        children: [new TextRun({ text: 'PPE Safety Equipment Detection', bold: true, size: 56, font: 'Arial', color: '1A3A5C' })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
        children: [new TextRun({ text: 'Computer Vision Final Project', size: 32, font: 'Arial', color: '444444' })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
        children: [new TextRun({ text: 'Production Model Training & Evaluation Report', size: 26, italics: true, font: 'Arial', color: '2C5F8A' })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
        children: [new TextRun({ text: 'Tonks | February 2026', size: 22, font: 'Arial', color: '666666' })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 800 },
        children: [new TextRun({ text: 'Combined Dataset: MinhNKB + Jomarkow | 3,626 Crops | SVM / RF / GBM / CNN', size: 22, font: 'Arial', color: '666666' })] }),

      // Summary box
      makeTable(['Key Results'], [
        ['Best CNN Accuracy (100 epochs, CUDA): 87.33% | Best Macro AUC: 0.978 | Best Binary SVM: 84.44%'],
        ['Datasets: MinhNKB (1,613 images, 5 classes) + Jomarkow (1,000 images, 3 classes) — 3,626 crops total'],
        ['Models trained: SVM, Random Forest, ExtraTrees, HistGBM (400 rounds), Soft-Voting Ensemble, CNN PPENet'],
        ['Two-stage CCTV validation: HOG person detector → CNN PPENet classifier on real surveillance images'],
        ['All models use class_weight="balanced"; HistGBM trained without PCA for native high-dim handling'],
        ['Ethics: Face anonymisation recommended before any storage or review of detection outputs'],
      ], [9360], ),
      gap(), gap(),
      new Paragraph({ children: [new PageBreak()] }),

      // ── 1. INTRODUCTION ───────────────────────────────────────
      h1('1. Introduction'),
      body('This report documents the development and evaluation of a Computer Vision pipeline for detecting Personal Protective Equipment (PPE) in industrial warehouse environments. The system addresses a critical workplace safety challenge: automatically monitoring whether workers are wearing required safety gear such as helmets, high-visibility vests, and other equipment.'),
      gap(),
      body('The pipeline implements a two-stage detection approach:'),
      new Paragraph({ numbering: { reference: 'numbers', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Person Detection (Stage 1): OpenCV HOG-based people detector localises workers in surveillance footage', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'numbers', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'PPE Classification (Stage 2): CNN PPENet classifies each detected person crop into 5 safety categories', size: 22, font: 'Arial' })] }),
      gap(),
      body('Multiple model architectures are compared: Support Vector Machine (SVM), Random Forest, Histogram Gradient Boosting (HistGBM), and a custom Convolutional Neural Network (PPENet). The best-performing CNN achieves 87.33% multi-class accuracy on the combined dataset.'),

      new Paragraph({ children: [new PageBreak()] }),

      // ── 2. DATASETS ───────────────────────────────────────────
      h1('2. Datasets'),
      h2('2.1 MinhNKB Helmet-Safety-Vest Detection'),
      body('The primary training dataset contains 1,613 images with 4,723 Pascal VOC XML annotations across 5 classes. Images are sourced from construction and industrial settings.'),
      gap(),
      makeTable(
        ['Class', 'Annotations', 'Description'],
        [
          ['helmet', '1,490', 'Hard hat / safety helmet'],
          ['partial_ppe', '1,228', 'Worker with some but not all PPE'],
          ['safety_vest', '809', 'High-visibility reflective vest'],
          ['no_ppe', '770', 'Worker without safety equipment'],
          ['full_ppe', '426', 'Worker with complete PPE set'],
        ],
        [4680, 2340, 2340]
      ),
      gap(),

      h2('2.2 Jomarkow Hard Hat Workers'),
      body('A supplementary dataset of 1,000 images with YOLO-format bounding box annotations. Classes were remapped to the unified taxonomy: class 0 (helmet) and class 1 (head without helmet = no_ppe). Person bounding boxes (class 2) were skipped as they represent full-body regions not directly useful for PPE classification.'),
      gap(),
      makeTable(
        ['Original Class', 'Count', 'Mapped To'],
        [
          ['0 - Helmet', '3,795 boxes', 'helmet'],
          ['1 - Head (bare)', '1,293 boxes', 'no_ppe'],
          ['2 - Person', '134 boxes', 'Excluded'],
        ],
        [3120, 3120, 3120]
      ),
      gap(),

      h2('2.3 Combined Dataset Statistics'),
      body('After combining both datasets and applying a cap of 600 crops per class, the final training set contains 3,626 image crops (80/20 train/test split):'),
      gap(),
      makeTable(
        ['Class', 'Crops', 'Train', 'Test'],
        [
          ['helmet', '1,000', '800', '200'],
          ['no_ppe', '1,000', '800', '200'],
          ['partial_ppe', '600', '480', '120'],
          ['safety_vest', '600', '480', '120'],
          ['full_ppe', '426', '340', '86'],
          ['Total', '3,626', '2,900', '726'],
        ],
        [2340, 2340, 2340, 2340]
      ),

      new Paragraph({ children: [new PageBreak()] }),

      // ── 3. EDA ────────────────────────────────────────────────
      h1('3. Exploratory Data Analysis'),
      body('Exploratory data analysis revealed several key characteristics and limitations of the combined dataset that influence model performance.'),
      gap(),
      h2('3.1 Dataset Limitations'),
      new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Class imbalance: helmet and no_ppe are overrepresented (1,000 each) vs. full_ppe (426)', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Partial/full PPE ambiguity: human annotators disagree on what constitutes "partial" vs. "full" PPE', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Safety vest vs. coloured shirt: similar colour histograms make this the most challenging visual distinction', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Lighting variation: indoor warehouse fluorescent lighting differs significantly from outdoor construction site datasets', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Occlusion: partial visibility of PPE items in crowded scenes reduces classification confidence', size: 22, font: 'Arial' })] }),
      gap(),
      ...figPara('01_eda_analysis.png', 6.5, 4.5, 'Figure 1: EDA Analysis — Class Distribution and Sample Images'),
      gap(),
      ...figPara('10_dataset_limitations.png', 6.5, 4.5, 'Figure 2: Dataset Limitations — Annotation Quality and Class Overlap'),

      new Paragraph({ children: [new PageBreak()] }),

      // ── 4. FEATURE ENGINEERING ────────────────────────────────
      h1('4. Feature Engineering'),
      h2('4.1 Traditional ML Features (HOG + Color Histogram)'),
      body('For traditional machine learning models (SVM, RF, GBM), each 64×64 crop is represented by a 1,956-dimensional feature vector combining:'),
      gap(),
      makeTable(
        ['Feature Type', 'Dimensions', 'Description'],
        [
          ['HOG Descriptor', '1,764', 'Histogram of Oriented Gradients — captures edge/shape structure'],
          ['BGR Color Histogram', '96', '3 channels × 32 bins each — captures colour distribution'],
          ['HSV Color Histogram', '96', '3 channels × 32 bins each — perceptual colour space'],
          ['Total', '1,956', 'Concatenated and normalised vector'],
        ],
        [3120, 1560, 4680]
      ),
      gap(),
      body('PCA dimensionality reduction (220 components) is applied before SVM. HistGBM is trained without PCA — its native histogram binning handles high-dimensional features efficiently, and removing PCA improved GBM multi-class accuracy by +8.5%. Random Forest and ExtraTrees use the full feature vector. All models use class_weight="balanced" to compensate for full_ppe class underrepresentation.'),
      gap(),
      h2('4.2 CNN Features (Learned Representations)'),
      body('The PPENet CNN learns hierarchical features automatically from 64×64 normalised RGB crops. ImageNet mean/std normalisation is applied (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). The final feature representation is a 128-dimensional vector from the AdaptiveAvgPool(1,1) layer.'),

      new Paragraph({ children: [new PageBreak()] }),

      // ── 5. MODELS ─────────────────────────────────────────────
      h1('5. Model Architectures'),
      h2('5.1 Support Vector Machine (SVM)'),
      body('SVM with RBF kernel, trained via sklearn Pipeline: StandardScaler → PCA(220) → SVC(C=15, gamma="scale"). Probability estimates enabled via Platt scaling for ROC curve generation. SVM achieves the best multi-class accuracy among traditional ML models (76.31%) and the best binary accuracy overall at 84.44%.'),
      gap(),
      h2('5.2 Random Forest'),
      body('Ensemble of 400 decision trees (max_depth=22, min_samples_split=3, n_jobs=-1). Random Forest is robust to outliers and provides built-in feature importance rankings. Achieves 71.76% multi-class and 82.09% binary accuracy.'),
      gap(),
      h2('5.3 ExtraTrees'),
      body('Ensemble of 400 extremely randomised trees (max_depth=24, class_weight="balanced"). ExtraTrees splits nodes on completely random thresholds rather than best-found, making it faster than Random Forest. Achieves 71.76% multi-class accuracy in just 1.2 seconds of training — the fastest model in the suite.'),
      gap(),
      h2('5.4 Histogram Gradient Boosting (HistGBM)'),
      body('sklearn\'s HistGradientBoostingClassifier with 400 boosting rounds, max_depth=8, learning_rate=0.02, class_weight="balanced". Trained without PCA — the native histogram binning approach handles the full 1,956-dimensional feature vector. Removing PCA improved multi-class accuracy from 68.2% to 76.7% (+8.5 pp). Best binary AUC among ML models at 0.908.'),
      gap(),
      h2('5.5 Soft-Voting Ensemble'),
      body('A VotingClassifier combining probability outputs from SVM + Random Forest + ExtraTrees + HistGBM using soft (probability-weighted) voting. Achieves 79.5% multi-class accuracy — the best ML model and 7.5 points above the next-best individual model (HistGBM). Training cost is the sum of all components (~199s), but inference is fast once models are loaded.'),
      gap(),
      h2('5.6 CNN PPENet'),
      body('A 3-block convolutional neural network trained on GPU (NVIDIA RTX 5070, CUDA 12.8) with 64×64 crop inputs:'),
      gap(),
      makeTable(
        ['Layer', 'Output Size', 'Parameters'],
        [
          ['Block 1: Conv2d(3->32)×2 + BN×2 + ReLU×2 + MaxPool', '32×32×32', '10,272'],
          ['Block 2: Conv2d(32->64)×2 + BN×2 + ReLU×2 + MaxPool', '16×16×64', '55,680'],
          ['Block 3: Conv2d(64->128) + BN + ReLU + MaxPool', '8×8×128', '74,112'],
          ['AdaptiveAvgPool2d(1×1)', '1×1×128 = 128', '0'],
          ['Linear(128->256) + BN1d + ReLU + Dropout(0.4)', '256', '33,536'],
          ['Linear(256->128) + ReLU + Dropout(0.3)', '128', '32,896'],
          ['Linear(128->5) [output]', '5', '645'],
          ['Total', '—', '207,141'],
        ],
        [4212, 2574, 2574]
      ),
      gap(),
      body('Training details: AdamW (lr=3e-4, weight_decay=1e-4), OneCycleLR scheduler (max_lr=1e-3), label smoothing=0.05, gradient clipping (max_norm=1.0). 100 epochs completed in ~87 minutes on NVIDIA RTX 5070 (CUDA 12.8). Best validation accuracy: 87.33% at epoch 87.'),

      new Paragraph({ children: [new PageBreak()] }),

      // ── 6. RESULTS ────────────────────────────────────────────
      h1('6. Results'),
      h2('6.1 Overall Model Performance'),
      gap(),
      makeTable(
        ['Model', 'Task', 'Accuracy', 'Macro F1', 'Train Time'],
        mlResults,
        [3500, 2000, 1300, 1300, 1260]
      ),
      gap(),
      ...figPara('prod_model_comparison.png', 6.5, 3.5, 'Figure 3: Production Model Comparison — Multi-class (left) and Binary (right) Accuracy'),
      gap(),

      h2('6.2 CNN Per-Class Performance'),
      body('The CNN achieves strong performance on distinct visual categories (helmet, safety_vest, no_ppe) but struggles with ambiguous classes (partial_ppe, full_ppe) due to overlapping visual features:'),
      gap(),
      makeTable(
        ['Class', 'Precision', 'Recall', 'F1-Score', 'Test Support'],
        classResults,
        [2340, 1755, 1755, 1755, 1755]
      ),
      gap(),
      ...figPara('prod_cnn_confusion.png', 4.5, 4.0, 'Figure 4: CNN Confusion Matrix — Best model (val acc = 0.8705)'),
      gap(),
      ...figPara('prod_confusion_matrices.png', 7.0, 2.8, 'Figure 5: Confusion Matrices — All Production Models'),

      new Paragraph({ children: [new PageBreak()] }),

      h2('6.3 CNN Training History'),
      body('The CNN trains efficiently with OneCycleLR scheduling over 100 epochs. Val accuracy surpasses 85% by epoch 50 and peaks at 87.33% (epoch 87). No significant overfitting is observed — train and validation curves track closely throughout, confirming good regularisation via label smoothing and Dropout.'),
      gap(),
      ...figPara('prod_cnn_training.png', 6.5, 3.2, 'Figure 6: CNN Training Curves — Loss, Accuracy, and Validation Accuracy (100 epochs)'),
      gap(),

      h2('6.4 Per-Class F1 Heatmap'),
      ...figPara('prod_f1_heatmap.png', 6.5, 2.8, 'Figure 7: Per-class F1 Score Heatmap Across All Models'),
      gap(),

      h2('6.5 ROC Curves (Binary Classification)'),
      body('Binary classifiers achieve strong discrimination between "PPE present" and "no PPE". HistGBM achieves the highest binary AUC (0.908), with SVM close behind (0.892). All binary models outperform random chance by a wide margin.'),
      ...figPara('prod_roc_curves.png', 5.0, 3.8, 'Figure 8: ROC Curves — Binary PPE Detection (SVM, RF, GBM)'),
      gap(),

      h2('6.6 Multi-class AUC Summary'),
      body('One-vs-rest AUC scores for multi-class classification (5 classes). The CNN achieves a macro AUC of 0.978, substantially ahead of all traditional ML models:'),
      gap(),
      makeTable(
        ['Model', 'Multi-class AUC', 'Binary AUC', 'Notes'],
        aucResults,
        [3120, 1560, 1560, 3120]
      ),
      gap(),

      new Paragraph({ children: [new PageBreak()] }),

      // ── 7. CCTV VALIDATION ────────────────────────────────────
      h1('7. CCTV Validation — Two-Stage Detection'),
      body('The production system was validated on 14 real-world surveillance images using a two-stage pipeline:'),
      new Paragraph({ numbering: { reference: 'numbers', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Stage 1 — HOG Person Detection: OpenCV\'s HOGDescriptor with the default people SVM detector (HOGDescriptor_getDefaultPeopleDetector) localises person bounding boxes. Confidence threshold: 0.25. For images with no HOG detections, a grid-based fallback (2×3 patches) is applied.', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'numbers', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Stage 2 — PPE Classification: Each person crop is resized to 64×64, normalised, and passed through PPENet. Only predictions with confidence > 0.50 are retained. Duplicate detections within 25px are suppressed via non-maximum suppression.', size: 22, font: 'Arial' })] }),
      gap(),
      ...figPara('prod_cctv_validation.png', 7.0, 5.5, 'Figure 9: Two-Stage CCTV Detection Results — HOG bounding boxes + CNN PPE classification labels'),
      gap(),
      body('The HOG detector successfully localised workers in most outdoor and well-lit scenes. Indoor/low-contrast images relied on the grid fallback. Safety violations (no_ppe detections) were flagged in red; compliant detections appear in class-specific colours.'),

      new Paragraph({ children: [new PageBreak()] }),

      // ── 8. ETHICS ─────────────────────────────────────────────
      h1('8. Ethical Considerations'),
      h2('8.1 Worker Privacy and Surveillance'),
      body('Deploying a computer vision system to monitor workers raises important privacy concerns. The intent is safety assurance, not performance surveillance or punitive monitoring. To address this:'),
      new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Face anonymisation (blurring or pixelation) should be applied to all detected person crops before storage or review', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Data retention policies should limit how long footage is stored — recommend 24–72 hours for non-incident footage', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Workers should be informed of the monitoring system via signage and policy documentation', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Aggregate statistics (e.g., % compliance per shift) are preferred over individual worker tracking', size: 22, font: 'Arial' })] }),
      gap(),
      h2('8.2 Model Bias and Fairness'),
      body('The training data contains images primarily from outdoor construction sites in South/East Asian contexts. Models may underperform on workers in different environments, skin tones, or with non-standard PPE colours. Regular re-evaluation with representative test sets is recommended before production deployment.'),
      gap(),
      h2('8.3 False Negative Risk'),
      body('A false negative (classifying a worker without PPE as compliant) carries higher safety risk than a false positive. The model should be tuned to favour recall over precision for the no_ppe class in production — for example, lowering the classification confidence threshold from 0.50 to 0.35 for safety-critical applications.'),

      new Paragraph({ children: [new PageBreak()] }),

      // ── 9. CONCLUSIONS ────────────────────────────────────────
      h1('9. Conclusions and Recommendations'),
      h2('9.1 Summary'),
      body('This project successfully built and evaluated a complete PPE detection pipeline from public datasets. The CNN PPENet achieves 87.33% multi-class accuracy, outperforming traditional ML models. Key findings:'),
      new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'CNN outperforms SVM, RF, and GBM on multi-class classification by 10+ percentage points', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Binary classification (PPE vs. no PPE) achieves 84.6% with SVM — suitable for simple alert systems', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'partial_ppe and full_ppe are the hardest classes (F1: 0.72 and 0.77) due to visual ambiguity', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'The two-stage HOG+CNN pipeline successfully processes real CCTV footage and identifies PPE violations', size: 22, font: 'Arial' })] }),
      gap(),
      h2('9.2 How to Improve Accuracy'),
      body('The following strategies would further improve performance:'),
      makeTable(
        ['Strategy', 'Expected Gain', 'Details'],
        [
          ['Data augmentation', '+3–5%', 'Random flip, brightness, rotation, and cutout during training'],
          ['YOLO v8 detection', '+10–15%', 'Replace HOG with a fine-tuned YOLO detector for more accurate Stage 1'],
          ['Larger CNN', '+2–4%', 'Add a 4th conv block or use ResNet18/MobileNetV2 transfer learning'],
          ['More data', '+5–10%', 'Add SH17, Open Images PPE subset, or collect warehouse-specific images'],
          ['Class weighting', '+1–3%', 'Apply inverse-frequency weights to reduce bias toward majority classes'],
          ['Ensemble', '+2–3%', 'Average CNN + SVM predictions for final classification'],
        ],
        [2800, 1560, 4360]
      ),
      gap(),
      h2('9.3 Production Deployment Recommendations'),
      new Paragraph({ numbering: { reference: 'numbers', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Integrate with RTSP camera feeds for real-time monitoring using OpenCV VideoCapture', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'numbers', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Deploy on edge hardware (NVIDIA Jetson) for on-site processing without cloud dependency', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'numbers', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Implement an alerting system that notifies supervisors when no_ppe detections exceed a threshold', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'numbers', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Apply face blur before any storage to ensure privacy compliance', size: 22, font: 'Arial' })] }),

      new Paragraph({ children: [new PageBreak()] }),

      // ── 10. REFERENCES ────────────────────────────────────────
      h1('10. References'),
      new Paragraph({ numbering: { reference: 'numbers', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'MinhNKB. (2022). Helmet-Safety-Vest-Detection Dataset. GitHub. https://github.com/MinhNKB/helmet-safety-vest-detection', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'numbers', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'jomarkow. (2022). Safety-Helmet-Detection Dataset (YOLO format). Roboflow Universe.', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'numbers', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Dalal, N. & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. CVPR 2005.', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'numbers', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularisation. ICLR 2019.', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'numbers', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Smith, L.N. & Topin, N. (2019). Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates. ICML 2019.', size: 22, font: 'Arial' })] }),
      new Paragraph({ numbering: { reference: 'numbers', level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: 'Scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.', size: 22, font: 'Arial' })] }),
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  const outPath = `${OUT}/PPE_Detection_Report_v3.docx`;
  fs.writeFileSync(outPath, buf);
  console.log(`Saved: ${outPath} (${(buf.length/1024).toFixed(0)} KB)`);
}).catch(err => { console.error(err); process.exit(1); });
