/**
 * Breakout + Nose Control (p5.js + ml5 BodyPose)
 * - Camera view is rendered on the canvas as background
 * - Nose X controls paddle
 * - Press [Space] or click to launch the ball
 * - Press [R] restart, [C] toggle camera overlay darkness
 */

console.log("sketch.js loaded", new Date().toISOString());
// Brick colors (palette)
const BRICK_PALETTE = [
  "#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93",
  "#ff924c", "#c77dff", "#4cc9f0", "#f72585", "#90be6d"
];

let video;

// MediaPipe Face Landmarker
let faceLandmarker = null;
let mpReady = false;
let lastVideoTime = -1;

// 用來估算臉部外框（橢圓）的索引集合（從 FACE_OVAL connections 轉出來）
let faceOvalIndices = null;

// Slim control（可選：讓遮罩更「窄」一點，視覺顯瘦但不會太假）
let beautySlimX = 0.88; // 0.82~0.95 建議範圍



// Camera rendering options
let cameraDim = 0.25;        // 0~1, dark overlay to make game visible (press C to toggle)
let mirrorCamera = true;     // mirror the camera display for natural control
let camYStretch = 1.15;
// Nose tracking
let noseX = null;           // mapped to canvas X
let noseXSmoothed = null;
const NOSE_CONFIDENCE_MIN = 0.25;
const NOSE_LERP = 0.18;     // smoothing factor


let uiButtons = [];
let uiScale = 1;
const UI_PAD = 12;

// Beauty / skin smoothing
let beautyEnabled = true;
let beautyBlur = 3;     // 2~6 通常合理
let beautyMix = 0.55;   // 0~1，越大越像磨皮
let beautyEveryNFrames = 2; // 降低成本：每 N 帧更新一次模糊結果

let camLayer = null;     // 原始相機緩衝
let camBlurLayer = null; // 模糊相機緩衝
let faceVideoEllipse = null; // 臉部框（在 video 座標系）

let beautyLayer = null;      // 存「模糊後」要疊上去的那層（已被遮罩）
let beautyMaskLayer = null;  // 存橢圓漸層遮罩
let beautyFeather = 0.28;    // 0~0.6，越大邊緣越柔、擴散越寬



// Game
let paddle, ball;
let bricks = [];
let rows = 6;
let cols = 10;              // 會在 applyLayout() 中依螢幕寬度自動調整
let brickPadding = 8;
let brickTopOffset = 70;
let brickSideMargin = 30;

let score = 0;
let lives = 3;
let gameState = "ready"; // ready | playing | gameover

// Responsive bricks config
const BRICK_H = 22;
const BRICK_W_MIN = 55;     // 每塊磚最小寬度（避免太小）
const BRICK_W_MAX = 120;    // 每塊磚最大寬度（避免太大）
const COLS_MIN = 6;
const COLS_MAX = 14;


async function initFaceLandmarker() {
  // 從 window 取（由 index.html 的 module import 掛上來）
  const FilesetResolver_ = window.FilesetResolver;
  const FaceLandmarker_  = window.FaceLandmarker;

  if (!FilesetResolver_ || !FaceLandmarker_) {
    throw new Error("MediaPipe Tasks Vision not loaded. Check index.html module import order.");
  }

  // 建議：這個版本要跟你 import 的一致（你用 latest 就維持 latest)
  const MP_VER = "0.10.14";
  const vision = await FilesetResolver_.forVisionTasks(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/wasm`
  );


  const modelUrl =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

  const common = {
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  };

  // 先嘗試 GPU，不行就 fallback CPU（你原本寫得很好，保留）
  try {
    faceLandmarker = await FaceLandmarker_.createFromOptions(vision, {
      baseOptions: { modelAssetPath: modelUrl, delegate: "GPU" },
      ...common,
    });
  } catch (e) {
    console.warn("GPU delegate failed, fallback to CPU:", e);
    faceLandmarker = await FaceLandmarker_.createFromOptions(vision, {
      baseOptions: { modelAssetPath: modelUrl, delegate: "CPU" },
      ...common,
    });
  }

  // 取得臉外框索引（不同版本可能不存在，保底）
  const OVAL = FaceLandmarker_.FACE_LANDMARKS_FACE_OVAL;
  faceOvalIndices = OVAL ? buildOvalIndexSet(OVAL) : null;

  mpReady = true;
}


function buildOvalIndexSet(connections) {
  if (!connections) return null;            // 防呆：沒有就回 null
  const s = new Set();
  for (const c of connections) {
    // 有些版本 connection 可能是 [start, end] 而不是 {start, end}
    const start = (c.start ?? c[0]);
    const end   = (c.end   ?? c[1]);
    if (start != null) s.add(start);
    if (end != null) s.add(end);
  }
  return Array.from(s);
}


function setup() {
  createCanvas(windowWidth, windowHeight);
  ensureBeautyLayers();

  video = createCapture(VIDEO, () => {});
  video.hide();

  initFaceLandmarker().catch(err => {
    console.error("FaceLandmarker init failed:", err);
  });


  // 保底：先給固定尺寸，避免初期 video.width/height = 0
  video.size(640, 480);

  // beauty buffers 先用保底尺寸建立
  camLayer = createGraphics(video.width, video.height);
  camBlurLayer = createGraphics(video.width, video.height);

  // 等 metadata 出來再做「精準比例」重設
  video.elt.onloadedmetadata = () => {
    const vw = video.elt.videoWidth || 0;
    const vh = video.elt.videoHeight || 0;
    if (vw === 0 || vh === 0) return; // 防呆：避免 0

    const targetW = 640;
    const targetH = Math.max(1, Math.round(targetW * (vh / vw))); // 至少 1
    video.size(targetW, targetH);

    // 防呆：確保不會 0
    const bw = Math.max(1, video.width);
    const bh = Math.max(1, video.height);

    camLayer = createGraphics(bw, bh);
    camBlurLayer = createGraphics(bw, bh);
  };


  resetGame();
  applyLayout();
}



function updateFaceFromMediaPipe() {
  if (!mpReady || !faceLandmarker || !video?.elt) return;
  if (video.elt.readyState < 2) return; // HAVE_CURRENT_DATA

  // 只在 video time 有變化時做一次推論（節省 CPU）
  const t = video.elt.currentTime;
  if (t === lastVideoTime) return;

  const tsMs = performance.now(); // detectForVideo 需要 ms timestamp :contentReference[oaicite:5]{index=5}
  const result = faceLandmarker.detectForVideo(video.elt, tsMs);
  lastVideoTime = t;

  const faces = result?.faceLandmarks;
  if (!faces || faces.length === 0) {
    // 沒偵測到臉：保留 fallback（mouse），並清掉 ellipse
    faceVideoEllipse = null;
    return;
  }

  const lm = faces[0]; // 第一張臉
  const vw = video.elt.videoWidth;
  const vh = video.elt.videoHeight;
  if (!vw || !vh) return;

  // ---- 1) Nose：用「鼻尖附近」幾個點取平均，比單點穩 ----
  // 常見鼻尖/鼻樑附近點位（FaceMesh 468 indices）
  const NOSE_IDXS = [1, 2, 4, 5, 19];
  let nx = 0, ny = 0, ncount = 0;

  for (const idx of NOSE_IDXS) {
    const p = lm[idx];
    if (!p) continue;
    nx += p.x; ny += p.y;
    ncount++;
  }
  if (ncount === 0) return;
  nx /= ncount; ny /= ncount; // normalized 0..1 (image space)

  // ---- 2) Face oval：用臉外框 indices 算 bounding box ----
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const idx of faceOvalIndices || []) {
    const p = lm[idx];
    if (!p) continue;
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }

  // 若外框沒資料，就退回用全部 landmark 的 bbox
  if (!(maxX > minX && maxY > minY)) {
    minX = 1; minY = 1; maxX = 0; maxY = 0;
    for (const p of lm) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
  }

  // 產出「video座標系」ellipse（你的 beauty 系統用的是 video coords）
  const cxN = (minX + maxX) / 2;
  const cyN = (minY + maxY) / 2;

  const faceW = (maxX - minX) * vw;
  const faceH = (maxY - minY) * vh;

  // 你原本就希望「比臉小一點」：保留這個策略
  const rx = (faceW * 0.50) * 0.90;
  const ry = (faceH * 0.50) * 0.92;

  faceVideoEllipse = {
    cx: cxN * vw,
    cy: cyN * vh,
    rx,
    ry
  };

  // ---- 3) 把鼻尖轉成你的 paddle 控制用 noseX（canvas X）----
  // 你的畫面是 cover 模式 + 可選鏡像，所以要走同一套 transform
  const tf = computeCoverTransform(vw, vh);
  if (!tf) return;

  let px = tf.dx + (nx * vw) * tf.sx; // canvas coord
  if (mirrorCamera) px = width - px;

  noseX = constrain(px, 0, width);
  if (noseXSmoothed == null) noseXSmoothed = noseX;
  noseXSmoothed = lerp(noseXSmoothed, noseX, NOSE_LERP);
}



function computeCoverTransform(vw, vh) {
  if (!vw || !vh) return null;

  const cw = width, ch = height;
  const coverScale = Math.max(cw / vw, ch / vh);
  const dw = vw * coverScale;
  const dh = vh * coverScale * camYStretch;
  const dx = (cw - dw) / 2;
  const dy = (ch - dh) / 2;

  return {
    coverScale,
    dx, dy, dw, dh,
    sx: coverScale,
    sy: coverScale * camYStretch
  };
}






function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  ensureBeautyLayers();

  const hasBricks = bricks && bricks.length > 0;

  // 若沒有磚塊，就重建（例如剛初始化、或未建立成功）
  if (!hasBricks) {
    applyLayout({ rebuildBricks: true });
    return;
  }

  // 如果你希望 gameover resize 後也還原一個乾淨畫面，可在此重建
  //（也可以改成 false，保持 gameover 畫面一致）
  if (gameState === "gameover") {
    applyLayout({ rebuildBricks: true });
    return;
  }

  // 正常情況：保留磚塊狀態
  applyLayout({ rebuildBricks: false });
}

function ensureBeautyLayers() {
  // 只要 canvas size 改了，就重建同尺寸的 buffer
  beautyLayer = createGraphics(width, height);
  beautyMaskLayer = createGraphics(width, height);

  // 初始化清空（避免殘影）
  beautyLayer.clear();
  beautyMaskLayer.clear();
}


function resetGame() {
  score = 0;
  lives = 3;
  gameState = "ready";

  paddle = {
    w: 160,
    h: 16,
    x: width / 2,
    y: height - 40,
  };

  resetBall();
  initBricks();
}

function resetBall() {
  ball = {
    r: 10,
    x: width / 2,
    y: height - 60,
    vx: 0,
    vy: 0,
    speed: 8,
  };
}



function ensureCamBuffers() {
  if (!video) return;

  const vw = video.width;
  const vh = video.height;

  if (vw > 0 && vh > 0) {
    if (!camLayer || camLayer.width !== vw || camLayer.height !== vh) {
      camLayer = createGraphics(vw, vh);
    }
    if (!camBlurLayer || camBlurLayer.width !== vw || camBlurLayer.height !== vh) {
      camBlurLayer = createGraphics(vw, vh);
    }
  }
}



function respawn() {
  // 掉一命後的重生：不重建磚塊，只把球/板子狀態回到可發球
  gameState = "ready";
  resetBall();

  // 只做 layout（更新 paddle 尺寸/位置、brick 尺寸等），但不要重建磚塊
  applyLayout({ rebuildBricks: false });
}

function nextLevel() {
  // 過關：生命回滿 + 進下一關 + 重建磚塊 + 增加難度
  lives = 3;

  rows = Math.min(rows + 1, 10);
  ball.speed = Math.min(ball.speed + 1, 12);

  gameState = "ready";
  resetBall();

  // 過關才需要重建磚塊（新關卡）
  applyLayout({ rebuildBricks: true });
}



/**
 * 依據螢幕大小做排版：
 * - paddle 尺寸與位置
 * - brickSideMargin / brickPadding / brickTopOffset
 * - cols 動態調整（讓磚塊寬度落在合理區間，並鋪滿整寬度）
 * - 重新 initBricks()
 */
function applyLayout(opts = {}) {
  const { rebuildBricks = true } = opts;

  // Paddle responsive
  paddle.w = constrain(width * 0.18, 120, 220);
  paddle.h = constrain(height * 0.02, 12, 18);
  paddle.y = height - constrain(height * 0.06, 34, 60);
  paddle.x = constrain(paddle.x ?? width / 2, paddle.w / 2, width - paddle.w / 2);

  // Brick spacing responsive
  brickSideMargin = constrain(width * 0.04, 16, 48);
  brickPadding = constrain(width * 0.01, 6, 12);
  brickTopOffset = constrain(height * 0.10, 60, 110);

  const usableW = width - brickSideMargin * 2;

  // 只有在「要重建磚塊」時才允許改 cols，否則 cols 固定，避免 relayout 亂掉
  if (rebuildBricks) {
    const idealBrickW = constrain(usableW / cols, BRICK_W_MIN, BRICK_W_MAX);
    let newCols = floor((usableW + brickPadding) / (idealBrickW + brickPadding));
    newCols = constrain(newCols, COLS_MIN, COLS_MAX);
    cols = newCols;
  }

  // 只有在需要的時候才重建磚塊
  if (rebuildBricks) {
    initBricks();
  } else {
    // 不重建磚塊：只要重新計算每塊磚的位置/尺寸即可（保留 alive 狀態與顏色）
    relayoutExistingBricks();
  }

  // 如果不是 playing，球會黏著 paddle；如果正在 playing，做安全限制避免出界
  if (gameState !== "playing") {
    ball.x = paddle.x;
    ball.y = paddle.y - 25;
  } else {
    ball.x = constrain(ball.x, ball.r, width - ball.r);
    ball.y = constrain(ball.y, ball.r, height - ball.r);
  }

  buildUIButtons();
}



function buildUIButtons() {
  uiButtons = [];

  // 依螢幕自動縮放
  uiScale = constrain(min(width, height) / 520, 0.85, 1.25);

  const btnH = 44 * uiScale;
  const btnW = 140 * uiScale;
  const gap = 10 * uiScale;
  const xRight = width - UI_PAD - btnW;
  let y = UI_PAD;

  uiButtons.push(makeBtn("beauty", xRight, y, btnW, btnH, () => {
    beautyEnabled = !beautyEnabled;
  }));
  y += btnH + gap;

  uiButtons.push(makeBtn("blur-", xRight, y, (btnW - gap) / 2, btnH, () => {
    beautyBlur = max(1, beautyBlur - 1);
  }));
  uiButtons.push(makeBtn("blur+", xRight + (btnW + gap) / 2, y, (btnW - gap) / 2, btnH, () => {
    beautyBlur = min(10, beautyBlur + 1);
  }));
  y += btnH + gap;

  uiButtons.push(makeBtn("mix-", xRight, y, (btnW - gap) / 2, btnH, () => {
    beautyMix = max(0, beautyMix - 0.05);
  }));
  uiButtons.push(makeBtn("mix+", xRight + (btnW + gap) / 2, y, (btnW - gap) / 2, btnH, () => {
    beautyMix = min(1, beautyMix + 0.05);
  }));
}

function makeBtn(id, x, y, w, h, onClick) {
  return { id, x, y, w, h, onClick };
}

function hitBtn(px, py) {
  for (const b of uiButtons) {
    if (px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h) return b;
  }
  return null;
}





function initBricks() {
  bricks = [];
  const usableW = width - brickSideMargin * 2;
  const brickW = (usableW - (cols - 1) * brickPadding) / cols;
  const brickH = BRICK_H;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = brickSideMargin + c * (brickW + brickPadding);
      const y = brickTopOffset + r * (brickH + brickPadding);
      const col = color(random(BRICK_PALETTE)); // 從調色盤隨機挑
      bricks.push({ x, y, w: brickW, h: brickH, alive: true, col });
    }
  }
}




function relayoutExistingBricks() {
  if (!bricks || bricks.length === 0) return;

  const usableW = width - brickSideMargin * 2;
  const brickW = (usableW - (cols - 1) * brickPadding) / cols;
  const brickH = BRICK_H;

  // 你現在 bricks 是一維陣列，且建立順序是 r(0..rows-1), c(0..cols-1)
  // 但 resize 後 cols 可能變了，原本的 bricks.length 也可能不等於 rows*cols
  // 因此用「目前 bricks.length」去重新對應 row/col
  for (let i = 0; i < bricks.length; i++) {
    const r = Math.floor(i / cols);
    const c = i % cols;

    // 若 r 超出 rows，代表原先數量不匹配（例如 cols 變化過大）
    // 這種情況保守處理：直接停止或把超出的磚留在原地都行。
    // 我這裡選擇：超出 rows 的就不再更新（避免亂跳）
    if (r >= rows) break;

    const x = brickSideMargin + c * (brickW + brickPadding);
    const y = brickTopOffset + r * (brickH + brickPadding);

    bricks[i].x = x;
    bricks[i].y = y;
    bricks[i].w = brickW;
    bricks[i].h = brickH;
  }
}



// function gotPoses(results) {
//   poses = results || [];
//   if (!poses.length) return;

//   // pick best nose among persons, keep the corresponding pose
//   let bestPose = null;
//   let bestNose = null;

//   for (const p of poses) {
//     const kps = p.keypoints || [];
//     const nose = kps.find(k => k.name === "nose");
//     if (!nose) continue;
//     if (nose.confidence >= NOSE_CONFIDENCE_MIN) {
//       if (!bestNose || nose.confidence > bestNose.confidence) {
//         bestNose = nose;
//         bestPose = p;
//       }
//     }
//   }
//   if (!bestNose || !bestPose) return;

//   const vW = video.elt?.videoWidth || video.width;
//   const vH = video.elt?.videoHeight || video.height;

//   // ---- Nose X mapping (handle mirror here) ----
//   const mappedX = mirrorCamera
//     ? map(bestNose.x, 0, vW, width, 0)   // mirror on canvas
//     : map(bestNose.x, 0, vW, 0, width);

//   noseX = constrain(mappedX, 0, width);

//   if (noseXSmoothed == null) noseXSmoothed = noseX;
//   noseXSmoothed = lerp(noseXSmoothed, noseX, NOSE_LERP);

//   // ---- Face ellipse (video coords) ----
//   const kps = bestPose.keypoints || [];
//   const pick = (name) => kps.find(k => k.name === name && k.confidence >= NOSE_CONFIDENCE_MIN);

//   const nose = pick("nose");
//   const le = pick("left_eye");
//   const re = pick("right_eye");
//   const lea = pick("left_ear");
//   const rea = pick("right_ear");

//   if (!nose && !(le && re)) {
//     faceVideoEllipse = null;
//     return;
//   }

//   // center x: prefer eyes midpoint, fallback to nose
//   const cx = (le && re) ? (le.x + re.x) / 2 : nose.x;

//   // a more "face-like" center y:
//   // - base: between eyes and nose
//   const eyeY = (le && re) ? (le.y + re.y) / 2 : (nose ? nose.y - 20 : 0);
//   const baseCy = nose ? (eyeY * 0.45 + nose.y * 0.55) : eyeY;

//   // width estimate:
//   // - ears distance is best; fallback to eye distance * factor
//   let baseW = null;
//   if (lea && rea) baseW = dist(lea.x, lea.y, rea.x, rea.y);
//   else if (le && re) baseW = dist(le.x, le.y, re.x, re.y) * 2.4;

//   // fallback if still null
//   if (!baseW) baseW = vW * 0.28;

//   // clamp + make it "a bit smaller than face"
//   baseW = constrain(baseW, vW * 0.14, vW * 0.48);
//   const faceW = baseW * 0.88;       // smaller (你要的「比臉小一點」)
//   const faceH = faceW * 1.25;       // typical face aspect

//   // nudge center slightly upward (cover forehead, avoid too much neck)
//   const cy = constrain(baseCy - faceH * 0.03, 0, vH);

//   faceVideoEllipse = {
//     cx: constrain(cx, 0, vW),
//     cy,
//     rx: faceW * 0.50,
//     ry: faceH * 0.50
//   };
// }


function draw() {
  updateFaceFromMediaPipe();  // 先更新 noseXSmoothed + faceVideoEllipse
  drawCameraBackground();

  updatePaddleFromNose();
  updateGame();

  drawBricks();
  drawPaddle();
  drawBall();
  drawHUD();
  drawStateHints();
  drawUIButtons();

  if (noseXSmoothed != null) {
    stroke(255, 40);     // 白色 + 透明度 (0~255)
    strokeWeight(8);      // 線條更粗
    line(noseXSmoothed, height / 1.3 - 13, noseXSmoothed, height / 1.3 + 13);
    noStroke();
  }
}


function drawUIButtons() {
  if (!uiButtons.length) return;

  push();
  textAlign(CENTER, CENTER);
  textSize(14 * uiScale);
  noStroke();

  for (const b of uiButtons) {
    // 背景
    const isBeauty = b.id === "beauty";
    const label =
      b.id === "beauty" ? (beautyEnabled ? "Beauty: ON" : "Beauty: OFF") :
      b.id === "blur-" ? "Blur -" :
      b.id === "blur+" ? "Blur +" :
      b.id === "mix-"  ? "Mix -" :
      b.id === "mix+"  ? "Mix +" : b.id;

    // 半透明黑底，圓角，看起來像浮動按鈕
    fill(0, 160);
    rect(b.x, b.y, b.w, b.h, 14);

    // 文字
    fill(255);
    text(label, b.x + b.w / 2, b.y + b.h / 2);
  }

  // 顯示數值（可選）
  fill(0, 160);
  const infoW = 200 * uiScale;
  const infoH = 34 * uiScale;
  const ix = width - UI_PAD - infoW;
  const iy = (UI_PAD + (44 * uiScale + 10 * uiScale) * 3);
  rect(ix, iy, infoW, infoH, 14);
  fill(255);
  text(`Blur: ${beautyBlur}  Mix: ${nf(beautyMix, 1, 2)}`, ix + infoW / 2, iy + infoH / 2);

  pop();
}



/** Render camera to canvas as background (cover mode) */
function drawCameraBackground() {
  ensureCamBuffers();
  background(10);

  // 1) 基本存在性檢查
  if (!video || !video.elt) return;

  // 2) 等到 video 有「目前幀」可以畫（HAVE_CURRENT_DATA = 2）
  if (video.elt.readyState < 2) return;

  const cw = width, ch = height;

  // 3) 一定要用 videoWidth/videoHeight（原生真實尺寸）
  const vw = video.elt.videoWidth;
  const vh = video.elt.videoHeight;
  if (!vw || !vh) return;

  // 4) cover 模式（鋪滿畫布）
  const coverScale = Math.max(cw / vw, ch / vh);
  const dw = vw * coverScale;
  const dh = vh * coverScale * camYStretch;
  const dx = (cw - dw) / 2;
  const dy = (ch - dh) / 2;

  push();
  if (mirrorCamera) {
    translate(width, 0);
    scale(-1, 1);
  }
  image(video, dx, dy, dw, dh);
  pop();

    // ---- Beauty smoothing overlay (face only) ----

const camReady =
  video &&
  (video.elt?.videoWidth > 0) &&
  (video.elt?.videoHeight > 0) &&
  camLayer && camBlurLayer &&
  camLayer.width > 0 && camLayer.height > 0 &&
  camBlurLayer.width > 0 && camBlurLayer.height > 0;

if (beautyEnabled && camReady && faceVideoEllipse) {
    // 1) Update buffers
    camLayer.image(video, 0, 0, camLayer.width, camLayer.height);

    if (frameCount % beautyEveryNFrames === 0) {
      camBlurLayer.image(camLayer, 0, 0);
      camBlurLayer.filter(BLUR, beautyBlur);
    }

    // 2) Convert face ellipse (video coords) -> canvas coords using cover transform
    const vw = video.elt.videoWidth;
    const vh = video.elt.videoHeight;
    const tf = computeCoverTransform(vw, vh);
    if (!tf) return;

    const { dx, dy, dw, dh, sx, sy } = tf;

    let cx = dx + faceVideoEllipse.cx * sx;
    let cy = dy + faceVideoEllipse.cy * sy;
    let rx = faceVideoEllipse.rx * sx;
    let ry = faceVideoEllipse.ry * sy;

    // mirror correction（你原本的邏輯保留）
    if (mirrorCamera) cx = width - cx;

    // 你的 FACE_SHRINK 保留（整體縮小一點更自然）
    const FACE_SHRINK = 0.85;
    rx *= FACE_SHRINK;
    ry *= FACE_SHRINK;

    // 顯瘦：只在 X 軸再縮一點（比「整體縮小」更像瘦臉）
    rx *= beautySlimX;

    // 3) beautyLayer：先把「模糊相機」畫上去（整張畫面）
    beautyLayer.clear();
    beautyLayer.push();
    if (mirrorCamera) {
      beautyLayer.translate(width, 0);
      beautyLayer.scale(-1, 1);
      beautyLayer.image(camBlurLayer, dx, dy, dw, dh);
    } else {
      beautyLayer.image(camBlurLayer, dx, dy, dw, dh);
    }
    beautyLayer.pop();

    // 4) 產生「漸層遮罩」：中心不透明、邊緣淡出
    beautyMaskLayer.clear();
    const mctx = beautyMaskLayer.drawingContext;

    // feather 寬度：以較大的半徑當基準
    const outer = Math.max(rx, ry);
    const inner = outer * (1.0 - beautyFeather);  // inner 越小，羽化越寬

    const grad = mctx.createRadialGradient(cx, cy, inner, cx, cy, outer);
    grad.addColorStop(0.0, "rgba(255,255,255,1)");
    grad.addColorStop(1.0, "rgba(255,255,255,0)");

    mctx.fillStyle = grad;
    mctx.beginPath();
    // 用 ellipse 畫遮罩形狀（不是圓）
    mctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    mctx.fill();

    // 5) 把遮罩套到 beautyLayer（destination-in）
    const bctx = beautyLayer.drawingContext;
    bctx.save();
    bctx.globalCompositeOperation = "destination-in";
    beautyLayer.image(beautyMaskLayer, 0, 0);
    bctx.restore();

    // 6) 疊回主畫面（用你的 beautyMix 控制強度）
    drawingContext.save();
    drawingContext.globalAlpha = beautyMix;
    image(beautyLayer, 0, 0);
    drawingContext.restore();
  }


  if (cameraDim > 0) {
    noStroke();
    fill(0, 255 * cameraDim);
    rect(0, 0, width, height);
  }

}

function updatePaddleFromNose() {
  // If nose not found, allow mouse as fallback
  let targetX = (noseXSmoothed != null) ? noseXSmoothed : mouseX;
  paddle.x = constrain(targetX, paddle.w / 2, width - paddle.w / 2);
}

function updateGame() {
  if (gameState !== "playing") {
    ball.x = paddle.x;
    ball.y = paddle.y - 25;
    return;
  }

  const prevX = ball.x;
  const prevY = ball.y;

  ball.x += ball.vx;
  ball.y += ball.vy;

  // Walls
  if (ball.x - ball.r <= 0) { ball.x = ball.r; ball.vx *= -1; }
  if (ball.x + ball.r >= width) { ball.x = width - ball.r; ball.vx *= -1; }
  if (ball.y - ball.r <= 0) { ball.y = ball.r; ball.vy *= -1; }

  // Bottom = lose life
  if (ball.y - ball.r > height) {
    lives -= 1;

    if (lives <= 0) {
      gameState = "gameover";
    } else {
      // 掉一命：不重置磚塊
      respawn();
    }
    return;
  }

  // Paddle collision
  if (circleRectHit(ball.x, ball.y, ball.r, paddle.x - paddle.w/2, paddle.y - paddle.h/2, paddle.w, paddle.h) && ball.vy > 0) {
    const offset = (ball.x - paddle.x) / (paddle.w / 2); // -1..1
    const angle = offset * radians(55);
    ball.vx = ball.speed * sin(angle);
    ball.vy = -ball.speed * cos(angle);
    ball.y = paddle.y - paddle.h/2 - ball.r - 0.5;
  }

  // Brick collisions
  for (const b of bricks) {
    if (!b.alive) continue;
    if (circleRectHit(ball.x, ball.y, ball.r, b.x, b.y, b.w, b.h)) {
      b.alive = false;
      score += 1;

      // bounce direction based on previous position
      const hitFromLeft   = prevX + ball.r <= b.x;
      const hitFromRight  = prevX - ball.r >= b.x + b.w;
      if (hitFromLeft || hitFromRight) ball.vx *= -1;
      else ball.vy *= -1;

      // Win condition: next level
      if (bricks.every(x => !x.alive)) {
        nextLevel();
      }
      break;
    }
  }
}

function drawBricks() {
  noStroke();
  for (const b of bricks) {
    if (!b.alive) continue;
    fill(b.col);
    rect(b.x, b.y, b.w, b.h, 6);
  }
}

function drawPaddle() {
  noStroke();
  fill(255);
  rectMode(CENTER);
  rect(paddle.x, paddle.y, paddle.w, paddle.h, 10);
  rectMode(CORNER);
}

function drawBall() {
  noStroke();
  fill(255);
  circle(ball.x, ball.y, ball.r * 2);
}

function drawHUD() {
  fill(255);
  textSize(16);
  textAlign(LEFT, TOP);
  text(`Score: ${score}`, 14, 12);
  text(`Lives: ${lives}`, 14, 32);

  const noseOk = noseXSmoothed != null;
  text(`Nose: ${noseOk ? "OK" : "Not detected (fallback: mouse)"}`, 14, 52);

  textSize(12);
  text(`Keys: Space/Click=Launch, R=Restart, C=Toggle Dim`, 14, 74);
}

function drawStateHints() {
  textAlign(CENTER, CENTER);

  if (gameState === "ready") {
    fill(255);
    textSize(18);
    text("Press SPACE or Click to launch", width/2, height/2);
    textSize(14);
    text("Use your nose to move the paddle left/right", width/2, height/2 + 26);
  } else if (gameState === "gameover") {
    fill(255);
    textSize(28);
    text("GAME OVER", width/2, height/2 - 10);
    textSize(16);
    text("Press R to restart", width/2, height/2 + 20);
  }
}

function keyPressed() {
  if (key === ' ' && gameState === "ready") launchBall();
  if (key === 'r' || key === 'R') {
    resetGame();
    applyLayout({ rebuildBricks: true }); // 重新開始後也套用一次排版
  }
  if (key === 'c' || key === 'C') cameraDim = (cameraDim > 0) ? 0 : 0.25;
}

function mousePressed() {
  // 先檢查是否點到 UI 按鈕
  const b = hitBtn(mouseX, mouseY);
  if (b) {
    b.onClick();
    return;
  }

  // 沒點到按鈕才 launch
  if (gameState === "ready") launchBall();
}

function touchStarted() {
  const b = hitBtn(touches[0]?.x ?? mouseX, touches[0]?.y ?? mouseY);
  if (b) {
    b.onClick();
    return false; // 阻止事件往下（避免滾動/雙擊縮放）
  }

  if (gameState === "ready") launchBall();
  return false;
}


function launchBall() {
  gameState = "playing";

  // Launch upward with slight random angle
  const angle = random(radians(200), radians(340));
  ball.vx = ball.speed * cos(angle);
  ball.vy = ball.speed * sin(angle);
  if (ball.vy > 0) ball.vy *= -1;
}

// Geometry: circle-rect collision
function circleRectHit(cx, cy, cr, rx, ry, rw, rh) {
  const closestX = constrain(cx, rx, rx + rw);
  const closestY = constrain(cy, ry, ry + rh);
  const dx = cx - closestX;
  const dy = cy - closestY;
  return (dx*dx + dy*dy) <= cr*cr;
}