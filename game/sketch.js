/**
 * Breakout + Nose Control (p5.js + MediaPipe Face Landmarker + Three.js glasses)
 * - Camera view is rendered on the canvas as background
 * - Nose X controls paddle
 * - Press [Space] or click to launch the ball
 * - Press [R] restart, [C] toggle camera overlay darkness
 *
 * 必要修正重點（已整合在此完整檔案）：
 * 1) Three.js overlay canvas 不只 remove，還要完整 dispose（避免 GPU/記憶體累積）
 * 2) initThree() 一開始先 disposeThree()，避免重複 init 疊資源
 * 3) windowResized() 用 p5 的 width/height 同步 threeRenderer.setSize(width, height)
 * 4) beforeunload 時 disposeThree()，避免關頁殘留 WebGL context
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

// Slim control（可選：讓遮罩更「窄」一點）
let beautySlimX = 0.88; // 0.82~0.95 建議範圍

// --- Three.js (3D glasses) ---
let threeRenderer, threeScene, threeCam;
let threeGlasses = null;
let threeGlassesRoot = null;
let threeReady = false;

// 調整用參數
let glassesZ = 10;          // 模型離相機的深度
let glassesScaleK = 0.4;  // 尺寸縮放係數：用 eyeDist 推尺寸（需微調）

// Camera rendering options
let cameraDim = 0.25;     // 0~1, dark overlay to make game visible (press C to toggle)
let mirrorCamera = true;  // mirror the camera display for natural control
let camYStretch = 1.15;

// Nose tracking
let noseX = null;           // mapped to canvas X
let noseXSmoothed = null;
// const NOSE_CONFIDENCE_MIN = 0.25;
const NOSE_LERP = 0.18;     // smoothing factor

let uiButtons = [];
let uiScale = 1;
const UI_PAD = 12;

// Beauty / skin smoothing
let beautyEnabled = true;
let beautyBlur = 3;         // 2~6 通常合理
let beautyMix = 0.55;       // 0~1，越大越像磨皮
let beautyEveryNFrames = 2; // 降低成本：每 N 帧更新一次模糊結果

let camLayer = null;        // 原始相機緩衝
let camBlurLayer = null;    // 模糊相機緩衝
let faceVideoEllipse = null;// 臉部框（在 video 座標系）

let beautyLayer = null;     // 模糊後要疊上去的那層（已被遮罩）
let beautyMaskLayer = null; // 橢圓漸層遮罩
let beautyFeather = 0.32;   // 0~0.6，越大邊緣越柔、擴散越寬

let glassesPose = null;     // { cx, cy, w, h, rot }
let faceMatrixData = null;     // 4x4 facial transformation matrix (flatten)
let sunglassesMode = false;    // 太陽眼鏡模式（鏡片變黑/變透明度）
let glassesEnabled = true;

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
const BRICK_W_MIN = 55;
const BRICK_W_MAX = 120;
const COLS_MIN = 6;
const COLS_MAX = 14;

/* =========================
 * MediaPipe init
 * ========================= */

async function initFaceLandmarker() {
  const FilesetResolver_ = window.FilesetResolver;
  const FaceLandmarker_ = window.FaceLandmarker;

  if (!FilesetResolver_ || !FaceLandmarker_) {
    throw new Error("MediaPipe Tasks Vision not loaded. Check index.html module import order.");
  }

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
    outputFacialTransformationMatrixes: true,
  };

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

  const OVAL = FaceLandmarker_.FACE_LANDMARKS_FACE_OVAL;
  faceOvalIndices = OVAL ? buildOvalIndexSet(OVAL) : null;

  mpReady = true;
}

function buildOvalIndexSet(connections) {
  if (!connections) return null;
  const s = new Set();
  for (const c of connections) {
    const start = (c.start ?? c[0]);
    const end = (c.end ?? c[1]);
    if (start != null) s.add(start);
    if (end != null) s.add(end);
  }
  return Array.from(s);
}

/* =========================
 * Three.js: 必要完整清理（重點）
 * ========================= */

function disposeThree() {
  // 1) 移除 DOM overlay（保險）
  document.querySelectorAll("canvas.three-overlay").forEach(el => el.remove());

  // 2) 釋放 scene 裡的 geometry/material/texture
  if (threeScene) {
    threeScene.traverse((obj) => {
      if (!obj) return;

      if (obj.geometry) obj.geometry.dispose?.();

      if (obj.material) {
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        for (const m of mats) {
          if (!m) continue;

          const maps = [
            "map", "alphaMap", "aoMap", "bumpMap", "displacementMap",
            "emissiveMap", "envMap", "lightMap", "metalnessMap",
            "normalMap", "roughnessMap"
          ];
          for (const k of maps) {
            if (m[k]) m[k].dispose?.();
          }

          m.dispose?.();
        }
      }
    });
  }

  // 3) 釋放 renderer（重點）
  if (threeRenderer) {
    try {
      // ★ 必要：一定移除 renderer 自己的 canvas（不靠 querySelector）
      if (threeRenderer.domElement && threeRenderer.domElement.parentNode) {
        threeRenderer.domElement.parentNode.removeChild(threeRenderer.domElement);
      }

      threeRenderer.dispose?.();
      // 某些情況下需要 forceContextLoss 才能真的釋放 GPU
      threeRenderer.forceContextLoss?.();
    } catch (e) {
      console.warn("threeRenderer dispose error:", e);
    }
  }

  // 4) 清掉引用
  threeRenderer = null;
  threeScene = null;
  threeCam = null;
  threeGlasses = null;
  threeGlassesRoot = null;
  threeReady = false;
}

// 關頁或重整時釋放 WebGL（必要）
window.addEventListener("beforeunload", () => {
  disposeThree();
});

/* =========================
 * p5 setup / resize
 * ========================= */

function setup() {
  createCanvas(windowWidth, windowHeight);
  ensureBeautyLayers();

  video = createCapture(VIDEO, () => {});
  video.hide();

  initFaceLandmarker().catch(err => {
    console.error("FaceLandmarker init failed:", err);
  });

  // 保底：避免初期 video.width/height = 0
  video.size(640, 480);

  camLayer = createGraphics(video.width, video.height);
  camBlurLayer = createGraphics(video.width, video.height);

  video.elt.onloadedmetadata = () => {
    const vw = video.elt.videoWidth || 0;
    const vh = video.elt.videoHeight || 0;
    if (vw === 0 || vh === 0) return;

    const targetW = 640;
    const targetH = Math.max(1, Math.round(targetW * (vh / vw)));
    video.size(targetW, targetH);

    const bw = Math.max(1, video.width);
    const bh = Math.max(1, video.height);

    camLayer = createGraphics(bw, bh);
    camBlurLayer = createGraphics(bw, bh);
  };

  resetGame();
  applyLayout({ rebuildBricks: true });

  initThree();
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  ensureBeautyLayers();

  const hasBricks = bricks && bricks.length > 0;

  if (!hasBricks) {
    applyLayout({ rebuildBricks: true });
  } else if (gameState === "gameover") {
    applyLayout({ rebuildBricks: true });
  } else {
    applyLayout({ rebuildBricks: false });
  }

  // Three renderer size：用 p5 的 width/height 同步（必要修正）
  if (threeRenderer) {
    threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    threeRenderer.setSize(width, height);
    threeRenderer.domElement.style.width = `${width}px`;
    threeRenderer.domElement.style.height = `${height}px`;
  }

  if (threeCam && threeCam.isPerspectiveCamera) {
    threeCam.aspect = width / height;
    threeCam.updateProjectionMatrix();
  }

  redraw();
}

function ensureBeautyLayers() {
  beautyLayer = createGraphics(width, height);
  beautyMaskLayer = createGraphics(width, height);

  beautyLayer.clear();
  beautyMaskLayer.clear();
}

/* =========================
 * Three.js init
 * ========================= */


function applyFaceRotationToObject(obj, matData) {
  const THREE = window.THREE;
  if (!THREE || !obj || !matData || matData.length !== 16) return false;

  // MediaPipe 的 matrix flatten 後丟進 Matrix4
  let m = new THREE.Matrix4().fromArray(matData);

  // 常見必要：row-major / column-major 修正
  // 如果你發現旋轉整個怪掉，試試把 transpose() 拿掉
  m.transpose();

  // 常見必要：座標系修正（螢幕 y 向下、以及 z 方向差異）
  const conv = new THREE.Matrix4().makeScale(1, -1, -1);
  m.premultiply(conv);

  const pos = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const scl = new THREE.Vector3();
  m.decompose(pos, quat, scl);

  // 自拍鏡像時通常要反 yaw/roll（依你 mirrorCamera）
  if (mirrorCamera) {
    const e = new THREE.Euler().setFromQuaternion(quat, "XYZ");
    e.y = -e.y; // yaw
    e.z = -e.z; // roll
    quat.setFromEuler(e);
  }

  obj.quaternion.copy(quat);
  return true;
}



function applySunglassesMaterial(on) {
  if (!threeGlasses) return;

  threeGlasses.traverse((o) => {
    if (!o.isMesh || !o.material) return;

    const mats = Array.isArray(o.material) ? o.material : [o.material];

    for (const m of mats) {
      // 第一次進來先備份原始材質參數
      if (!m.userData._orig) {
        m.userData._orig = {
          opacity: m.opacity,
          transparent: m.transparent,
          metalness: ("metalness" in m) ? m.metalness : undefined,
          roughness: ("roughness" in m) ? m.roughness : undefined,
          color: m.color ? m.color.clone() : null,
        };
      }

      const orig = m.userData._orig;

      if (on) {
        // 太陽眼鏡：偏黑、稍微透明、反光感一點
        if (m.color) m.color.setRGB(0.05, 0.05, 0.05);
        m.transparent = true;
        m.opacity = 0.65;
        if ("metalness" in m) m.metalness = 0.6;
        if ("roughness" in m) m.roughness = 0.2;
      } else {
        // 還原
        if (orig.color && m.color) m.color.copy(orig.color);
        m.transparent = orig.transparent;
        m.opacity = orig.opacity;
        if ("metalness" in m && orig.metalness !== undefined) m.metalness = orig.metalness;
        if ("roughness" in m && orig.roughness !== undefined) m.roughness = orig.roughness;
      }

      m.needsUpdate = true;
    }
  });
}





function initThree() {
  // 必要：每次 init 前先完整釋放舊資源（避免 GPU/記憶體累積）
  disposeThree();

  const THREE = window.THREE;
  const GLTFLoader = window.GLTFLoader;
  if (!THREE || !GLTFLoader) {
    console.error("Three.js not loaded. Check index.html import order.");
    return;
  }

  threeRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  // 必要：用 p5 畫布尺寸
  threeRenderer.setSize(width, height);

  threeRenderer.domElement.style.position = "fixed";
  threeRenderer.domElement.style.left = "0";
  threeRenderer.domElement.style.top = "0";
  threeRenderer.domElement.style.pointerEvents = "none";
  threeRenderer.domElement.style.zIndex = "999";
  threeRenderer.domElement.className = "three-overlay";
  document.body.appendChild(threeRenderer.domElement);

  threeScene = new THREE.Scene();

  // x: 0..width, y: 0..height（top=height, bottom=0）
  const fov = 45; // 可調：35~55 都常見
  threeCam = new THREE.PerspectiveCamera(fov, width / height, 0.1, 5000);
  threeCam.position.set(0, 0, 900);   // 相機離你「臉平面」的距離
  threeCam.lookAt(0, 0, 0);

  const amb = new THREE.AmbientLight(0xffffff, 1.0);
  threeScene.add(amb);

  const loader = new GLTFLoader();
  loader.load(
    "assets/glasses.glb",
    (gltf) => {
      // 0) root 容器（定位/旋轉/縮放都操作它）
      threeGlassesRoot = new THREE.Group();
      threeScene.add(threeGlassesRoot);

      // 1) 模型本體
      threeGlasses = gltf.scene;

      // 2) pivot 置中（只動模型本體）
      const box = new THREE.Box3().setFromObject(threeGlasses);
      const center = box.getCenter(new THREE.Vector3());
      threeGlasses.position.sub(center);

      // 3) 存尺寸（可選）
      const size = box.getSize(new THREE.Vector3());
      threeGlasses.userData.baseSize = size;

      threeGlasses.rotation.x = Math.PI;

      // 4) 掛到 root
      threeGlassesRoot.add(threeGlasses);
      applySunglassesMaterial(sunglassesMode);

      threeReady = true;
    },
    undefined,
    (err) => console.error("glasses.glb load failed:", err)
  );
}

/* =========================
 * Face update
 * ========================= */

function updateFaceFromMediaPipe() {
  if (!mpReady || !faceLandmarker || !video?.elt) return;
  if (video.elt.readyState < 2) return;

  const t = video.elt.currentTime;
  if (t === lastVideoTime) return;

  const tsMs = performance.now();
  const result = faceLandmarker.detectForVideo(video.elt, tsMs);

  const ftm = result?.facialTransformationMatrixes?.[0];
  faceMatrixData = (ftm && ftm.rows === 4 && ftm.columns === 4 && Array.isArray(ftm.data))
    ? ftm.data.slice()
    : null;

  lastVideoTime = t;

  const faces = result?.faceLandmarks;
  if (!faces || faces.length === 0) {
    faceVideoEllipse = null;
    glassesPose = null;
    return;
  }

  const lm = faces[0];
  const vw = video.elt.videoWidth;
  const vh = video.elt.videoHeight;
  if (!vw || !vh) return;

  // 1) Nose average
  const NOSE_IDXS = [1, 2, 4, 5, 19];
  let nx = 0, ny = 0, ncount = 0;

  for (const idx of NOSE_IDXS) {
    const p = lm[idx];
    if (!p) continue;
    nx += p.x; ny += p.y;
    ncount++;
  }
  if (ncount === 0) return;
  nx /= ncount; ny /= ncount;

  // 2) Face oval bbox
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const idx of faceOvalIndices || []) {
    const p = lm[idx];
    if (!p) continue;
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }

  if (!(maxX > minX && maxY > minY)) {
    minX = 1; minY = 1; maxX = 0; maxY = 0;
    for (const p of lm) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
  }

  const cxN = (minX + maxX) / 2;
  const cyN = (minY + maxY) / 2;

  const faceW = (maxX - minX) * vw;
  const faceH = (maxY - minY) * vh;

  const rx = (faceW * 0.50) * 0.90;
  const ry = (faceH * 0.50) * 0.92;

  faceVideoEllipse = {
    cx: cxN * vw,
    cy: cyN * vh,
    rx,
    ry
  };

  // 3) Nose -> canvas X
  const tf = computeCoverTransform(vw, vh);
  if (!tf) return;

  let px = tf.dx + (nx * vw) * tf.sx;
  if (mirrorCamera) px = width - px;

  noseX = constrain(px, 0, width);
  if (noseXSmoothed == null) noseXSmoothed = noseX;
  noseXSmoothed = lerp(noseXSmoothed, noseX, NOSE_LERP);

  // 4) Glasses pose (eyes)
  const leO = lm[33], leI = lm[133];
  const reO = lm[263], reI = lm[362];

  if (leO && leI && reO && reI) {
    const le = { x: (leO.x + leI.x) / 2, y: (leO.y + leI.y) / 2 };
    const re = { x: (reO.x + reI.x) / 2, y: (reO.y + reI.y) / 2 };

    const toCanvas = (pN) => {
      let x = tf.dx + (pN.x * vw) * tf.sx;
      let y = tf.dy + (pN.y * vh) * tf.sy;
      if (mirrorCamera) x = width - x;
      return { x, y };
    };

    const leC = toCanvas(le);
    const reC = toCanvas(re);

    const dx = reC.x - leC.x;
    const dy = reC.y - leC.y;
    const eyeDist = Math.hypot(dx, dy);

    const cx = (leC.x + reC.x) / 2;
    const cy = (leC.y + reC.y) / 2 + eyeDist * 0.05;

    const rot = Math.atan2(dy, dx);

    const w = eyeDist * 1.75;
    const h = eyeDist * 0.60;

    const ALPHA = 0.25;
    if (!glassesPose) {
      glassesPose = { cx, cy, w, h, rot };
    } else {
      glassesPose.cx = lerp(glassesPose.cx, cx, ALPHA);
      glassesPose.cy = lerp(glassesPose.cy, cy, ALPHA);
      glassesPose.w = lerp(glassesPose.w, w, ALPHA);
      glassesPose.h = lerp(glassesPose.h, h, ALPHA);
      glassesPose.rot = lerpAngle(glassesPose.rot, rot, ALPHA);
    }
  } else {
    glassesPose = null;
  }
}

function lerpAngle(a, b, t) {
  let d = b - a;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return a + d * t;
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

/* =========================
 * Three render
 * ========================= */


function screenToWorld(xPx, yPx, zWorld = 0) {
  const THREE = window.THREE;
  if (!THREE || !threeCam) return { x: 0, y: 0, z: zWorld };
  // 把螢幕像素映射到 z = zWorld 的平面上（配合 PerspectiveCamera）
  const d = threeCam.position.z - zWorld; // 相機到該平面的距離
  const vH = 2 * Math.tan(THREE.MathUtils.degToRad(threeCam.fov / 2)) * d;
  const vW = vH * threeCam.aspect;

  const nx = (xPx / width) - 0.5;   // -0.5..0.5
  const ny = (yPx / height) - 0.5;

  const wx = nx * vW;
  const wy = -ny * vH;              // 螢幕 y 向下，three 世界 y 向上
  return { x: wx, y: wy, z: zWorld };
}



// function applyFaceRotationToObject(obj, matData) {
//   if (!matData || matData.length !== 16) return false;

//   // MediaPipe 的 matrix 是 flattened 16 values；Three.js Matrix4 讀入 array
//   // 注意：如果你發現旋轉整個怪掉（例如 pitch/yaw 軸交換），試著把 transpose() 打開/關掉。
//   const m = new THREE.Matrix4().fromArray(matData);

//   // ★ 很多情況下 mediapipe 的 data 會是 row-major；three.js 內部偏 column-major。
//   // 如果旋轉不對，就改成：m.transpose();
//   m.transpose();

//   // 座標系修正：three 世界 y 向上；你畫面是 y 向下，且 mediapipe 的 z 方向可能相反
//   const conv = new THREE.Matrix4().makeScale(1, -1, -1);
//   m.premultiply(conv);

//   const pos = new THREE.Vector3();
//   const quat = new THREE.Quaternion();
//   const scl = new THREE.Vector3();
//   m.decompose(pos, quat, scl);

//   // 鏡像自拍時，yaw/roll 通常要反向（依你的 mirrorCamera）
//   if (mirrorCamera) {
//     const e = new THREE.Euler().setFromQuaternion(quat, "XYZ");
//     e.y = -e.y; // yaw
//     e.z = -e.z; // roll
//     quat.setFromEuler(e);
//   }

//   obj.quaternion.copy(quat);
//   return true;
// }


function renderThreeGlasses() {
  if (!threeRenderer || !threeScene || !threeCam) return;

  if (threeReady && glassesEnabled && glassesPose && threeGlassesRoot) {
    // 1) 位置：把螢幕像素(cx,cy)投到 3D 世界的 z=0 平面
    const P = screenToWorld(glassesPose.cx, glassesPose.cy, glassesZ);
    threeGlassesRoot.position.set(P.x, P.y, P.z);

    // 2) 旋轉：用 MediaPipe facialTransformationMatrix 取得真 3D 旋轉
    const ok = applyFaceRotationToObject(threeGlassesRoot, faceMatrixData);

    // 3) 尺寸：先沿用 eyeDist 推尺度（可控、好調）
    const s = glassesPose.w * glassesScaleK;
    threeGlassesRoot.scale.set(s, s, s);

    // 4) 拿不到 matrix 時，fallback 用你原本的 2D roll
    if (!ok) {
      const rot = mirrorCamera ? -glassesPose.rot : glassesPose.rot;
      threeGlassesRoot.rotation.set(0, 0, rot);
    }

    threeGlassesRoot.visible = true;
  } else if (threeGlassesRoot) {
    threeGlassesRoot.visible = false;
  }

  threeRenderer.render(threeScene, threeCam);
}

/* =========================
 * draw loop
 * ========================= */

function draw() {
  updateFaceFromMediaPipe();
  drawCameraBackground();
  renderThreeGlasses();

  updatePaddleFromNose();
  updateGame();

  drawBricks();
  drawPaddle();
  drawBall();
  drawHUD();
  drawStateHints();
  drawUIButtons();

  if (noseXSmoothed != null) {
    stroke(255, 40);
    strokeWeight(8);
    line(noseXSmoothed, height / 1.3 - 13, noseXSmoothed, height / 1.3 + 13);
    noStroke();
  }
}

/* =========================
 * UI Buttons
 * ========================= */

function buildUIButtons() {
  uiButtons = [];
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

  y += btnH + gap;
  uiButtons.push(makeBtn("glasses", xRight, y, btnW, btnH, () => {
    glassesEnabled = !glassesEnabled;
  }));
  y += btnH + gap;

  uiButtons.push(makeBtn("sun", xRight, y, btnW, btnH, () => {
    sunglassesMode = !sunglassesMode;
    applySunglassesMaterial(sunglassesMode);
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

function drawUIButtons() {
  if (!uiButtons.length) return;

  push();
  textAlign(CENTER, CENTER);
  textSize(14 * uiScale);
  noStroke();

  for (const b of uiButtons) {
    const label =
      b.id === "beauty" ? (beautyEnabled ? "Beauty: ON" : "Beauty: OFF") :
      b.id === "blur-" ? "Blur -" :
      b.id === "blur+" ? "Blur +" :
      b.id === "mix-" ? "Mix -" :
      b.id === "mix+" ? "Mix +" :
      b.id === "glasses" ? (glassesEnabled ? "Glasses: ON" : "Glasses: OFF") :
      b.id === "sun" ? (sunglassesMode ? "Sunglasses: ON" : "Sunglasses: OFF") :
      b.id;

    fill(0, 160);
    rect(b.x, b.y, b.w, b.h, 14);

    fill(255);
    text(label, b.x + b.w / 2, b.y + b.h / 2);
  }

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

/* =========================
 * Camera background + beauty
 * ========================= */

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

function drawCameraBackground() {
  ensureCamBuffers();
  background(10);

  if (!video || !video.elt) return;
  if (video.elt.readyState < 2) return;

  const cw = width, ch = height;

  const vw = video.elt.videoWidth;
  const vh = video.elt.videoHeight;
  if (!vw || !vh) return;

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
    camLayer.image(video, 0, 0, camLayer.width, camLayer.height);

    if (frameCount % beautyEveryNFrames === 0) {
      camBlurLayer.image(camLayer, 0, 0);
      camBlurLayer.filter(BLUR, beautyBlur);
    }

    const tf = computeCoverTransform(vw, vh);
    if (!tf) return;

    const { dx: tdx, dy: tdy, dw: tdw, dh: tdh, sx, sy } = tf;

    let cx = tdx + faceVideoEllipse.cx * sx;
    let cy = tdy + faceVideoEllipse.cy * sy;
    let rx = faceVideoEllipse.rx * sx;
    let ry = faceVideoEllipse.ry * sy;

    if (mirrorCamera) cx = width - cx;

    const FACE_SHRINK = 0.85;
    rx *= FACE_SHRINK;
    ry *= FACE_SHRINK;

    rx *= beautySlimX;

    const BEAUTY_EXPAND = 1.15;
    rx *= BEAUTY_EXPAND;
    ry *= BEAUTY_EXPAND;

    beautyLayer.clear();
    beautyLayer.push();
    if (mirrorCamera) {
      beautyLayer.translate(width, 0);
      beautyLayer.scale(-1, 1);
      beautyLayer.image(camBlurLayer, tdx, tdy, tdw, tdh);
    } else {
      beautyLayer.image(camBlurLayer, tdx, tdy, tdw, tdh);
    }
    beautyLayer.pop();

    beautyMaskLayer.clear();
    const mctx = beautyMaskLayer.drawingContext;

    const outer = Math.max(rx, ry);
    const inner = outer * (1.0 - beautyFeather);

    const grad = mctx.createRadialGradient(cx, cy, inner, cx, cy, outer);
    grad.addColorStop(0.0, "rgba(255,255,255,1)");
    grad.addColorStop(1.0, "rgba(255,255,255,0)");

    mctx.fillStyle = grad;
    mctx.beginPath();
    mctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    mctx.fill();

    const bctx = beautyLayer.drawingContext;
    bctx.save();
    bctx.globalCompositeOperation = "destination-in";
    beautyLayer.image(beautyMaskLayer, 0, 0);
    bctx.restore();

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

/* =========================
 * Game logic
 * ========================= */

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

function respawn() {
  gameState = "ready";
  resetBall();
  applyLayout({ rebuildBricks: false });
}

function nextLevel() {
  lives = 3;
  rows = Math.min(rows + 1, 10);
  ball.speed = Math.min(ball.speed + 1, 12);

  gameState = "ready";
  resetBall();
  applyLayout({ rebuildBricks: true });
}

function applyLayout(opts = {}) {
  const { rebuildBricks = true } = opts;

  paddle.w = constrain(width * 0.18, 120, 220);
  paddle.h = constrain(height * 0.02, 12, 18);
  paddle.y = height - constrain(height * 0.06, 34, 60);
  paddle.x = constrain(paddle.x ?? width / 2, paddle.w / 2, width - paddle.w / 2);

  brickSideMargin = constrain(width * 0.04, 16, 48);
  brickPadding = constrain(width * 0.01, 6, 12);
  brickTopOffset = constrain(height * 0.10, 60, 110);

  const usableW = width - brickSideMargin * 2;

  if (rebuildBricks) {
    const idealBrickW = constrain(usableW / cols, BRICK_W_MIN, BRICK_W_MAX);
    let newCols = floor((usableW + brickPadding) / (idealBrickW + brickPadding));
    newCols = constrain(newCols, COLS_MIN, COLS_MAX);
    cols = newCols;
  }

  if (rebuildBricks) initBricks();
  else relayoutExistingBricks();

  if (gameState !== "playing") {
    ball.x = paddle.x;
    ball.y = paddle.y - 25;
  } else {
    ball.x = constrain(ball.x, ball.r, width - ball.r);
    ball.y = constrain(ball.y, ball.r, height - ball.r);
  }

  buildUIButtons();
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
      const col = color(random(BRICK_PALETTE));
      bricks.push({ x, y, w: brickW, h: brickH, alive: true, col });
    }
  }
}

function relayoutExistingBricks() {
  if (!bricks || bricks.length === 0) return;

  const usableW = width - brickSideMargin * 2;
  const brickW = (usableW - (cols - 1) * brickPadding) / cols;
  const brickH = BRICK_H;

  for (let i = 0; i < bricks.length; i++) {
    const r = Math.floor(i / cols);
    const c = i % cols;
    if (r >= rows) break;

    const x = brickSideMargin + c * (brickW + brickPadding);
    const y = brickTopOffset + r * (brickH + brickPadding);

    bricks[i].x = x;
    bricks[i].y = y;
    bricks[i].w = brickW;
    bricks[i].h = brickH;
  }
}

function updatePaddleFromNose() {
  const targetX = (noseXSmoothed != null) ? noseXSmoothed : mouseX;
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

  if (ball.x - ball.r <= 0) { ball.x = ball.r; ball.vx *= -1; }
  if (ball.x + ball.r >= width) { ball.x = width - ball.r; ball.vx *= -1; }
  if (ball.y - ball.r <= 0) { ball.y = ball.r; ball.vy *= -1; }

  if (ball.y - ball.r > height) {
    lives -= 1;
    if (lives <= 0) gameState = "gameover";
    else respawn();
    return;
  }

  if (circleRectHit(ball.x, ball.y, ball.r, paddle.x - paddle.w/2, paddle.y - paddle.h/2, paddle.w, paddle.h) && ball.vy > 0) {
    const offset = (ball.x - paddle.x) / (paddle.w / 2);
    const angle = offset * radians(55);
    ball.vx = ball.speed * sin(angle);
    ball.vy = -ball.speed * cos(angle);
    ball.y = paddle.y - paddle.h/2 - ball.r - 0.5;
  }

  for (const b of bricks) {
    if (!b.alive) continue;
    if (circleRectHit(ball.x, ball.y, ball.r, b.x, b.y, b.w, b.h)) {
      b.alive = false;
      score += 1;

      const hitFromLeft = prevX + ball.r <= b.x;
      const hitFromRight = prevX - ball.r >= b.x + b.w;
      if (hitFromLeft || hitFromRight) ball.vx *= -1;
      else ball.vy *= -1;

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

/* =========================
 * Input
 * ========================= */

function keyPressed() {
  if (key === ' ' && gameState === "ready") launchBall();
  if (key === 'r' || key === 'R') {
    resetGame();
    applyLayout({ rebuildBricks: true });
  }
  if (key === 'c' || key === 'C') cameraDim = (cameraDim > 0) ? 0 : 0.25;

  if (key === 'g' || key === 'G') glassesEnabled = !glassesEnabled;
  if (key === 's' || key === 'S') {
    sunglassesMode = !sunglassesMode;
    applySunglassesMaterial(sunglassesMode);
  }

}

function mousePressed() {
  const b = hitBtn(mouseX, mouseY);
  if (b) {
    b.onClick();
    return;
  }
  if (gameState === "ready") launchBall();
}

function touchStarted() {
  const b = hitBtn(touches[0]?.x ?? mouseX, touches[0]?.y ?? mouseY);
  if (b) {
    b.onClick();
    return false;
  }
  if (gameState === "ready") launchBall();
  return false;
}

function launchBall() {
  gameState = "playing";
  const angle = random(radians(200), radians(340));
  ball.vx = ball.speed * cos(angle);
  ball.vy = ball.speed * sin(angle);
  if (ball.vy > 0) ball.vy *= -1;
}

/* =========================
 * Geometry
 * ========================= */

function circleRectHit(cx, cy, cr, rx, ry, rw, rh) {
  const closestX = constrain(cx, rx, rx + rw);
  const closestY = constrain(cy, ry, ry + rh);
  const dx = cx - closestX;
  const dy = cy - closestY;
  return (dx*dx + dy*dy) <= cr*cr;
}