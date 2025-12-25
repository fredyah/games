/**
 * Breakout + Nose Control (p5.js + ml5 BodyPose)
 * - Camera view is rendered on the canvas as background
 * - Nose X controls paddle
 * - Press [Space] or click to launch the ball
 * - Press [R] restart, [C] toggle camera overlay darkness
 */

// Brick colors (palette)
const BRICK_PALETTE = [
  "#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93",
  "#ff924c", "#c77dff", "#4cc9f0", "#f72585", "#90be6d"
];

let video;
let bodyPose;
let poses = [];

// Camera rendering options
let cameraDim = 0.25;        // 0~1, dark overlay to make game visible (press C to toggle)
let mirrorCamera = true;     // mirror the camera display for natural control

// Nose tracking
let noseX = null;           // mapped to canvas X
let noseXSmoothed = null;
const NOSE_CONFIDENCE_MIN = 0.25;
const NOSE_LERP = 0.18;     // smoothing factor

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

function preload() {
  // MoveNet + flipped:true makes L/R align with mirrored feel (like a selfie camera)
  bodyPose = ml5.bodyPose("MoveNet", { flipped: true });
}

function setup() {
  createCanvas(windowWidth, windowHeight);

  video = createCapture(VIDEO, () => {});
  video.size(640, 480);
  video.hide();

  bodyPose.detectStart(video, gotPoses);

  resetGame();
  applyLayout(); // 重要：補齊並在初始化後排版與重建磚塊
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  applyLayout(); // 重要：resize 後重建磚塊與更新尺寸
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

/**
 * 依據螢幕大小做排版：
 * - paddle 尺寸與位置
 * - brickSideMargin / brickPadding / brickTopOffset
 * - cols 動態調整（讓磚塊寬度落在合理區間，並鋪滿整寬度）
 * - 重新 initBricks()
 */
function applyLayout() {
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

  // 根據可用寬度與希望的磚塊寬度區間，動態估算 cols
  // 目的：讓每塊磚不要太小/太大，同時整排鋪滿 usableW
  const idealBrickW = constrain(usableW / cols, BRICK_W_MIN, BRICK_W_MAX);
  let newCols = floor((usableW + brickPadding) / (idealBrickW + brickPadding));
  newCols = constrain(newCols, COLS_MIN, COLS_MAX);
  cols = newCols;

  // 重建磚塊（會用新的 cols / padding / margin / width 重新算 brickW）
  initBricks();

  // 如果不是 playing，球會黏著 paddle；如果正在 playing，做安全限制避免出界
  if (gameState !== "playing") {
    ball.x = paddle.x;
    ball.y = paddle.y - 25;
  } else {
    ball.x = constrain(ball.x, ball.r, width - ball.r);
    ball.y = constrain(ball.y, ball.r, height - ball.r);
  }
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

function gotPoses(results) {
  poses = results || [];
  if (!poses.length) return;

  // pick best nose among persons
  let best = null;
  for (const p of poses) {
    const kps = p.keypoints || [];
    const nose = kps.find(k => k.name === "nose");
    if (!nose) continue;
    if (nose.confidence >= NOSE_CONFIDENCE_MIN) {
      if (!best || nose.confidence > best.confidence) best = nose;
    }
  }
  if (!best) return;

  const mappedX = map(best.x, 0, video.width, 0, width);
  noseX = constrain(mappedX, 0, width);

  if (noseXSmoothed == null) noseXSmoothed = noseX;
  noseXSmoothed = lerp(noseXSmoothed, noseX, NOSE_LERP);
}

function draw() {
  drawCameraBackground();

  updatePaddleFromNose();
  updateGame();

  drawBricks();
  drawPaddle();
  drawBall();
  drawHUD();
  drawStateHints();

  if (noseXSmoothed != null) {
    stroke(255, 15);     // 白色 + 透明度 (0~255)
    strokeWeight(8);      // 線條更粗
    line(noseXSmoothed, 0, noseXSmoothed, height);
    noStroke();
  }
}

/** Render camera to canvas as background (cover mode) */
function drawCameraBackground() {
  background(10);

  if (!video || video.width === 0 || video.height === 0) return;

  const cw = width, ch = height;
  const vw = video.width, vh = video.height;

  // 改名：不要叫 scale，避免蓋到 p5.scale()
  const coverScale = Math.max(cw / vw, ch / vh);

  const dw = vw * coverScale;
  const dh = vh * coverScale;
  const dx = (cw - dw) / 2;
  const dy = (ch - dh) / 2;

  push();
  if (mirrorCamera) {
    translate(width, 0);
    scale(-1, 1);
    image(video, dx, dy, dw, dh);
  } else {
    image(video, dx, dy, dw, dh);
  }
  pop();

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
      gameState = "ready";
      resetBall();
      // 重新套用 layout，避免 resize 後 paddle/ball 不一致
      applyLayout();
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
        rows = Math.min(rows + 1, 10);
        ball.speed = Math.min(ball.speed + 1, 12);
        gameState = "ready";
        resetBall();
        applyLayout(); // 重要：下一關也要重排一次，確保寬度仍鋪滿
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
    applyLayout(); // 重新開始後也套用一次排版
  }
  if (key === 'c' || key === 'C') cameraDim = (cameraDim > 0) ? 0 : 0.25;
}

function mousePressed() {
  if (gameState === "ready") launchBall();
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