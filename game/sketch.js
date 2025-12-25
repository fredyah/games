/**
 * Breakout + Nose Control (p5.js + ml5 BodyPose)
 * - Camera view is rendered on the canvas as background
 * - Nose X controls paddle
 * - Press [Space] or click to launch the ball
 * - Press [R] restart, [C] toggle camera overlay darkness
 */

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
let cols = 10;
let brickPadding = 8;
let brickTopOffset = 70;
let brickSideMargin = 30;

let score = 0;
let lives = 3;
let gameState = "ready"; // ready | playing | gameover

function preload() {
  // MoveNet + flipped:true makes L/R align with mirrored feel (like a selfie camera)
  bodyPose = ml5.bodyPose("MoveNet", { flipped: true });
}

function setup() {
  createCanvas(900, 600);

  // Camera
  video = createCapture(VIDEO, () => {});
  video.size(640, 480);
  video.hide();

  // Start pose detection loop
  bodyPose.detectStart(video, gotPoses);

  resetGame();
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

function initBricks() {
  bricks = [];
  const usableW = width - brickSideMargin * 2;
  const brickW = (usableW - (cols - 1) * brickPadding) / cols;
  const brickH = 22;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = brickSideMargin + c * (brickW + brickPadding);
      const y = brickTopOffset + r * (brickH + brickPadding);
      bricks.push({ x, y, w: brickW, h: brickH, alive: true });
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
    stroke(0, 255, 0);
    strokeWeight(2);
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
    scale(-1, 1);           // 這裡就會回到 p5 的 scale()
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
        initBricks();
      }
      break;
    }
  }
}

function drawBricks() {
  noStroke();
  for (const b of bricks) {
    if (!b.alive) continue;
    fill(210);
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
  if (key === 'r' || key === 'R') resetGame();
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