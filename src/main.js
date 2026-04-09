import './style.css';
import * as faceapi from '@vladmandic/face-api';

// ─── DOM Refs ────────────────────────────────────────────────────────────────
const video        = document.getElementById('video');
const canvas       = document.getElementById('overlay');
const loadingEl    = document.getElementById('loading-overlay');
const loadingText  = document.getElementById('loading-text');
const infoPanel    = document.getElementById('info-panel');
const panelContent = document.getElementById('panel-content');
const closeBtn     = document.getElementById('close-btn');
const statusDot    = document.getElementById('status-dot');
const statusText   = document.getElementById('status-text');

// ─── Config ──────────────────────────────────────────────────────────────────
const MATCH_THRESHOLD   = 0.78;   // lenient for live-face vs. reference-photo matching
const MIN_FACE_CONF     = 0.4;    // low so even partially occluded faces are found
const CONFIRM_FRAMES    = 2;      // consecutive hits before showing name label

// ─── State ───────────────────────────────────────────────────────────────────
let studentsData   = [];
let imageMap       = {};
let faceMatcher    = null;
let hitCounters    = {};           // rollNo → consecutive hit count
let currentFaces   = [];          // [{rollNo, box, student}] for click handling
let isLocked       = false;

// ─── Init ────────────────────────────────────────────────────────────────────
async function init() {
  try {
    updateStatus('LOADING AI MODELS…', 'idle');
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri('/Ar-Human-Scanner/models'),
      faceapi.nets.faceLandmark68Net.loadFromUri('/Ar-Human-Scanner/models'),
      faceapi.nets.faceRecognitionNet.loadFromUri('/Ar-Human-Scanner/models')
    ]);

    updateStatus('LOADING STUDENT DATABASE…', 'idle');
    const [res, imgRes] = await Promise.all([
      fetch('/Ar-Human-Scanner/data/students.json'),
      fetch('/Ar-Human-Scanner/data/images.json')
    ]);
    studentsData = await res.json();
    imageMap = await imgRes.json();

    updateStatus(`ENCODING FACES…`, 'idle');
    const labeled = await buildDescriptors();

    if (labeled.length > 0) {
      faceMatcher = new faceapi.FaceMatcher(labeled, MATCH_THRESHOLD);
      console.log(`✅ Face matcher ready with ${labeled.length} students`);
    } else {
      console.warn('⚠️ No reference faces encoded — check /public/images/');
    }

    updateStatus('STARTING CAMERA…', 'idle');
    await startCamera();
  } catch (err) {
    console.error(err);
    loadingText.innerText = 'ERROR — CHECK CONSOLE';
  }
}

async function buildDescriptors() {
  const promises = studentsData.map(async (s) => {
    // ── 1. Find the image ──────────────────────────────────────────
    const fileName = imageMap[String(s.rollNo)];
    if (!fileName) {
      return null;
    }

    let srcImg = null;
    let usedUrl = `/Ar-Human-Scanner/images/${fileName}`;
    try {
      srcImg = await faceapi.fetchImage(usedUrl);
    } catch { 
      return null;
    }

    // ── 2. Build descriptors ────────────────────────────────────
    const descriptors = [];
    try {
      const det = await faceapi
        .detectSingleFace(srcImg)
        .withFaceLandmarks()
        .withFaceDescriptor();
      if (det) descriptors.push(det.descriptor);
    } catch { /* may fail */ }

    if (descriptors.length === 0) {
      console.warn(`⚠️ No face detected in ${usedUrl}`);
      return null;
    }

    console.log(`✅ ${s.name} — ${descriptors.length} descriptors (${usedUrl})`);
    return new faceapi.LabeledFaceDescriptors(String(s.rollNo), descriptors);
  });

  const results = await Promise.all(promises);
  return results.filter(res => res !== null);
}

// ── Image Augmentation Helpers ─────────────────────────────────────────────
function flipH(img) {
  const c = document.createElement('canvas');
  c.width = img.width; c.height = img.height;
  const ctx = c.getContext('2d');
  ctx.translate(c.width, 0); ctx.scale(-1, 1);
  ctx.drawImage(img, 0, 0);
  return c;
}

function brighten(img, factor) {
  const c = document.createElement('canvas');
  c.width = img.width; c.height = img.height;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height);
  for (let i = 0; i < d.data.length; i += 4) {
    d.data[i]   = Math.min(255, d.data[i]   * factor);
    d.data[i+1] = Math.min(255, d.data[i+1] * factor);
    d.data[i+2] = Math.min(255, d.data[i+2] * factor);
  }
  ctx.putImageData(d, 0, 0);
  return c;
}

function cropCenter(img, scale) {
  const c = document.createElement('canvas');
  const nw = Math.floor(img.width  * scale);
  const nh = Math.floor(img.height * scale);
  c.width = nw; c.height = nh;
  const ctx = c.getContext('2d');
  const ox = (img.width  - nw) / 2;
  const oy = (img.height - nh) / 2;
  ctx.drawImage(img, ox, oy, nw, nh, 0, 0, nw, nh);
  return c;
}

// Helper: try to resolve the real image URL for a student (used in the info card)
async function resolveImageUrl(rollNo) {
  const fileName = imageMap[String(rollNo)];
  if (fileName) {
    return `/Ar-Human-Scanner/images/${fileName}`;
  }
  return null;
}

// ─── Camera ──────────────────────────────────────────────────────────────────
function startCamera() {
  return new Promise((resolve, reject) => {
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 } } })
      .then(stream => {
        video.srcObject = stream;
        video.onloadedmetadata = () => { video.play(); resolve(); };
      })
      .catch(err => {
        loadingText.innerText = 'CAMERA ERROR — ALLOW ACCESS';
        reject(err);
      });
  });
}

video.addEventListener('play', () => {
  const displaySize = { width: video.videoWidth, height: video.videoHeight };
  faceapi.matchDimensions(canvas, displaySize);
  loadingEl.classList.add('hidden');
  updateStatus('SCANNING', 'scanning');
  startDetectionLoop();
});

// ─── Detection Loop ───────────────────────────────────────────────────────────
let busy = false;

function startDetectionLoop() {
  setInterval(async () => {
    if (busy || isLocked) return;
    busy = true;

    try {
      const dets = await faceapi
        .detectAllFaces(video, new faceapi.SsdMobilenetv1Options({ minConfidence: MIN_FACE_CONF }))
        .withFaceLandmarks()
        .withFaceDescriptors();

      const displaySize = { width: video.videoWidth, height: video.videoHeight };
      const resized = faceapi.resizeResults(dets, displaySize);

      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      currentFaces = [];

      if (resized.length === 0) {
        hitCounters = {};
        updateStatus('SCANNING', 'scanning');
        busy = false;
        return;
      }

      let anyKnown = false;

      resized.forEach(det => {
        const box    = det.detection.box;
        const result = faceMatcher ? faceMatcher.findBestMatch(det.descriptor) : null;

        if (result && result.label !== 'unknown') {
          const rollNo = result.label;
          hitCounters[rollNo] = (hitCounters[rollNo] || 0) + 1;

          if (hitCounters[rollNo] >= CONFIRM_FRAMES) {
            const student = studentsData.find(s => String(s.rollNo) === rollNo);
            if (student) {
              anyKnown = true;
              currentFaces.push({ rollNo, box, student });
              drawFaceBox(ctx, box, '#ffffff', student.name);
            }
          } else {
            // Building confidence — draw dim box
            drawFaceBox(ctx, box, '#555555', '…');
          }
        } else {
          Object.keys(hitCounters).forEach(k => hitCounters[k] = Math.max(0, hitCounters[k] - 1));
          drawFaceBox(ctx, box, '#333333', 'UNKNOWN');
          updateStatus('UNKNOWN FACE', 'unknown');
        }
      });

      if (anyKnown) {
        updateStatus('TAP NAME TO VIEW', 'detected');
      } else if (resized.length > 0) {
        updateStatus('SCANNING', 'scanning');
      }

    } catch (e) {
      console.error(e);
    } finally {
      busy = false;
    }
  }, 120);
}

// ─── Canvas Drawing ───────────────────────────────────────────────────────────
function drawFaceBox(ctx, box, color, name) {
  const { x, y, width, height } = box;
  const arm = 24;
  const isKnown = name !== 'UNKNOWN' && name !== '…';

  ctx.save();

  // Corner bracket frame
  ctx.strokeStyle = color;
  ctx.lineWidth   = isKnown ? 2.5 : 1.5;
  ctx.shadowColor = isKnown ? '#ffffff' : 'transparent';
  ctx.shadowBlur  = isKnown ? 10 : 0;

  const drawCorner = (cx, cy, dx, dy) => {
    ctx.beginPath();
    ctx.moveTo(cx + dx * arm, cy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx, cy + dy * arm);
    ctx.stroke();
  };
  drawCorner(x, y, 1, 1);
  drawCorner(x + width, y, -1, 1);
  drawCorner(x, y + height, 1, -1);
  drawCorner(x + width, y + height, -1, -1);

  // Name chip above box (only for known/unknown labels)
  if (name) {
    ctx.shadowBlur = 0;
    const fontSize  = isKnown ? 14 : 11;
    const chipH     = isKnown ? 28 : 22;
    const padding   = 12;
    ctx.font = `600 ${fontSize}px 'Share Tech Mono', monospace`;
    const textW = ctx.measureText(name).width;
    const chipW = textW + padding * 2;
    const chipX = x;
    const chipY = y - chipH - 6;

    // Chip background
    ctx.fillStyle = isKnown ? 'rgba(255,255,255,0.95)' : 'rgba(40,40,40,0.85)';
    roundRect(ctx, chipX, chipY, chipW, chipH, 4);

    // Chip text
    ctx.fillStyle = isKnown ? '#000000' : '#666666';
    ctx.fillText(name, chipX + padding, chipY + chipH - 8);

    // Store tag bounds for click detection
    if (isKnown) {
      const face = currentFaces.find(f => f.box === box);
      if (face) {
        face.tagBounds = { x: chipX, y: chipY, w: chipW, h: chipH };
      }
    }
  }

  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
  ctx.fill();
}

// ─── Canvas Click → Open Detail Panel ────────────────────────────────────────
canvas.addEventListener('click', (e) => {
  if (isLocked || currentFaces.length === 0) return;

  const rect = canvas.getBoundingClientRect();

  // Map CSS click coord → canvas internal coord (accounting for object-fit:cover)
  const vidAspect = canvas.width / canvas.height;
  const cssAspect = rect.width / rect.height;

  let renderW, renderH, renderX, renderY;
  if (vidAspect > cssAspect) {
    renderH = rect.height; renderW = rect.height * vidAspect;
    renderX = (rect.width - renderW) / 2; renderY = 0;
  } else {
    renderW = rect.width; renderH = rect.width / vidAspect;
    renderX = 0; renderY = (rect.height - renderH) / 2;
  }

  const scaleX  = canvas.width  / renderW;
  const scaleY  = canvas.height / renderH;
  const canvasX = (e.clientX - rect.left - renderX) * scaleX;
  const canvasY = (e.clientY - rect.top  - renderY) * scaleY;

  for (const face of currentFaces) {
    // Allow clicking either the name chip or body box
    const { x, y, width, height } = face.box;
    const inBox  = canvasX >= x && canvasX <= x + width && canvasY >= y && canvasY <= y + height;
    const inTag  = face.tagBounds &&
                   canvasX >= face.tagBounds.x && canvasX <= face.tagBounds.x + face.tagBounds.w &&
                   canvasY >= face.tagBounds.y && canvasY <= face.tagBounds.y + face.tagBounds.h;

    if (inBox || inTag) {
      showStudentPanel(face.student);
      break;
    }
  }
});

// ─── Student Detail Panel ─────────────────────────────────────────────────────
async function showStudentPanel(student) {
  isLocked = true;
  updateStatus('TARGET LOCKED', 'locked');

  // Resolve the correct image URL (handles .jpg / .jpeg / .png)
  const imgUrl = await resolveImageUrl(student.rollNo);
  const imgTag = imgUrl
    ? `<img src="${imgUrl}" alt="Profile" class="id-photo" />`
    : `<div class="id-photo" style="background:#222;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:0.6rem;color:#555;border-radius:50%;">NO PHOTO</div>`;

  panelContent.innerHTML = `
    <div class="card-header">
      <div class="photo-wrap">
        ${imgTag}
      </div>
      <div class="student-headline">
        <div class="student-name">${student.name.toUpperCase()}</div>
        <div class="roll-tag">ROLL — ${student.rollNo}</div>
        <div class="dept-badge">${student.department}</div>
      </div>
    </div>

    <div class="divider"></div>

    <div class="info-grid">
      <div class="info-cell">
        <div class="cell-label">YEAR</div>
        <div class="cell-value">${student.year}</div>
      </div>
      <div class="info-cell cgpa-cell">
        <div class="cell-label">CGPA</div>
        <div class="cell-value big-num">${student.cgpa}</div>
      </div>
    </div>

    <div class="info-row">
      <div class="cell-label">CONTACT</div>
      <div class="cell-value">${student.phone}</div>
    </div>
    <div class="info-row">
      <div class="cell-label">ADDRESS</div>
      <div class="cell-value">${student.address}</div>
    </div>
  `;

  infoPanel.classList.remove('hidden');
  infoPanel.classList.add('visible');
}

// ─── Close ────────────────────────────────────────────────────────────────────
closeBtn.addEventListener('click', () => {
  isLocked    = false;
  hitCounters = {};
  currentFaces = [];

  infoPanel.classList.remove('visible');
  infoPanel.classList.add('hidden');

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  updateStatus('SCANNING', 'scanning');
});

// ─── Status HUD ───────────────────────────────────────────────────────────────
function updateStatus(text, state) {
  statusText.innerText = text;
  statusDot.className  = `status-dot ${state}`;
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
init();
