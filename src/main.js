import './style.css';
import * as faceapi from '@vladmandic/face-api';

// ─── DOM Refs ────────────────────────────────────────────────────────────────
const video        = document.getElementById('video');
const canvas       = document.getElementById('overlay');
const loadingEl    = document.getElementById('loading-overlay');
const loadingText  = document.getElementById('loading-text');
const statusDot    = document.getElementById('status-dot');
const statusText   = document.getElementById('status-text');

// ─── Config ──────────────────────────────────────────────────────────────────
const MATCH_THRESHOLD   = 0.78;
const MIN_FACE_CONF     = 0.4;
const CONFIRM_FRAMES    = 2;

// ─── State ───────────────────────────────────────────────────────────────────
let studentsData   = [];
let imageMap       = {};
let faceMatcher    = null;
let hitCounters    = {};
let currentFaces   = [];

// Track which rollNos currently have a visible popup (so we don't spam-recreate them)
const activePopups = new Map(); // rollNo -> popup DOM element
const openModals   = new Set(); // rollNos whose detail modal is open

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

    updateStatus('ENCODING FACES…', 'idle');
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
    const fileName = imageMap[String(s.rollNo)];
    if (!fileName) return null;

    let srcImg = null;
    const usedUrl = `/Ar-Human-Scanner/images/${fileName}`;
    try {
      srcImg = await faceapi.fetchImage(usedUrl);
    } catch {
      return null;
    }

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

// ─── Camera ──────────────────────────────────────────────────────────────────
function startCamera() {
  return new Promise((resolve, reject) => {
    // Prefer back camera; fall back gracefully
    const constraints = {
      video: {
        facingMode: { ideal: 'environment' },
        width:  { ideal: 1280 },
        height: { ideal: 720 }
      }
    };
    navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {
        video.srcObject = stream;
        video.onloadedmetadata = () => { video.play(); resolve(); };
      })
      .catch(() => {
        // Fallback: any camera
        navigator.mediaDevices.getUserMedia({ video: true })
          .then(stream => {
            video.srcObject = stream;
            video.onloadedmetadata = () => { video.play(); resolve(); };
          })
          .catch(err => {
            loadingText.innerText = 'CAMERA ERROR — ALLOW ACCESS';
            reject(err);
          });
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
    if (busy) return;
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
        removeStalePopups([]);
        busy = false;
        return;
      }

      let anyKnown = false;
      const seenRollNos = [];

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
              seenRollNos.push(rollNo);
              currentFaces.push({ rollNo, box, student });
              drawFaceBox(ctx, box, '#ffffff', student.name);

              // Show or update the floating popup
              showOrUpdatePopup(rollNo, student, box);
            }
          } else {
            drawFaceBox(ctx, box, '#555555', '…');
          }
        } else {
          Object.keys(hitCounters).forEach(k => hitCounters[k] = Math.max(0, hitCounters[k] - 1));
          drawFaceBox(ctx, box, '#333333', 'UNKNOWN');
          updateStatus('UNKNOWN FACE', 'unknown');
        }
      });

      // Remove popups for faces no longer visible
      removeStalePopups(seenRollNos);

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

// ─── Floating Popup Cards ────────────────────────────────────────────────────
// These sit absolutely over the canvas for each detected person.

function getCanvasRect() {
  return canvas.getBoundingClientRect();
}

function canvasToScreen(box) {
  const rect = getCanvasRect();
  const vidAspect = canvas.width / canvas.height;
  const cssAspect = rect.width / rect.height;

  let renderW, renderH, renderX, renderY;
  if (vidAspect > cssAspect) {
    renderH = rect.height; renderW = rect.height * vidAspect;
    renderX = rect.left + (rect.width - renderW) / 2; renderY = rect.top;
  } else {
    renderW = rect.width; renderH = rect.width / vidAspect;
    renderX = rect.left; renderY = rect.top + (rect.height - renderH) / 2;
  }

  const scaleX = renderW / canvas.width;
  const scaleY = renderH / canvas.height;

  return {
    left:   renderX + box.x * scaleX,
    top:    renderY + box.y * scaleY,
    width:  box.width  * scaleX,
    height: box.height * scaleY
  };
}

function showOrUpdatePopup(rollNo, student, box) {
  // Don't layer a popup over an open detail modal for the same person
  if (openModals.has(rollNo)) return;

  const screenBox = canvasToScreen(box);
  let popup = activePopups.get(rollNo);

  if (!popup) {
    popup = createPopupEl(rollNo, student);
    document.getElementById('scanner-root').appendChild(popup);
    activePopups.set(rollNo, popup);
    // Animate in
    requestAnimationFrame(() => popup.classList.add('popup-visible'));
  }

  // Reposition above the face box
  const popupW = 200;
  let left = screenBox.left + screenBox.width / 2 - popupW / 2;
  let top  = screenBox.top - 90;

  // Clamp to window
  left = Math.max(8, Math.min(left, window.innerWidth - popupW - 8));
  top  = Math.max(8, top);

  popup.style.left  = left + 'px';
  popup.style.top   = top  + 'px';
  popup.style.width = popupW + 'px';
}

function createPopupEl(rollNo, student) {
  const div = document.createElement('div');
  div.className = 'face-popup';
  div.dataset.rollno = rollNo;

  div.innerHTML = `
    <div class="popup-name">${student.name.toUpperCase()}</div>
    <button class="popup-details-btn" id="popup-details-${rollNo}">SHOW DETAILS</button>
  `;

  div.querySelector('.popup-details-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    openDetailModal(rollNo, student);
  });

  return div;
}

function removeStalePopups(seenRollNos) {
  for (const [rollNo, popup] of activePopups.entries()) {
    if (!seenRollNos.includes(rollNo) && !openModals.has(rollNo)) {
      popup.classList.remove('popup-visible');
      setTimeout(() => {
        popup.remove();
        activePopups.delete(rollNo);
      }, 300);
    }
  }
}

// ─── Detail Modal ─────────────────────────────────────────────────────────────
async function openDetailModal(rollNo, student) {
  // Mark modal open so popup stays hidden while it's showing
  openModals.add(rollNo);

  // Hide the popup temporarily
  const popup = activePopups.get(rollNo);
  if (popup) {
    popup.classList.remove('popup-visible');
    setTimeout(() => { popup.remove(); activePopups.delete(rollNo); }, 300);
  }

  const imgUrl = await resolveImageUrl(student.rollNo);
  const imgTag = imgUrl
    ? `<img src="${imgUrl}" alt="Profile" class="id-photo" />`
    : `<div class="id-photo" style="background:#222;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:0.6rem;color:#555;border-radius:50%;">NO PHOTO</div>`;

  const modal = document.createElement('div');
  modal.className = 'detail-modal hidden';
  modal.dataset.rollno = rollNo;

  modal.innerHTML = `
    <div class="detail-card">
      <button class="close-btn detail-close-btn" aria-label="Close">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
        <span>CLOSE</span>
      </button>

      <div class="card-header">
        <div class="photo-wrap">${imgTag}</div>
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
    </div>
  `;

  document.getElementById('scanner-root').appendChild(modal);

  // Animate in
  requestAnimationFrame(() => {
    modal.classList.remove('hidden');
    modal.classList.add('modal-visible');
  });

  modal.querySelector('.detail-close-btn').addEventListener('click', () => {
    modal.classList.remove('modal-visible');
    modal.classList.add('hidden');
    setTimeout(() => { modal.remove(); }, 400);
    openModals.delete(rollNo);
  });
}

// Helper: resolve actual image URL
async function resolveImageUrl(rollNo) {
  const fileName = imageMap[String(rollNo)];
  if (fileName) return `/Ar-Human-Scanner/images/${fileName}`;
  return null;
}

// ─── Status HUD ───────────────────────────────────────────────────────────────
function updateStatus(text, state) {
  statusText.innerText = text;
  statusDot.className  = `status-dot ${state}`;
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
init();
