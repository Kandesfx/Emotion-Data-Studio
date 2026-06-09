// review.js — Review page logic

let clips = [];
let currentIndex = -1;
let currentClip = null;
let selectedEmotion = null;
let currentSentiment = 0.0;

const EMOTIONS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral'];

async function loadClips() {
  try {
    const resp = await fetch('/api/clips?status=needs_review&limit=100');
    const data = await resp.json();
    clips = data.items || [];
    renderClipList();
    if (clips.length > 0) selectClip(0);
    document.getElementById('pendingCount').textContent = clips.length;
  } catch (err) {
    console.error('Failed to load clips:', err);
  }
}

function renderClipList() {
  const container = document.getElementById('clipList');
  container.innerHTML = '';
  clips.forEach((clip, i) => {
    const div = document.createElement('div');
    div.className = 'clip-list-item';
    div.style.cssText = `
      padding: 8px 10px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 11px;
      color: var(--text-muted);
      border: 1px solid transparent;
      transition: all 0.15s;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    `;
    div.textContent = `#${clip.id} — ${clip.predicted_emotion || '?'} — ${clip.status || '?'}`;
    div.title = div.textContent;
    div.addEventListener('mouseover', () => {
      div.style.background = 'var(--surface-2)';
      div.style.color = 'var(--text)';
    });
    div.addEventListener('mouseout', () => {
      div.style.background = '';
      div.style.color = '';
    });
    div.addEventListener('click', () => selectClip(i));
    container.appendChild(div);
  });
}

async function selectClip(index) {
  currentIndex = index;
  currentClip = clips[index];
  if (!currentClip) return;

  // Highlight selected in list
  const items = document.querySelectorAll('#clipList > div');
  items.forEach((el, i) => {
    el.style.background = i === index ? 'rgba(108,99,255,0.15)' : '';
    el.style.borderColor = i === index ? 'var(--primary)' : 'transparent';
    el.style.color = i === index ? 'var(--primary)' : '';
  });

  // Load clip data
  document.getElementById('mId').textContent = currentClip.id || '--';
  document.getElementById('mVideo').textContent = currentClip.video_id || '--';
  document.getElementById('mDuration').textContent =
    currentClip.duration ? currentClip.duration.toFixed(1) + 's' : '--';
  document.getElementById('mFace').textContent =
    currentClip.face_quality != null ? currentClip.face_quality.toFixed(2) : '--';
  document.getElementById('mSnr').textContent =
    currentClip.snr_db != null ? currentClip.snr_db.toFixed(1) + ' dB' : '--';
  document.getElementById('mQuality').textContent =
    currentClip.quality_score != null ? currentClip.quality_score.toFixed(2) : '--';
  document.getElementById('clipTranscript').textContent =
    currentClip.transcript || '--';

  // AI prediction
  const emotion = currentClip.predicted_emotion || 'unknown';
  document.getElementById('aiEmotion').textContent = emotion;
  document.getElementById('aiConfidence').textContent =
    currentClip.confidence != null ? (currentClip.confidence * 100).toFixed(0) + '%' : '--';
  document.getElementById('aiAgreement').textContent = currentClip.agreement || '--';
  document.getElementById('aiIncongruity').textContent =
    currentClip.has_incongruity ? '⚠️ Có' : '✅ Không';

  // Per-model scores
  const pms = currentClip.per_model_scores || {};
  const perModelEl = document.getElementById('perModelScores');
  perModelEl.innerHTML = '';
  const modelLabels = {
    visual: '🎭 Visual',
    audio: '🔊 Audio',
    text: '📝 Text',
  };
  for (const [model, scores] of Object.entries(pms)) {
    if (!scores || typeof scores !== 'object') continue;
    const top = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
    if (!top) continue;
    const div = document.createElement('div');
    div.className = 'badge badge-neutral';
    div.style.cssText = 'font-size:11px;padding:4px 10px;';
    div.textContent = `${modelLabels[model] || model}: ${top[0]} (${(top[1]*100).toFixed(0)}%)`;
    perModelEl.appendChild(div);
  }

  // Human labels
  selectedEmotion = currentClip.user_emotion || null;
  currentSentiment = currentClip.user_sentiment || 0.0;
  document.getElementById('sentimentSlider').value = currentSentiment;
  document.getElementById('sentimentValue').textContent = currentSentiment.toFixed(1);
  document.getElementById('reviewNotes').value = currentClip.review_notes || '';

  // Emotion picker
  document.querySelectorAll('.emotion-btn').forEach(btn => {
    btn.classList.toggle('selected', btn.dataset.emotion === selectedEmotion);
  });

  // Video
  const videoEl = document.getElementById('clipVideo');
  if (currentClip.clip_path) {
    videoEl.src = `/api/clips/${currentClip.id}/video`;
    videoEl.load();
  } else {
    videoEl.src = '';
  }
}

async function saveCurrentClip(action) {
  if (!currentClip) return;
  try {
    const payload = {
      review_type: 'human',
      review_notes: document.getElementById('reviewNotes').value,
    };

    if (action === 'approve') {
      payload.status = 'approved';
      payload.user_emotion = selectedEmotion || currentClip.predicted_emotion;
      payload.user_sentiment = parseFloat(document.getElementById('sentimentValue').textContent) || 0;
    } else if (action === 'reject') {
      payload.status = 'rejected';
    }

    const resp = await fetch(`/api/clips/${currentClip.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (resp.ok) {
      // Remove from list and move to next
      clips.splice(currentIndex, 1);
      renderClipList();
      const nextIndex = Math.min(currentIndex, clips.length - 1);
      if (clips.length > 0) {
        selectClip(nextIndex >= 0 ? nextIndex : 0);
      } else {
        clearPreview();
      }
      document.getElementById('pendingCount').textContent = clips.length;
    }
  } catch (err) {
    console.error('Save failed:', err);
  }
}

// Event listeners
document.getElementById('sentimentSlider').addEventListener('input', (e) => {
  currentSentiment = parseFloat(e.target.value);
  document.getElementById('sentimentValue').textContent = currentSentiment.toFixed(1);
});

document.querySelectorAll('.emotion-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    selectedEmotion = btn.dataset.emotion;
    document.querySelectorAll('.emotion-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
  });
});

document.getElementById('btnApprove').addEventListener('click', () => saveCurrentClip('approve'));
document.getElementById('btnReject').addEventListener('click', () => saveCurrentClip('reject'));
document.getElementById('btnSkip').addEventListener('click', () => {
  const nextIndex = Math.min(currentIndex + 1, clips.length - 1);
  if (clips.length > 0) selectClip(nextIndex >= 0 ? nextIndex : 0);
});
document.getElementById('btnSave').addEventListener('click', () => saveCurrentClip('save'));

function clearPreview() {
  currentClip = null;
  ['mId','mVideo','mDuration','mFace','mSnr','mQuality',
   'clipTranscript','aiEmotion','aiConfidence','aiAgreement','aiIncongruity'].forEach(id => {
    document.getElementById(id).textContent = '--';
  });
  document.getElementById('clipVideo').src = '';
  document.getElementById('perModelScores').innerHTML = '';
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (!currentClip) return;
  const EMOTION_KEYS = {'1':'happy','2':'sad','3':'angry','4':'fear','5':'surprise','6':'disgust','7':'neutral'};
  if (EMOTION_KEYS[e.key]) {
    const emotion = EMOTION_KEYS[e.key];
    selectedEmotion = emotion;
    document.querySelectorAll('.emotion-btn').forEach(b => {
      b.classList.toggle('selected', b.dataset.emotion === emotion);
    });
  }
  if (e.key === 'a' || e.key === 'A') saveCurrentClip('approve');
  if (e.key === 'r' || e.key === 'R') saveCurrentClip('reject');
  if (e.key === 'ArrowRight') {
    const next = Math.min(currentIndex + 1, clips.length - 1);
    if (clips.length > 0) selectClip(next >= 0 ? next : 0);
  }
  if (e.key === 'ArrowLeft') {
    const prev = Math.max(currentIndex - 1, 0);
    if (clips.length > 0) selectClip(prev);
  }
});

// Load
loadClips();
setInterval(loadClips, 20000);
