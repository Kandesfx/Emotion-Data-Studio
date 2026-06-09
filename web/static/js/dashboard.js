// dashboard.js — Dashboard page logic

const EMOTION_COLORS = {
  happy: '#f59e0b',
  sad: '#3b82f6',
  angry: '#ef4444',
  fear: '#a855f7',
  surprise: '#06b6d4',
  disgust: '#22c55e',
  neutral: '#6c63ff',
};

const EMOTION_ORDER = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral'];

async function loadStats() {
  try {
    const resp = await fetch('/api/stats');
    if (!resp.ok) return;
    const data = await resp.json();

    // Overall progress
    const total = data.emotion_target_total || 3000;
    const approved = data.approved || 0;
    const pct = total > 0 ? Math.min(100, (approved / total * 100)) : 0;
    document.getElementById('approvedCount').textContent = approved.toLocaleString();
    document.getElementById('targetTotal').textContent = total.toLocaleString();
    document.getElementById('approvalRate').textContent = (data.approval_rate || 0).toFixed(1);
    document.getElementById('overallProgress').style.width = pct + '%';

    // Stats
    document.getElementById('totalVideos').textContent = (data.total_videos || 0).toLocaleString();
    document.getElementById('totalClips').textContent = (data.total_clips || 0).toLocaleString();
    document.getElementById('approvedStat').textContent = approved.toLocaleString();
    document.getElementById('rejectedStat').textContent = (data.rejected || 0).toLocaleString();
    document.getElementById('pendingReview').textContent = (data.pending_review || 0).toLocaleString();
    document.getElementById('autoApproved').textContent = (data.auto_approved || 0).toLocaleString();
    document.getElementById('humanReviewed').textContent = (data.human_reviewed || 0).toLocaleString();

    // Queue
    const q = data.queue || {};
    document.getElementById('queuePending').textContent = (q.pending || 0).toLocaleString();
    document.getElementById('queueRunning').textContent = (q.running || 0).toLocaleString();
    document.getElementById('queuePendingVal').textContent = (q.pending || 0);
    document.getElementById('queueRunningVal').textContent = (q.running || 0);
    document.getElementById('queueCompleted').textContent = (q.completed || 0);
    document.getElementById('queueCount').textContent = (q.pending || 0);

    // GPU info
    const healthResp = await fetch('/health');
    const health = await healthResp.json();
    document.getElementById('gpuInfo').textContent =
      health.gpu && health.gpu !== 'Unknown'
        ? health.gpu + ' ' + health.gpu_memory_gb + 'GB'
        : 'Không có GPU';

    // Emotion quota bars
    renderEmotionQuota(data.emotion_quota || {});

  } catch (err) {
    console.error('Failed to load stats:', err);
  }
}

function renderEmotionQuota(quota) {
  const container = document.getElementById('emotionQuotaList');
  if (!container) return;
  container.innerHTML = '';

  for (const emotion of EMOTION_ORDER) {
    const item = quota[emotion] || { count: 0, target: emotion === 'neutral' ? 600 : 300 };
    const count = item.count || 0;
    const target = item.target || 300;
    const pct = target > 0 ? Math.min(100, count / target * 100) : 0;
    const color = EMOTION_COLORS[emotion] || '#6c63ff';

    let status = '✅';
    if (pct >= 100) status = '✅';
    else if (pct >= 80) status = '⚠️';
    else if (pct >= 50) status = '⚠️';
    else status = '❌';

    const row = document.createElement('div');
    row.className = 'emotion-row';
    row.innerHTML = `
      <span class="emotion-name">${emotion}</span>
      <div class="emotion-bar-wrap">
        <div class="emotion-bar" style="width:${pct}%;background:${color};"></div>
      </div>
      <span class="emotion-count">${count}/${target} ${status}</span>
      <span class="emotion-status">${pct >= 100 ? '✅' : pct >= 80 ? '⚠️' : '❌'}</span>
    `;
    container.appendChild(row);
  }
}

// Load on page load
loadStats();
setInterval(loadStats, 15000); // refresh every 15s
