// harvest.js — Harvest page logic

let queueRefreshTimer = null;

function addLog(msg, type = '') {
  const logEl = document.getElementById('pipelineLog');
  const div = document.createElement('div');
  div.className = 'log-line' + (type ? ' ' + type : '');
  const ts = new Date().toLocaleTimeString();
  div.textContent = `[${ts}] ${msg}`;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
}

async function loadQueue() {
  try {
    const resp = await fetch('/api/queue');
    const data = await resp.json();
    const items = data.items || [];
    document.getElementById('queueTotal').textContent = items.length;
    renderQueueTable(items);
  } catch (err) {
    console.error('Failed to load queue:', err);
  }
}

function renderQueueTable(items) {
  const tbody = document.getElementById('queueTableBody');
  if (!items.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-muted);padding:20px;">No videos in queue</td></tr>';
    return;
  }
  tbody.innerHTML = '';
  items.forEach((item, i) => {
    const statusClass = item.status === 'done' ? 'badge-success'
      : item.status === 'running' ? 'badge-warning'
      : item.status === 'error' ? 'badge-error'
      : 'badge-muted';
    const statusLabel = item.status === 'done' ? '✅ Done'
      : item.status === 'running' ? '🔄 Running'
      : item.status === 'error' ? '❌ Error'
      : '⏳ Pending';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td style="max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${item.title || ''}">${item.title || '—'}</td>
      <td><span class="badge ${statusClass}">${statusLabel}</span></td>
      <td>${item.created_at ? new Date(item.created_at).toLocaleString() : '—'}</td>
      <td>${item.started_at ? new Date(item.started_at).toLocaleString() : '—'}</td>
      <td style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--error);" title="${item.error || ''}">${item.error || '—'}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function importUrls(startProcessing = false) {
  const urlText = document.getElementById('urlInput').value.trim();
  if (!urlText) {
    addLog('No URLs provided', 'error');
    return;
  }
  const urls = urlText.split('\n').map(u => u.trim()).filter(Boolean);
  if (!urls.length) {
    addLog('No valid URLs found', 'error');
    return;
  }

  // Target emotions
  const targetEmotions = [];
  document.querySelectorAll('#targetEmotionGroup input[type="checkbox"]:checked').forEach(cb => {
    if (cb.value !== 'any') targetEmotions.push(cb.value);
  });

  document.getElementById('importStatus').textContent = `Importing ${urls.length} URLs...`;

  try {
    const resp = await fetch('/api/harvest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        urls,
        target_emotions: targetEmotions,
        start_processing: startProcessing,
      }),
    });
    const result = await resp.json();
    const added = result.added || [];
    const errors = result.errors || [];

    addLog(`Imported ${added.length} video(s)`, added.length ? 'success' : '');
    errors.forEach(e => addLog(`Error: ${e.url} — ${e.error}`, 'error'));

    if (added.length) {
      document.getElementById('urlInput').value = '';
      addLog('URLs imported successfully!', 'success');
    }
  } catch (err) {
    addLog('Import failed: ' + err.message, 'error');
  } finally {
    document.getElementById('importStatus').textContent = '';
    loadQueue();
  }
}

async function pauseQueue() {
  try {
    await fetch('/api/queue/pause', { method: 'POST' });
    addLog('Queue paused', 'warning');
    loadQueue();
  } catch (err) {
    addLog('Pause failed: ' + err.message, 'error');
  }
}

async function resumeQueue() {
  try {
    await fetch('/api/queue/resume', { method: 'POST' });
    addLog('Queue resumed', 'success');
    loadQueue();
  } catch (err) {
    addLog('Resume failed: ' + err.message, 'error');
  }
}

async function startPipeline() {
  try {
    await fetch('/api/pipeline/start', { method: 'POST' });
    addLog('Pipeline started!', 'success');
    // Poll queue every 5s
    if (!queueRefreshTimer) {
      queueRefreshTimer = setInterval(() => {
        loadQueue().then(() => {
          // auto-stop polling when no running items
          fetch('/api/queue').then(r => r.json()).then(data => {
            const running = (data.items || []).filter(i => i.status === 'running').length;
            if (running === 0 && queueRefreshTimer) {
              clearInterval(queueRefreshTimer);
              queueRefreshTimer = null;
              addLog('Pipeline idle', '');
            }
          });
        });
      }, 5000);
    }
  } catch (err) {
    addLog('Start pipeline failed: ' + err.message, 'error');
  }
}

// Event listeners
document.getElementById('importBtn').addEventListener('click', () => importUrls(false));
document.getElementById('importAndStartBtn').addEventListener('click', () => importUrls(true));
document.getElementById('pauseQueueBtn').addEventListener('click', pauseQueue);
document.getElementById('resumeQueueBtn').addEventListener('click', resumeQueue);
document.getElementById('startPipelineBtn').addEventListener('click', startPipeline);

// Checkbox group toggle
document.querySelectorAll('#targetEmotionGroup .checkbox-item').forEach(item => {
  item.addEventListener('click', () => {
    const cb = item.querySelector('input[type="checkbox"]');
    cb.checked = !cb.checked;
    item.classList.toggle('checked', cb.checked);

    // Uncheck "Any" if specific emotion selected
    const anyCb = document.querySelector('#targetEmotionGroup input[value="any"]');
    if (cb.value !== 'any' && cb.checked) {
      anyCb.checked = false;
      document.querySelector('#targetEmotionGroup .checkbox-item input[value="any"]')
        .closest('.checkbox-item').classList.remove('checked');
    }
  });
});

// Initialize
loadQueue();
setInterval(loadQueue, 10000);
addLog('Harvest page ready', '');
