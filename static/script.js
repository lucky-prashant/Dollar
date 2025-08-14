const btn = document.getElementById('analyzeBtn');
const statusText = document.getElementById('statusText');
const cards = document.getElementById('cards');

function pairColor(status) {
  if (status === 'TRADE') return 'ok';
  if (status === 'RISKY') return 'warn';
  return 'muted';
}

function candleEmoji(signal) {
  if (signal === 'CALL') return '🟩';
  if (signal === 'PUT') return '🟥';
  return '⬜️';
}

function renderResults(results) {
  cards.innerHTML = '';
  if (!results || !Array.isArray(results) || results.length === 0) {
    cards.innerHTML = `<div class="card muted"><div class="title">No Results</div><p>Backend returned no results.</p></div>`;
    return;
  }

  results.forEach(r => {
    const sig = r.signal || 'SIDEWAYS';
    const status = r.status || 'NO TRADE';
    const acc = (r.accuracy != null) ? `${r.accuracy}%` : '—';
    const cwrv = r.cwrv || 'No';
    const conf = (r.cwrv_conf != null) ? `${r.cwrv_conf}%` : '0%';
    const wave = r.wave || 'unknown';
    const why = r.why || '';

    const card = document.createElement('div');
    card.className = `card ${pairColor(status)}`;
    card.innerHTML = `
      <div class="title">${r.pair || 'PAIR'} <span class="sig">${candleEmoji(sig)} ${sig}</span></div>
      <div class="row">
        <div><strong>Status:</strong> ${status}</div>
        <div><strong>Accuracy:</strong> ${acc}</div>
      </div>
      <div class="row">
        <div><strong>CWRV 1-2-3:</strong> ${cwrv}</div>
        <div><strong>Confidence:</strong> ${conf}</div>
      </div>
      <div class="row">
        <div><strong>Elliott:</strong> ${wave}</div>
      </div>
      <details class="why">
        <summary>Details</summary>
        <pre>${why}</pre>
      </details>
    `;
    cards.appendChild(card);
  });
}

async function analyze() {
  try {
    btn.disabled = true;
    btn.textContent = 'Analyzing…';
    statusText.textContent = 'Fetching data and computing signals…';

    const res = await fetch('/analyze', { method: 'GET' });
    const ct = res.headers.get('content-type') || '';
    if (!res.ok) {
      const msg = await res.text();
      statusText.textContent = `Error ${res.status}: ${msg}`;
      btn.disabled = false;
      btn.textContent = 'Analyze';
      return;
    }
    if (!ct.includes('application/json')) {
      const text = await res.text();
      statusText.textContent = 'Unexpected response (not JSON). See Details card.';
      renderResults([{pair:'All', signal:'SIDEWAYS', status:'NO TRADE', accuracy:100, cwrv:'No', cwrv_conf:0, wave:'unknown', why: text.slice(0, 2000)}]);
      btn.disabled = false;
      btn.textContent = 'Analyze';
      return;
    }

    const data = await res.json();
    if (data.error) {
      statusText.textContent = `Server error: ${data.error}`;
    } else {
      statusText.textContent = 'Analysis complete.';
    }
    renderResults(data.results);

  } catch (err) {
    console.error(err);
    statusText.textContent = `JS error: ${err.message}`;
    renderResults([{pair:'All', signal:'SIDEWAYS', status:'NO TRADE', accuracy:100, cwrv:'No', cwrv_conf:0, wave:'unknown', why: String(err)}]);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Analyze';
  }
}

btn.addEventListener('click', analyze);

// Auto-run on load once
window.addEventListener('load', () => {
  analyze();
});