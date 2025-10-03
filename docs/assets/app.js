(async function () {
  async function loadJSON(path, fallback=null) {
    try { 
      const r = await fetch(path); 
      if (!r.ok) throw 0; 
      return await r.json(); 
    }
    catch { return fallback; }
  }

  const status = await loadJSON('data/status/summary.json', {});
  const base = 'data/latest';
  const forb3d = await loadJSON(`${base}/forbidden_points.json`, {accessible:[], forbidden:[]});

  document.getElementById('phase').textContent = (status.status || 'Testing').charAt(0).toUpperCase() + (status.status || 'Testing').slice(1);
  document.getElementById('runid').textContent = status.last_updated?.split('T')[0] || '—';
  
  const gEl = document.getElementById('grade');
  gEl.textContent = status.evidence_grade || '—';
  gEl.style.borderColor = (status.evidence_grade || '').startsWith('B') ? 'var(--ok)' : (status.evidence_grade || '').startsWith('C') ? 'var(--warn)' : 'var(--bad)';
  
  document.getElementById('commit').textContent = `${status.n_experiments || 0} experiments`;
  document.getElementById('dl-summary').href = 'data/status/summary.json';

  const completed = status.experiments_completed || [];
  if (completed.includes('phase_sweep')) document.getElementById('bar-core').style.width = '100%';
  if (completed.includes('forbidden')) document.getElementById('bar-forbidden').style.width = '100%';

  const validation = status.validation || {};
  document.getElementById('surrogate').textContent = validation.surrogate_controls || '—';
  document.getElementById('nulls').textContent = validation.null_models || '—';
  document.getElementById('repl').textContent = validation.replication || '—';
})();
