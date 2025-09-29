// Helper: optional fetch (returns null on 404)
async function ftry(path) {
  try {
    const r = await fetch(path, {cache:'no-store'});
    if (!r.ok) return null;
    const ct = r.headers.get('content-type') || '';
    if (ct.includes('application/json')) return await r.json();
    return await r.text();
  } catch { return null; }
}

function bind(id, map) {
  const el = document.getElementById(id);
  if (!el) return;
  Object.entries(map).forEach(([key, val]) => {
    const t = el.querySelector(`[data-bind="${key}"]`);
    if (t) t.textContent = val ?? 'N/A';
  });
}
function fillPre(id, key, text) {
  const el = document.getElementById(id);
  if (!el) return;
  const pre = el.querySelector(`[data-pre="${key}"]`);
  if (pre && text) pre.textContent = typeof text === 'string' ? text : JSON.stringify(text, null, 2);
}
function putImage(id, key, pathCandidates=[]) {
  const el = document.getElementById(id);
  if (!el) return;
  const box = el.querySelector(`.media`);
  (async () => {
    for (const p of pathCandidates) {
      try {
        const r = await fetch(p, {cache:'no-store'});
        if (r.ok) {
          const img = document.createElement('img');
          img.src = p;
          img.alt = key;
          img.loading = 'lazy';
          box.innerHTML = '';
          box.appendChild(img);
          return;
        }
      } catch {}
    }
    box.textContent = 'No figure yet.';
  })();
}

(async () => {
  // Forbidden regions (expected JSON location if produced by detector)
  const forb = await ftry('./results/forbidden/summary.json') ||
               await ftry('./results/topo_test/forbidden_summary.json');
  if (forb) {
    bind('forbidden-card', {
      forbiddenPct: (forb.forbidden_pct ?? forb.percent_forbidden ?? 0).toFixed(2) + '%',
      largestComp: forb.largest_component ?? '—',
      runs: forb.runs ?? forb.n_runs ?? '—'
    });
    fillPre('forbidden-card','forbiddenJson',forb);
  }
  putImage('forbidden-card','forbiddenPlot',[
    './figures/forbidden_map.png',
    './figures/topo_test/forbidden_map.png'
  ]);

  // Fractal boundary (box-counting)
  const fract = await ftry('./results/topo_test/fractal_dim.json');
  if (fract) {
    bind('fractal-card', {
      H: (fract.H_estimate ?? fract.H ?? 'N/A'),
      r2: (fract.r2 ?? 'N/A'),
      ci: fract.ci ? `[${fract.ci[0]}, ${fract.ci[1]}]` : 'N/A'
    });
    fillPre('fractal-card','fractalJson',fract);
  }
  putImage('fractal-card','fractalPlot',[
    './figures/fractal_fit.png',
    './figures/topo_test/fractal_fit.png'
  ]);

  // Ricci curvature
  const ricci = await ftry('./results/ricci/summary.json');
  if (ricci) {
    bind('ricci-card', {
      avgKappa: (ricci.avg_kappa ?? ricci.avg ?? 'N/A'),
      coverage: ricci.coverage ? ricci.coverage + '%' : 'N/A'
    });
    fillPre('ricci-card','ricciJson',ricci);
  }
  putImage('ricci-card','ricciPlot',[
    './figures/ricci_map.png',
    './figures/ci/ricci_map.png'
  ]);

  // Mapper/TDA
  const mapper = await ftry('./results/mapper/report.json');
  if (mapper) {
    bind('mapper-card', {
      betti: mapper.betti ? `[${mapper.betti.join(', ')}]` : 'N/A',
      embeddings: 'saved'
    });
    fillPre('mapper-card','mapperJson',mapper);
  }
  putImage('mapper-card','mapperPlot',[
    './figures/mapper_persistence.png',
    './figures/ci/mapper_persistence.png'
  ]);

  // Figure gallery (best-effort load of common outputs)
  const gallery = document.getElementById('figs');
  const candidates = [
    'figures/forbidden_map.png',
    'figures/topo_test/forbidden_map.png',
    'figures/fractal_fit.png',
    'figures/topo_test/fractal_fit.png',
    'figures/mapper_persistence.png',
    'figures/ci/mapper_persistence.png',
    'figures/ring_holonomy.png'
  ];
  for (const path of candidates) {
    try {
      const r = await fetch('./' + path, {cache:'no-store'});
      if (r.ok) {
        const img = document.createElement('img');
        img.src = './' + path;
        img.alt = path.split('/').pop();
        img.loading = 'lazy';
        gallery.appendChild(img);
      }
    } catch {}
  }

  // Commit hash (if Pages injects it; otherwise left as …)
  const hash = (window.__RG_BUILD_HASH__) || '';
  if (hash) document.getElementById('build-hash').textContent = hash.slice(0,7);
})();
