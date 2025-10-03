(async function () {
async function loadJSON(path, fallback=null) {
try {
const r = await fetch(path);
if (!r.ok) throw 0;
return await r.json();
}
catch { return fallback; }
}

// Load your new status format
const status = await loadJSON(‘data/status/summary.json’, {});

// Load experimental data
const base = ‘data/latest’;
const forb3d = await loadJSON(`${base}/forbidden_points.json`, {accessible:[], forbidden:[]});
const fractal = await loadJSON(`${base}/fractal.json`, null);
const curvature = await loadJSON(`${base}/curvature.json`, null);
const hyst = await loadJSON(`${base}/hysteresis.json`, null);
const ring = await loadJSON(`${base}/ringing_map.json`, null);
const mf = await loadJSON(`${base}/multifreq.json`, null);

// HERO SECTION - Updated for new format
const phase = status.status || ‘Testing’;
const timestamp = status.last_updated || ‘—’;
const grade = status.evidence_grade || ‘—’;

document.getElementById(‘phase’).textContent = phase.charAt(0).toUpperCase() + phase.slice(1);
document.getElementById(‘runid’).textContent = timestamp.split(‘T’)[0] || ‘—’;

const gEl = document.getElementById(‘grade’);
gEl.textContent = grade;
// Color based on evidence grade
if (grade.startsWith(‘A’) || grade.startsWith(‘B’)) {
gEl.style.borderColor = ‘var(–ok)’;
} else if (grade.startsWith(‘C’)) {
gEl.style.borderColor = ‘var(–warn)’;
} else {
gEl.style.borderColor = ‘var(–bad)’;
}

document.getElementById(‘commit’).textContent = `${status.n_experiments || 0} experiments`;

// Update download link
const dl = document.getElementById(‘dl-summary’);
dl.href = ‘data/status/summary.json’;

// PROGRESS BARS - Based on experiments_completed
const completed = status.experiments_completed || [];
const progressMap = {
‘phase_sweep’: ‘bar-core’,
‘forbidden’: ‘bar-forbidden’,
‘fractal’: ‘bar-fractal’,
‘curvature’: ‘bar-curvature’,
‘multi_freq’: ‘bar-null’
};

Object.entries(progressMap).forEach(([expName, barId]) => {
const el = document.getElementById(barId);
if (el && completed.includes(expName)) {
el.style.width = ‘100%’;
}
});

// Forbidden 3D scatter (λ, β, A)
(function drawForbidden3D() {
const acc = forb3d.accessible || [];
const forb = forb3d.forbidden || [];
const toXYZ = arr => arr.length ? arr.reduce((acc,p)=>{
acc.x.push(p[0]);
acc.y.push(p[1]);
acc.z.push(p[2]);
return acc;
},{x:[],y:[],z:[]}) : {x:[],y:[],z:[]};

```
const A = toXYZ(acc), F = toXYZ(forb);
const trA = { 
  x:A.x, y:A.y, z:A.z, 
  mode:'markers', 
  type:'scatter3d', 
  name:'Accessible', 
  marker:{size:3, color:'#3bd67f', opacity:0.7} 
};
const trF = { 
  x:F.x, y:F.y, z:F.z, 
  mode:'markers', 
  type:'scatter3d', 
  name:'Forbidden', 
  marker:{size:4, color:'#ef476f'} 
};

Plotly.newPlot('forbidden3d', [trA, trF], {
  margin:{l:0, r:0, t:0, b:0}, 
  scene:{
    xaxis:{title:'λ'},
    yaxis:{title:'β'},
    zaxis:{title:'A'}
  }
}, {displayModeBar:false});
```

})();

// Fractal boundary fit
(function drawFractal() {
if (!fractal) return;
const xs = fractal.log_inv_eps, ys = fractal.log_counts;
Plotly.newPlot(‘fractalfit’, [
{x:xs, y:ys, mode:‘markers’, name:‘data’, marker:{size:6}},
{x:xs, y:xs.map(x => fractal.fit.intercept + fractal.fit.slope*x), mode:‘lines’, name:`fit: slope=${fractal.fit.slope.toFixed(2)}`}
], {
margin:{l:40, r:10, t:10, b:40},
xaxis:{title:‘log(1/ε)’},
yaxis:{title:‘log N(ε)’}
}, {displayModeBar:false});

```
const meta = document.getElementById('fractalmeta');
meta.textContent = `H ≈ ${fractal.H.toFixed(2)}  (95% CI: ${fractal.CI.map(v=>v.toFixed(2)).join(' – ')},  R²=${fractal.R2.toFixed(2)})`;
```

})();

// Curvature near boundaries
(function drawCurv() {
if (!curvature) return;
Plotly.newPlot(‘curvature’, [{
x: curvature.dist || [],
y: curvature.kappa || [],
mode:‘markers’,
type:‘scatter’,
name:‘κ vs distance’,
marker:{size:6, color:’#9ecbff’}
}], {
margin:{l:40, r:10, t:10, b:40},
xaxis:{title:‘distance to boundary’},
yaxis:{title:‘Ollivier–Ricci κ’}
}, {displayModeBar:false});

```
const hit = (curvature.coverage || 0)*100;
document.getElementById('curvemeta').textContent = `Coverage κ<-0.1: ${hit.toFixed(1)}%  |  Mean κ: ${(+curvature.mean).toFixed(3)}`;
```

})();

// Hysteresis
(function drawHyst() {
if (!hyst) return;
Plotly.newPlot(‘hyst’, [{
x: hyst.loop_x,
y: hyst.loop_y,
mode:‘lines’,
name:‘loop’
}], {
margin:{l:40, r:10, t:10, b:40},
xaxis:{title:‘drive’},
yaxis:{title:‘response’}
}, {displayModeBar:false});
})();

// Ringing boundary map
(function drawRing() {
if (!ring) return;
Plotly.newPlot(‘ringing’, [{
z: ring.phase_map,
type:‘heatmap’,
colorscale:‘Jet’
}], {
margin:{l:40, r:10, t:10, b:40},
xaxis:{title:‘α’},
yaxis:{title:‘η’}
}, {displayModeBar:false});
})();

// Multi-frequency
(function drawMF() {
if (!mf) return;
Plotly.newPlot(‘multifreq’, [{
x: mf.bands,
y: mf.lambda_star,
type:‘bar’,
name:‘λ* by band’
}], {
margin:{l:40, r:10, t:10, b:40},
yaxis:{title:‘λ*’}
}, {displayModeBar:false});
})();

// Validation badges (update when you add these to status)
const validation = status.validation || {};
document.getElementById(‘surrogate’).textContent = validation.surrogate_controls || ‘—’;
document.getElementById(‘mcc’).textContent = validation.mcc || ‘—’;
document.getElementById(‘nulls’).textContent = validation.null_models || ‘—’;
document.getElementById(‘repl’).textContent = validation.replication || ‘—’;
})();
