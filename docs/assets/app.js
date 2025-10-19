// Minimal client-side renderer for overview.json
(async function () {
  async function getJSON(path) {
    try {
      const response = await fetch(path, { cache: "no-cache" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.warn("Failed to load", path, error);
      return null;
    }
  }

  function h(tag, attrs = {}, inner = "") {
    const el = document.createElement(tag);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
    if (typeof inner === "string") {
      el.innerHTML = inner;
    } else if (inner instanceof Node) {
      el.appendChild(inner);
    } else if (Array.isArray(inner)) {
      inner.forEach((child) => el.appendChild(child));
    }
    return el;
  }

  const root = document.getElementById("app");
  if (!root) return;

  const overview = await getJSON("./data/overview.json");
  if (!overview) {
    root.appendChild(h("div", { class: "card" }, "<h3>Dashboard</h3><p>No overview.json yet.</p>"));
    return;
  }

  root.appendChild(
    h(
      "div",
      { class: "header" },
      `<h1>Resonance Geometry — Unified Dashboard</h1>
       <p>Last update: ${overview.timestamp}</p>`
    )
  );

  const phase = overview.phase || {};
  root.appendChild(
    h(
      "div",
      { class: "bar" },
      `<span>Current: <b>${phase.current || "—"}</b></span>
       <span>Next: <b>${phase.next || "—"}</b></span>`
    )
  );

  try {
    const fragmentResponse = await fetch("./assets/fragment.html", { cache: "no-cache" });
    if (fragmentResponse.ok) {
      const fragmentHtml = await fragmentResponse.text();
      const wrapper = h("div", { class: "cards" });
      wrapper.innerHTML = fragmentHtml;
      root.appendChild(wrapper);
    }
  } catch (error) {
    console.warn("Failed to load fragment", error);
  }

  const boundaryPoints = overview.experiments?.jacobian?.boundary_points || [];
  if (Array.isArray(boundaryPoints) && boundaryPoints.length) {
    const table = h("table", { class: "table" });
    const rows = boundaryPoints.slice(0, 10).map((point) => {
      const alpha = typeof point.alpha === "number" ? point.alpha.toFixed(3) : "—";
      const k0 = typeof point.K0 === "number" ? point.K0.toFixed(2) : "—";
      const beta = typeof point.beta_c === "number" ? point.beta_c.toFixed(4) : "—";
      const method = point.method ?? "—";
      return `<tr><td>${alpha}</td><td>${k0}</td><td>${beta}</td><td>${method}</td></tr>`;
    });
    table.innerHTML = `
      <thead><tr><th>α</th><th>K₀</th><th>β_c</th><th>method</th></tr></thead>
      <tbody>${rows.join("")}</tbody>`;

    root.appendChild(h("div", { class: "card" }, `<h3>Boundary (sample)</h3>`));
    root.appendChild(table);
  }
})();
