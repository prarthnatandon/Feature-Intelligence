/**
 * visualization.js — D3.js v7 interactive skill dependency graph
 *
 * Node color  → lesson category (8 categories)
 * Node size   → retention score (larger = higher retention)
 * Edge        → skill dependency (arrow from prerequisite → skill)
 * Click       → show skill detail tooltip
 * Scroll      → zoom
 * Drag        → pan
 */

const CATEGORY_COLORS = {
  core_vocabulary:      '#1cb0f6',  // blue
  thematic_vocabulary:  '#2bd9fe',  // cyan
  grammar_introduction: '#ce82ff',  // purple
  grammar_practice:     '#a560f0',  // dark purple
  pronunciation_phonetics: '#ff9600', // orange
  cultural_context:     '#ff6b6b',  // red
  review_consolidation: '#58cc02',  // green
  compound_mixed:       '#ffd900',  // yellow
};

const CATEGORY_LABELS = {
  core_vocabulary:       'Core Vocabulary',
  thematic_vocabulary:   'Thematic Vocabulary',
  grammar_introduction:  'Grammar Introduction',
  grammar_practice:      'Grammar Practice',
  pronunciation_phonetics: 'Pronunciation',
  cultural_context:      'Cultural Context',
  review_consolidation:  'Review',
  compound_mixed:        'Compound / Mixed',
};

let graphInstance = null;

function renderGraph(graphData) {
  const section = document.getElementById('graph-section');
  section.style.display = '';

  const container = document.getElementById('graph-container');
  const tooltip = document.getElementById('graph-tooltip');
  const svgEl = document.getElementById('graph-svg');

  const W = container.clientWidth || 700;
  const H = container.clientHeight || 480;

  // Clear previous render
  d3.select(svgEl).selectAll('*').remove();

  const svg = d3.select(svgEl)
    .attr('width', W)
    .attr('height', H);

  // Defs — arrowhead marker
  const defs = svg.append('defs');
  defs.append('marker')
    .attr('id', 'arrow')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 20)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', '#444866');

  // Zoom + pan
  const g = svg.append('g');
  const zoom = d3.zoom()
    .scaleExtent([0.2, 3])
    .on('zoom', (e) => g.attr('transform', e.transform));
  svg.call(zoom);

  // Initial zoom to fit
  const initialScale = 0.75;
  svg.call(zoom.transform, d3.zoomIdentity
    .translate(W / 2, H / 2)
    .scale(initialScale)
    .translate(-W / 2, -H / 2));

  // Node radius based on retention score
  const rScale = d3.scaleSqrt()
    .domain([0, 1])
    .range([7, 18]);

  // --- Build force simulation ---
  const nodes = graphData.nodes.map(d => ({ ...d }));
  const links = graphData.links.map(d => ({ ...d }));

  // Map string IDs to node objects for link resolution
  const nodeById = Object.fromEntries(nodes.map(n => [n.id, n]));
  const resolvedLinks = links
    .filter(l => nodeById[l.source] && nodeById[l.target])
    .map(l => ({ ...l, source: nodeById[l.source], target: nodeById[l.target] }));

  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(resolvedLinks)
      .id(d => d.id)
      .distance(70)
      .strength(0.6))
    .force('charge', d3.forceManyBody().strength(-180))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .force('collision', d3.forceCollide().radius(d => rScale(d.retention_score || 0.65) + 4))
    .force('y', d3.forceY(d => (d.coords_y || 0) * 55 + H * 0.15).strength(0.25))
    .force('x', d3.forceX(d => (d.coords_x || 0) * 90 + W * 0.5).strength(0.2));

  // --- Links ---
  const link = g.append('g').attr('class', 'links')
    .selectAll('line')
    .data(resolvedLinks)
    .join('line')
    .attr('stroke', '#2e3350')
    .attr('stroke-width', d => (d.strength || 1) * 1.2)
    .attr('marker-end', 'url(#arrow)')
    .attr('opacity', 0.6);

  // --- Nodes ---
  const node = g.append('g').attr('class', 'nodes')
    .selectAll('g')
    .data(nodes)
    .join('g')
    .attr('class', 'node')
    .style('cursor', 'pointer')
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end', (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

  // Circle
  node.append('circle')
    .attr('r', d => rScale(d.retention_score || 0.65))
    .attr('fill', d => CATEGORY_COLORS[d.category] || '#6b7280')
    .attr('stroke', '#0f1117')
    .attr('stroke-width', 1.5)
    .attr('opacity', 0.9);

  // Label (only for larger nodes to avoid clutter)
  node.filter(d => (d.retention_score || 0.65) > 0.72)
    .append('text')
    .attr('text-anchor', 'middle')
    .attr('dy', '0.35em')
    .attr('font-size', '8px')
    .attr('fill', '#fff')
    .attr('pointer-events', 'none')
    .text(d => d.title.split(' ')[0].slice(0, 7));

  // --- Tooltip on hover ---
  node.on('mouseenter', (e, d) => {
    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left + 12;
    const y = e.clientY - rect.top + 12;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
    tooltip.style.opacity = '1';
    tooltip.innerHTML = `
      <div style="font-weight:700;margin-bottom:4px;color:${CATEGORY_COLORS[d.category] || '#fff'}">${d.title}</div>
      <div style="color:var(--text-dim);font-size:11px;margin-bottom:6px">${CATEGORY_LABELS[d.category] || d.category}</div>
      <div><b>Retention:</b> ${((d.retention_score || 0.65) * 100).toFixed(0)}%</div>
      <div><b>Vocab load:</b> ${d.vocab_load || '?'} words</div>
      ${d.grammar_concepts?.length ? `<div><b>Grammar:</b> ${d.grammar_concepts.slice(0, 2).join(', ')}</div>` : ''}
    `;
  })
  .on('mouseleave', () => { tooltip.style.opacity = '0'; });

  // --- Tick ---
  simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);

    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  graphInstance = { simulation, nodes, links: resolvedLinks };

  // Render legend
  renderLegend(graphData.nodes);
}


function renderLegend(nodes) {
  const legendEl = document.getElementById('graph-legend');
  const categories = [...new Set(nodes.map(n => n.category))];

  legendEl.innerHTML = categories.map(cat => `
    <div class="legend-item">
      <div class="legend-dot" style="background:${CATEGORY_COLORS[cat] || '#6b7280'}"></div>
      ${CATEGORY_LABELS[cat] || cat}
    </div>
  `).join('');
}
