/**
 * charts.js — lightweight CSS-based chart utilities
 * No external dependencies.
 */

'use strict';

/**
 * Animate bar fills after they're in the DOM.
 * Call after renderEvidence() populates the bar chart.
 */
function animateBars() {
  const fills = document.querySelectorAll('.bar-fill');
  fills.forEach(fill => {
    const target = fill.style.width;
    fill.style.width = '0%';
    requestAnimationFrame(() => {
      setTimeout(() => { fill.style.width = target; }, 80);
    });
  });
}

// Trigger bar animation when Evidence tab is first opened
(function patchSwitchTab() {
  const original = window.switchTab;
  window.switchTab = function(tab) {
    original(tab);
    if (tab === 'evidence') {
      setTimeout(animateBars, 50);
    }
  };
})();
