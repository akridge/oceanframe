/** Small DOM and formatting helpers shared by the library modules. */

export const $ = id => document.getElementById(id);
export const $$ = sel => Array.from(document.querySelectorAll(sel));

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export function bytes(n) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = Number(n) || 0;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i += 1; }
  return `${value.toFixed(value < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

export const count = n => (Number(n) || 0).toLocaleString();

export function when(ts) {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleString();
}

/** Quality band name, kept in step with the server's facet buckets. */
export function band(quality) {
  if (quality >= 70) return 'excellent';
  if (quality >= 50) return 'good';
  if (quality >= 25) return 'fair';
  return 'poor';
}

let toastTimer = null;
export function notice(message, tone = 'info') {
  const bar = $('notice-bar');
  const text = $('notice-text');
  if (!bar || !text) return;
  text.textContent = message;
  bar.classList.remove('hidden');
  bar.dataset.tone = tone;
  if (toastTimer) clearTimeout(toastTimer);
  if (tone !== 'sticky') toastTimer = setTimeout(() => bar.classList.add('hidden'), 9000);
}

export function bindModal(id) {
  const modal = $(id);
  if (!modal) return { open() {}, close() {} };
  const close = () => modal.classList.add('hidden');
  modal.querySelectorAll('[data-close]').forEach(node => node.addEventListener('click', close));
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !modal.classList.contains('hidden')) close();
  });
  return { open: () => modal.classList.remove('hidden'), close, node: modal };
}
