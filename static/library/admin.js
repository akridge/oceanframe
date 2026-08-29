/** Sources, indexing, embedding and model status drawer. */
import { $, el, clear, count, when, bindModal, notice, bytes } from './ui.js';
import { api } from './api.js';
import { watchJob } from './jobs.js';

export function createAdmin({ onRefresh }) {
  const modal = bindModal('admin');

  function render(status) {
    const sources = clear($('source-list'));
    if (!status.sources.length) {
      sources.append(el('div', { class: 'muted-row', text: 'No sources yet.' }));
    }
    for (const source of status.sources) {
      sources.append(el('div', { class: 'list-row' }, [
        el('div', { class: 'grow' }, [
          el('div', { text: source.label || source.root }),
          el('div', { class: 'sub', text: `${source.kind} · ${source.root} · ${count(source.assets)} assets · scanned ${when(source.scanned_at)}` }),
        ]),
        el('button', {
          type: 'button', class: 'btn btn-ghost btn-sm', text: 'Re-index',
          onclick: () => runIndex(source.root, false),
        }),
        el('button', {
          type: 'button', class: 'btn btn-ghost btn-sm', text: 'Forget',
          onclick: async () => {
            if (!confirm(`Remove ${source.root} from the catalog? Bucket objects are not touched.`)) return;
            await api.removeSource(source.id);
            onRefresh();
          },
        }),
      ]));
    }

    const embedder = status.embedder || {};
    $('embed-detail').textContent =
      `${embedder.backend || 'not loaded'} · ${embedder.dim || 0}d · ` +
      `${count(status.stats.embedded)} of ${count(status.stats.ok)} assets embedded. ${embedder.detail || ''}`;

    const models = clear($('model-status'));
    for (const annotator of status.annotators) {
      models.append(el('div', { class: `list-row ${annotator.available ? 'good' : 'bad'}` }, [
        el('div', { class: 'grow' }, [
          el('div', { text: `${annotator.name} — ${annotator.available ? 'ready' : 'unavailable'}` }),
          el('div', { class: 'sub', text: annotator.detail }),
        ]),
      ]));
    }

    const jobList = clear($('job-list'));
    if (!status.jobs.length) jobList.append(el('div', { class: 'muted-row', text: 'Nothing has run yet.' }));
    for (const job of status.jobs) {
      jobList.append(el('div', { class: `list-row ${job.status === 'error' ? 'bad' : ''}` }, [
        el('div', { class: 'grow' }, [
          el('div', { text: `${job.kind} · ${job.status}` }),
          el('div', { class: 'sub', text: `${count(job.done)}/${count(job.total)} · ${job.elapsed}s · ${job.message || ''}` }),
        ]),
      ]));
    }

    const stats = status.stats;
    $('status-assets').textContent =
      `${count(stats.ok)} assets · ${count(stats.folders)} folders · ${bytes(stats.bytes)}`;
    $('status-embedder').textContent = embedder.supports_text
      ? `semantic search on (${embedder.backend})`
      : 'image similarity only';
    $('status-embedder').title = embedder.detail || '';
  }

  function runIndex(root, force) {
    api.index({ root, force })
      .then(job => watchJob(job, { onDone: onRefresh }))
      .catch(error => notice(error.message, 'sticky'));
  }

  $('index-run').addEventListener('click', () => {
    const root = $('source-root').value.trim();
    if (!root) { notice('Enter a gs:// URI or a local directory first.'); return; }
    runIndex(root, $('index-force').checked);
  });
  $('embed-missing').addEventListener('click', () => {
    api.embed(false).then(job => watchJob(job, { onDone: onRefresh })).catch(e => notice(e.message, 'sticky'));
  });
  $('embed-rebuild').addEventListener('click', () => {
    if (!confirm('Rebuild every vector? Thumbnails, quality scores and tags are kept.')) return;
    api.embed(true).then(job => watchJob(job, { onDone: onRefresh })).catch(e => notice(e.message, 'sticky'));
  });

  return { open: modal.open, render };
}
