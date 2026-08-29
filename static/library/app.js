/**
 * OceanFrame Library — wiring.
 *
 * One query object drives everything: changing a filter, a facet, the folder
 * scope or the search box emits 'query', which re-runs the search and refreshes
 * the rail. Selection is separate so tagging never disturbs the result set.
 */
'use strict';

import { $, notice } from './ui.js';
import { api } from './api.js';
import { state, subscribe, setQuery, emit, clearSelection, clearVectorQuery } from './state.js';
import { createGrid } from './grid.js';
import { createFilters } from './filters.js';
import { createDetail } from './detail.js';
import { createAdmin } from './admin.js';
import { createDatasets } from './datasets.js';
import { watchJob } from './jobs.js';

const detail = createDetail();
const grid = createGrid({ onOpen: id => detail.open(id) });
const filters = createFilters();
const datasets = createDatasets();
const admin = createAdmin({ onRefresh: () => { refreshStatus(); emit('query'); } });

let searchToken = 0;

// ── Search ────────────────────────────────────────────────────────────────

async function runSearch() {
  const token = ++searchToken;
  try {
    const results = await api.search(state.query);
    if (token !== searchToken) return;      // a newer search already landed
    state.results = results;
    grid.render();
    if (results.note) notice(results.note);
  } catch (error) {
    notice(error.message, 'sticky');
  }
}

subscribe(event => {
  if (event === 'query') {
    runSearch();
    filters.refresh();
  } else if (event === 'selection') {
    renderSelection();
    grid.render();
  }
});

// ── Selection bar ─────────────────────────────────────────────────────────

function renderSelection() {
  const bar = $('action-bar');
  const n = state.selection.size;
  $('selection-count').textContent = `${n.toLocaleString()} selected`;
  bar.classList.toggle('hidden', n === 0);
}

function selectedIds() {
  return Array.from(state.selection);
}

$('tag-add').addEventListener('click', async () => {
  const names = splitNames($('tag-input').value);
  if (!names.length) { notice('Type one or more tag names first.'); return; }
  try {
    await api.addTags({ asset_ids: selectedIds(), names });
    $('tag-input').value = '';
    notice(`Tagged ${state.selection.size} images.`);
    emit('query');
  } catch (error) { notice(error.message, 'sticky'); }
});

$('tag-remove').addEventListener('click', async () => {
  const names = splitNames($('tag-input').value);
  if (!names.length) { notice('Type the tag to remove.'); return; }
  try {
    await api.removeTags({ asset_ids: selectedIds(), names });
    emit('query');
  } catch (error) { notice(error.message, 'sticky'); }
});

$('annotate-run').addEventListener('click', () => {
  const annotator = $('annotate-model').value;
  const prompts = splitNames($('annotate-prompts').value);
  if (annotator === 'sam3' && !prompts.length) {
    notice('SAM 3 needs concepts to look for — e.g. "fish, bleached coral".');
    return;
  }
  api.annotate({ annotator, asset_ids: selectedIds(), prompts })
    .then(job => watchJob(job, { onDone: () => { refreshStatus(); emit('query'); } }))
    .catch(error => notice(error.message, 'sticky'));
});

$('dataset-create').addEventListener('click', async () => {
  const name = $('dataset-name').value.trim();
  if (!name) { notice('Give the dataset a name first.'); return; }
  try {
    const dataset = await api.createDataset({ name, asset_ids: selectedIds(), split_mode: 'by_folder' });
    $('dataset-name').value = '';
    notice(`Built “${dataset.name}” with ${dataset.size} images.`);
    datasets.open();
  } catch (error) { notice(error.message, 'sticky'); }
});

$('selection-clear').addEventListener('click', clearSelection);

function splitNames(raw) {
  return raw.split(',').map(s => s.trim()).filter(Boolean);
}

// ── Search bar ────────────────────────────────────────────────────────────

function submitSearch() {
  clearVectorQuery();
  setQuery({
    text: $('search-text').value.trim(),
    mode: $('search-mode').value,
    sort: $('search-text').value.trim() ? 'relevance' : state.query.sort,
  });
}

$('search-btn').addEventListener('click', submitSearch);
$('search-text').addEventListener('keydown', event => { if (event.key === 'Enter') submitSearch(); });
$('search-mode').addEventListener('change', submitSearch);
$('sort-select').addEventListener('change', event => setQuery({ sort: event.target.value }));

$('search-image-btn').addEventListener('click', () => $('search-image-input').click());
$('search-image-input').addEventListener('change', async event => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const results = await api.searchByImage(file, { ...state.query, page: 0, similar_to: null });
    state.results = results;
    state.query.vector = [];                // marks the chip; the server did the embedding
    grid.render();
    filters.renderChips();
    notice(`Ranked ${results.matched} images against ${file.name}.`);
  } catch (error) {
    notice(error.message, 'sticky');
  } finally {
    event.target.value = '';
  }
});

// ── Header / drawers ──────────────────────────────────────────────────────

$('open-admin').addEventListener('click', event => { event.preventDefault(); admin.open(); });
$('open-datasets').addEventListener('click', event => { event.preventDefault(); datasets.open(); });
$('notice-close').addEventListener('click', () => $('notice-bar').classList.add('hidden'));

// ── Bootstrap ─────────────────────────────────────────────────────────────

async function refreshStatus() {
  try {
    const status = await api.status();
    state.status = status;
    admin.render(status);
    if (status.config.default_source && !$('source-root').value) {
      $('source-root').value = status.config.default_source;
    }
    $('annotate-prompts').placeholder = (status.config.sam3_prompts || []).join(', ') || 'fish, coral';
    if (!status.stats.assets) {
      notice('Nothing indexed yet — open “Index” and point the library at a bucket or folder.', 'sticky');
      admin.open();
    } else if (!status.embedder.supports_text) {
      notice(status.embedder.detail, 'sticky');
    }
    return status;
  } catch (error) {
    notice(error.message, 'sticky');
    return null;
  }
}

$('sort-select').value = state.query.sort;
renderSelection();
refreshStatus().then(() => emit('query'));
