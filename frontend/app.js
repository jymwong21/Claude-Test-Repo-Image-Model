'use strict';

const $ = id => document.getElementById(id);

const state = {
  photoFile: null,
  appearanceDescription: '',
  selectedStyle: null,
  isAnalyzing: false,
  isGenerating: false,
  generatedImages: [],
};

const API = {
  async loadStyles() {
    const res = await fetch('/api/styles');
    if (!res.ok) throw new Error('Failed to load styles');
    return res.json();
  },

  async analyzePhoto(file) {
    const fd = new FormData();
    fd.append('photo', file);
    const res = await fetch('/api/analyze', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Analysis failed');
    }
    return res.json();
  },

  async generateImage(payload) {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Generation failed');
    }
    return res.json();
  },

  async healthCheck() {
    const res = await fetch('/api/health');
    if (!res.ok) return null;
    return res.json();
  },
};

// ===== Loading Overlay =====
function showLoading(title, sub) {
  $('loadingTitle').textContent = title;
  $('loadingSub').textContent = sub;
  $('loadingOverlay').classList.remove('hidden');
}
function hideLoading() {
  $('loadingOverlay').classList.add('hidden');
}

// ===== Toast =====
function showToast(type, title, message, duration = 5000) {
  const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
    <div class="toast-body">
      <p class="toast-title">${title}</p>
      ${message ? `<p class="toast-msg">${message}</p>` : ''}
    </div>
  `;
  $('toastContainer').appendChild(toast);
  setTimeout(() => {
    toast.classList.add('fade-out');
    toast.addEventListener('animationend', () => toast.remove(), { once: true });
  }, duration);
}

// ===== Style Grid =====
function renderStyleGrid(styles) {
  const grid = $('styleGrid');
  grid.innerHTML = '';
  for (const [key, info] of Object.entries(styles)) {
    const card = document.createElement('div');
    card.className = 'style-card';
    card.dataset.style = key;
    card.innerHTML = `
      <span class="style-emoji">${info.emoji}</span>
      <p class="style-name">${info.name}</p>
      <p class="style-desc">${info.description}</p>
    `;
    card.addEventListener('click', () => selectStyle(key));
    grid.appendChild(card);
  }
}

function selectStyle(key) {
  state.selectedStyle = key;
  document.querySelectorAll('.style-card').forEach(c => {
    c.classList.toggle('selected', c.dataset.style === key);
  });
  updateGenerateBtn();
}

// ===== Photo Handling =====
function setPhoto(file) {
  if (!file || !file.type.startsWith('image/')) {
    showToast('error', 'Invalid file', 'Please select an image file.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('error', 'File too large', 'Please use an image under 10 MB.');
    return;
  }
  state.photoFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    $('previewImg').src = e.target.result;
    $('uploadPlaceholder').classList.add('hidden');
    $('uploadPreview').classList.remove('hidden');
  };
  reader.readAsDataURL(file);
  $('analyzeBtn').disabled = false;
}

function clearPhoto() {
  state.photoFile = null;
  $('fileInput').value = '';
  $('previewImg').src = '';
  $('uploadPlaceholder').classList.remove('hidden');
  $('uploadPreview').classList.add('hidden');
  $('analyzeBtn').disabled = true;
}

// ===== Analyze =====
async function handleAnalyze() {
  if (!state.photoFile || state.isAnalyzing) return;
  state.isAnalyzing = true;
  showLoading('Analyzing your photo…', 'Claude is examining your features');
  try {
    const result = await API.analyzePhoto(state.photoFile);
    state.appearanceDescription = result.appearance_description;
    $('appearanceInput').value = result.appearance_description;
    $('step2').classList.remove('hidden');
    $('step2').scrollIntoView({ behavior: 'smooth', block: 'start' });
    showToast('success', 'Analysis complete!', 'Your appearance has been captured.');
  } catch (err) {
    showToast('error', 'Analysis failed', err.message);
  } finally {
    hideLoading();
    state.isAnalyzing = false;
  }
}

// ===== Generate =====
function updateGenerateBtn() {
  const scene = $('sceneInput').value.trim();
  const appearance = $('appearanceInput').value.trim();
  const ready = !!state.selectedStyle && !!scene && !!appearance;
  const btn = $('generateBtn');
  btn.disabled = !ready;
  btn.classList.toggle('pulsing', ready);
}

async function handleGenerate() {
  if (!state.selectedStyle || state.isGenerating) return;
  const appearance = $('appearanceInput').value.trim();
  const scene = $('sceneInput').value.trim();
  const details = $('detailsInput').value.trim();
  const size = document.querySelector('input[name="orientation"]:checked')?.value || '1024x1024';

  if (!appearance) { showToast('warning', 'Missing description', 'Please provide an appearance description.'); return; }
  if (!scene) { showToast('warning', 'Missing scene', 'Please describe a scene or setting.'); return; }

  state.isGenerating = true;
  showLoading('Creating your portrait…', 'Claude is crafting the perfect prompt for DALL·E');
  try {
    const result = await API.generateImage({
      appearance_description: appearance,
      style: state.selectedStyle,
      scene_prompt: scene,
      additional_details: details,
      image_size: size,
    });
    addGeneratedImage(result);
    showToast('success', 'Portrait ready!', `Your ${result.style_name} portrait has been created.`);
    $('gallery').classList.remove('hidden');
    $('galleryGrid').firstElementChild?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    showToast('error', 'Generation failed', err.message);
  } finally {
    hideLoading();
    state.isGenerating = false;
  }
}

// ===== Gallery =====
function addGeneratedImage(data) {
  state.generatedImages.unshift(data);
  const grid = $('galleryGrid');
  const item = document.createElement('div');
  item.className = 'gallery-item';
  item.innerHTML = `
    <img src="${data.image_url}" alt="${data.style_name} portrait" loading="lazy" />
    <div class="gallery-item-overlay">
      <span class="gallery-item-label">${data.style_emoji} ${data.style_name}</span>
    </div>
  `;
  item.addEventListener('click', () => openLightbox(data));
  grid.prepend(item);
}

// ===== Lightbox =====
function openLightbox(data) {
  $('lightboxImg').src = data.image_url;
  $('lightboxStyle').textContent = `${data.style_emoji} ${data.style_name}`;
  $('lightboxPrompt').textContent = data.revised_prompt || data.enhanced_prompt || '';
  const dl = $('lightboxDownload');
  dl.href = data.image_url;
  dl.download = `portrait-${data.image_id}.png`;
  $('lightbox').classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  $('lightbox').classList.add('hidden');
  document.body.style.overflow = '';
  $('lightboxImg').src = '';
}

// ===== Health Check =====
async function updateStatus() {
  try {
    const h = await API.healthCheck();
    const dot = document.querySelector('.status-dot');
    const text = document.querySelector('.status-text');
    if (h && h.anthropic_configured && h.openai_configured) {
      dot.classList.add('ok');
      text.textContent = 'APIs connected';
    } else {
      dot.classList.add('error');
      const missing = [];
      if (!h?.anthropic_configured) missing.push('Anthropic');
      if (!h?.openai_configured) missing.push('OpenAI');
      text.textContent = `Missing: ${missing.join(', ')}`;
      showToast('warning', 'API keys missing', `Configure: ${missing.join(', ')} in .env`);
    }
  } catch {
    document.querySelector('.status-dot').classList.add('error');
    document.querySelector('.status-text').textContent = 'Backend offline';
  }
}

// ===== Init =====
async function init() {
  // Load styles
  try {
    const styles = await API.loadStyles();
    renderStyleGrid(styles);
  } catch {
    showToast('error', 'Could not load styles', 'Is the backend running?');
  }

  // Upload area click
  $('uploadArea').addEventListener('click', e => {
    if (e.target === $('browseBtn') || e.target === $('uploadArea') || e.target.closest('.upload-placeholder')) {
      $('fileInput').click();
    }
  });
  $('browseBtn').addEventListener('click', e => { e.stopPropagation(); $('fileInput').click(); });

  // File input
  $('fileInput').addEventListener('change', e => {
    if (e.target.files[0]) setPhoto(e.target.files[0]);
  });

  // Drag and drop
  const ua = $('uploadArea');
  ua.addEventListener('dragenter', e => { e.preventDefault(); ua.classList.add('drag-over'); });
  ua.addEventListener('dragover', e => { e.preventDefault(); ua.classList.add('drag-over'); });
  ua.addEventListener('dragleave', e => { if (!ua.contains(e.relatedTarget)) ua.classList.remove('drag-over'); });
  ua.addEventListener('drop', e => {
    e.preventDefault();
    ua.classList.remove('drag-over');
    const file = e.dataTransfer?.files[0];
    if (file) setPhoto(file);
  });

  // Buttons
  $('removePhotoBtn').addEventListener('click', e => { e.stopPropagation(); clearPhoto(); });
  $('analyzeBtn').addEventListener('click', handleAnalyze);
  $('generateBtn').addEventListener('click', handleGenerate);

  // Scene/appearance input → update generate btn
  $('sceneInput').addEventListener('input', updateGenerateBtn);
  $('appearanceInput').addEventListener('input', updateGenerateBtn);

  // Lightbox
  $('lightboxClose').addEventListener('click', closeLightbox);
  $('lightboxBackdrop').addEventListener('click', closeLightbox);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

  // Health
  updateStatus();
}

document.addEventListener('DOMContentLoaded', init);
