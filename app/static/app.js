const form = document.querySelector("#edit-form");
const imageInput = document.querySelector("#image");
const sourcePreview = document.querySelector("#source-preview");
const maskPreview = document.querySelector("#mask-preview");
const resultPreview = document.querySelector("#result-preview");
const videoPreview = document.querySelector("#video-preview");
const statusBox = document.querySelector("#status");
const plannedBox = document.querySelector("#planned");
const planButton = document.querySelector("#plan-button");
const segmentButton = document.querySelector("#segment-button");
const editButton = document.querySelector("#edit-button");
const videoButton = document.querySelector("#video-button");
const tryonForm = document.querySelector("#tryon-form");
const tryonPersonInput = document.querySelector("#tryon-person");
const tryonGarmentInput = document.querySelector("#tryon-garment");
const tryonButton = document.querySelector("#tryon-button");

// ── Tab switching ──────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("tab--active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("tab-panel--active"));
    tab.classList.add("tab--active");
    document.querySelector(`[data-panel="${tab.dataset.tab}"]`).classList.add("tab-panel--active");
  });
});

function setBusy(isBusy) {
  planButton.disabled = isBusy;
  segmentButton.disabled = isBusy;
  editButton.disabled = isBusy;
  videoButton.disabled = isBusy;
  tryonButton.disabled = isBusy;
}

function buildFormData() {
  const data = new FormData(form);
  if (!data.get("seed")) {
    data.delete("seed");
  }
  return data;
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  sourcePreview.src = URL.createObjectURL(file);
  maskPreview.removeAttribute("src");
  resultPreview.removeAttribute("src");
  videoPreview.removeAttribute("src");
});

segmentButton.addEventListener("click", async () => {
  if (!imageInput.files[0]) {
    statusBox.textContent = "Upload an image first.";
    return;
  }
  setBusy(true);
  statusBox.textContent = "Detecting person and clothing regions...";

  try {
    const data = new FormData();
    data.set("image", imageInput.files[0]);
    const response = await fetch("/api/segment", { method: "POST", body: data });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "Segmentation failed.");
    }
    const blob = await response.blob();
    maskPreview.src = URL.createObjectURL(blob);
    statusBox.textContent = "Green = clothing detected. Edit will apply only to that region.";
  } catch (error) {
    statusBox.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

planButton.addEventListener("click", async () => {
  setBusy(true);
  statusBox.textContent = "Planning prompt...";
  plannedBox.textContent = "";

  try {
    const data = new FormData();
    data.set("prompt", document.querySelector("#prompt").value);
    const response = await fetch("/api/plan", { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Prompt planning failed.");
    plannedBox.textContent = payload.prompt;
    statusBox.textContent = "Prompt ready.";
  } catch (error) {
    statusBox.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(true);
  statusBox.textContent = "Generating edit. First run can take several minutes...";
  plannedBox.textContent = "";

  try {
    const response = await fetch("/api/edit", { method: "POST", body: buildFormData() });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "Image edit failed.");
    }
    plannedBox.textContent = response.headers.get("X-Planned-Prompt") || "";
    const blob = await response.blob();
    resultPreview.src = URL.createObjectURL(blob);
    statusBox.textContent = "Edit complete.";
  } catch (error) {
    statusBox.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

videoButton.addEventListener("click", async () => {
  setBusy(true);
  statusBox.textContent = "Generating video. First run can take several minutes...";
  plannedBox.textContent = "";

  try {
    const response = await fetch("/api/video", { method: "POST", body: buildFormData() });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "Video generation failed.");
    }
    plannedBox.textContent = response.headers.get("X-Planned-Prompt") || "";
    const blob = await response.blob();
    videoPreview.src = URL.createObjectURL(blob);
    await videoPreview.play().catch(() => {});
    statusBox.textContent = "Video complete.";
  } catch (error) {
    statusBox.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

// Show person photo in the source slot when selected in the Try On tab
tryonPersonInput.addEventListener("change", () => {
  const file = tryonPersonInput.files[0];
  if (file) sourcePreview.src = URL.createObjectURL(file);
});

// Show garment photo in the mask/garment slot
tryonGarmentInput.addEventListener("change", () => {
  const file = tryonGarmentInput.files[0];
  if (file) maskPreview.src = URL.createObjectURL(file);
});

tryonForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!tryonPersonInput.files[0] || !tryonGarmentInput.files[0]) {
    statusBox.textContent = "Upload both a person photo and a garment photo.";
    return;
  }
  setBusy(true);
  statusBox.textContent = "Running IDM-VTON try-on. First run downloads the model (~7 GB)...";
  resultPreview.removeAttribute("src");

  try {
    const data = new FormData(tryonForm);
    const seed = data.get("seed");
    if (!seed) data.delete("seed");

    const response = await fetch("/api/vton", { method: "POST", body: data });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "Try-on failed.");
    }
    const blob = await response.blob();
    resultPreview.src = URL.createObjectURL(blob);
    statusBox.textContent = "Try-on complete.";
  } catch (error) {
    statusBox.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

fetch("/api/health")
  .then((response) => response.json())
  .then((health) => {
    statusBox.textContent = `Runtime: ${health.device}, image: ${health.model}, video: ${health.video_model}`;
  })
  .catch(() => {
    statusBox.textContent = "Local API is not responding.";
  });
