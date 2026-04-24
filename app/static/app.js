const form = document.querySelector("#edit-form");
const imageInput = document.querySelector("#image");
const sourcePreview = document.querySelector("#source-preview");
const resultPreview = document.querySelector("#result-preview");
const videoPreview = document.querySelector("#video-preview");
const statusBox = document.querySelector("#status");
const plannedBox = document.querySelector("#planned");
const planButton = document.querySelector("#plan-button");
const editButton = document.querySelector("#edit-button");
const videoButton = document.querySelector("#video-button");

function setBusy(isBusy) {
  planButton.disabled = isBusy;
  editButton.disabled = isBusy;
  videoButton.disabled = isBusy;
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
  resultPreview.removeAttribute("src");
  videoPreview.removeAttribute("src");
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

fetch("/api/health")
  .then((response) => response.json())
  .then((health) => {
    statusBox.textContent = `Runtime: ${health.device}, image: ${health.model}, video: ${health.video_model}`;
  })
  .catch(() => {
    statusBox.textContent = "Local API is not responding.";
  });
