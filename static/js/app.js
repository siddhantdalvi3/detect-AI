class DetectAIApp {
	constructor() {
		this.historyKey = "detectai_history";
		this.themeKey = "detectai_theme";
		this.toastEnabledKey = "detectai_toast_enabled";
		this.maxHistory = 10;
		this.loadingTimer = null;
		this.queueStatusTimer = null;
		this.backendBadgeTimer = null;
		this.requestStartedAt = 0;
		this.lastLoadLevel = "normal";
		this.lastQueuedCount = 0;
		this.progressSim = 12;
		this.bindElements();
		this.bindEvents();
		this.applySavedTheme();
		this.applySavedToastPreference();
		this.startBackendLoadPolling();
		this.renderHistory();
		this.updateCharCounter();
	}

	bindElements() {
		this.inputText = document.getElementById("inputText");
		this.inputFile = document.getElementById("inputFile");
		this.fileTriggerBtn = document.getElementById("fileTriggerBtn");
		this.selectedFileName = document.getElementById("selectedFileName");
		this.charCounter = document.getElementById("charCounter");
		this.analyzeBtn = document.getElementById("analyzeBtn");
		this.detailedToggle = document.getElementById("detailedAnalysisToggle");
		this.themeToggle = document.getElementById("themeToggle");
		this.clearHistoryBtn = document.getElementById("clearHistoryBtn");
		this.toastToggle = document.getElementById("toastToggle");

		this.loadingSpinner = document.getElementById("loadingSpinner");
		this.queueStatusText = document.getElementById("queueStatusText");
		this.backendLoadBadge = document.getElementById("backendLoadBadge");
		this.toastContainer = document.getElementById("toastContainer");
		this.errorAlert = document.getElementById("errorAlert");
		this.errorMessage = document.getElementById("errorMessage");
		this.resultsSection = document.getElementById("resultsSection");

		this.bigScoreValue = document.getElementById("bigScoreValue");
		this.ensemblePrediction = document.getElementById("ensemblePrediction");
		this.ensembleProgress = document.getElementById("ensembleProgress");
		this.ensembleScore = document.getElementById("ensembleScore");
		this.highlightedText = document.getElementById("highlightedText");
		this.modelResults = document.getElementById("modelResults");
		this.modelBreakdown = document.getElementById("modelBreakdown");
		this.humanizerSection = document.getElementById("humanizerSection");
		this.humanizerSuggestions = document.getElementById("humanizerSuggestions");
		this.historyList = document.getElementById("historyList");
	}

	bindEvents() {
		this.inputText.addEventListener("input", () => this.updateCharCounter());
		this.fileTriggerBtn.addEventListener("click", () => this.inputFile.click());
		this.inputFile.addEventListener("change", () => this.onFileSelected());
		this.analyzeBtn.addEventListener("click", () => this.handleAnalyze());
		this.themeToggle.addEventListener("click", () => this.toggleTheme());
		this.clearHistoryBtn.addEventListener("click", () => this.clearHistory());
		if (this.toastToggle) {
			this.toastToggle.addEventListener("change", () =>
				this.setToastEnabled(this.toastToggle.checked),
			);
		}
	}

	updateCharCounter() {
		const count = (this.inputText.value || "").length;
		this.charCounter.textContent = `${count} chars`;
	}

	onFileSelected() {
		const file = this.inputFile.files && this.inputFile.files[0];
		this.selectedFileName.textContent = file ? file.name : "No file selected";
	}

	applySavedTheme() {
		const saved = localStorage.getItem(this.themeKey);
		if (saved === "dark") {
			document.body.classList.add("dark-theme");
		}
	}

	applySavedToastPreference() {
		const saved = localStorage.getItem(this.toastEnabledKey);
		const enabled = saved === null ? true : saved === "true";
		if (this.toastToggle) {
			this.toastToggle.checked = enabled;
		}
	}

	toggleTheme() {
		document.body.classList.toggle("dark-theme");
		const value = document.body.classList.contains("dark-theme")
			? "dark"
			: "light";
		localStorage.setItem(this.themeKey, value);
	}

	isToastEnabled() {
		if (!this.toastToggle) return false;
		return this.toastToggle.checked;
	}

	setToastEnabled(enabled) {
		localStorage.setItem(this.toastEnabledKey, String(Boolean(enabled)));
	}

	showToast(title, message, type = "info") {
		if (!this.isToastEnabled() || !this.toastContainer) return;
		if (!window.bootstrap || typeof window.bootstrap.Toast !== "function")
			return;

		const typeClass =
			{
				info: "text-bg-primary",
				success: "text-bg-success",
				warning: "text-bg-warning",
				danger: "text-bg-danger",
			}[type] || "text-bg-primary";

		const el = document.createElement("div");
		el.className = `toast align-items-center border-0 ${typeClass}`;
		el.setAttribute("role", "alert");
		el.setAttribute("aria-live", "assertive");
		el.setAttribute("aria-atomic", "true");
		el.innerHTML = `
      <div class="d-flex">
        <div class="toast-body">
          <strong>${this.escapeHtml(title)}:</strong> ${this.escapeHtml(message)}
        </div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
      </div>
    `;

		this.toastContainer.appendChild(el);
		const toast = new window.bootstrap.Toast(el, { delay: 3200 });
		el.addEventListener("hidden.bs.toast", () => el.remove());
		toast.show();
	}

	showError(message) {
		this.errorMessage.textContent = message;
		this.errorAlert.classList.remove("d-none");
	}

	clearError() {
		this.errorAlert.classList.add("d-none");
		this.errorMessage.textContent = "";
	}

	showLoading() {
		this.loadingSpinner.classList.remove("d-none");
		this.analyzeBtn.disabled = true;
		this.requestStartedAt = Date.now();
		if (this.queueStatusText) {
			this.queueStatusText.textContent = "Checking queue...";
		}
		this.startQueueStatusPolling();
		this.progressSim = 12;
		this.loadingTimer = window.setInterval(() => {
			this.progressSim = Math.min(this.progressSim + 9, 90);
			this.ensembleProgress.style.width = `${this.progressSim}%`;
			this.ensembleProgress.setAttribute(
				"aria-valuenow",
				`${this.progressSim}`,
			);
		}, 220);
	}

	hideLoading() {
		this.loadingSpinner.classList.add("d-none");
		this.analyzeBtn.disabled = false;
		this.stopQueueStatusPolling();
		if (this.queueStatusText) {
			this.queueStatusText.textContent = "";
		}
		if (this.loadingTimer) {
			window.clearInterval(this.loadingTimer);
			this.loadingTimer = null;
		}
	}

	startQueueStatusPolling() {
		this.stopQueueStatusPolling();
		this.updateQueueStatus();
		this.queueStatusTimer = window.setInterval(
			() => this.updateQueueStatus(),
			700,
		);
	}

	startBackendLoadPolling() {
		this.updateBackendLoadBadge();
		this.backendBadgeTimer = window.setInterval(
			() => this.updateBackendLoadBadge(),
			3000,
		);
	}

	applyQueueSnapshot(data, fromRequestPolling = false) {
		if (!data) return;

		const queued = Number(data.queued || 0);
		const running = Number(data.running || 0);
		const maxConcurrent = Math.max(1, Number(data.max_concurrent || 1));
		const maxQueued = Math.max(1, Number(data.max_queued || 1));
		const totalCapacity = maxConcurrent + maxQueued;
		const loadPercent = Math.min(
			100,
			Math.round(((running + queued) / Math.max(1, totalCapacity)) * 100),
		);

		let level = "normal";
		if (loadPercent >= 85) level = "hot";
		else if (loadPercent >= 45) level = "busy";

		if (this.backendLoadBadge) {
			this.backendLoadBadge.textContent = `Load: ${loadPercent}% (${running}R/${queued}Q)`;
			this.backendLoadBadge.classList.remove(
				"load-normal",
				"load-busy",
				"load-hot",
			);
			this.backendLoadBadge.classList.add(`load-${level}`);
		}

		if (fromRequestPolling && this.queueStatusText) {
			const elapsed = Math.max(
				0,
				Math.floor((Date.now() - this.requestStartedAt) / 1000),
			);
			const eta = Number(data.estimated_wait_seconds || 0).toFixed(1);
			this.queueStatusText.textContent = `Queue: ${queued}/${maxQueued}, Running: ${running}/${maxConcurrent}, ETA: ${eta}s, Elapsed: ${elapsed}s`;
		}

		if (this.isToastEnabled()) {
			if (queued > 0 && this.lastQueuedCount === 0) {
				this.showToast(
					"Queue",
					"Requests are queued. Processing in FIFO order.",
					"warning",
				);
			}
			if (queued === 0 && this.lastQueuedCount > 0) {
				this.showToast("Queue", "Queue is clear.", "success");
			}
			if (level !== this.lastLoadLevel) {
				if (level === "hot") {
					this.showToast(
						"Backend Load",
						"Server load is high. Expect slower responses.",
						"danger",
					);
				} else if (level === "busy") {
					this.showToast("Backend Load", "Server load is elevated.", "warning");
				}
			}
		}

		this.lastQueuedCount = queued;
		this.lastLoadLevel = level;
	}

	async updateBackendLoadBadge() {
		try {
			const res = await fetch("/api/queue/status", { cache: "no-store" });
			if (!res.ok) return;
			const data = await res.json();
			this.applyQueueSnapshot(data, false);
		} catch {
			if (this.backendLoadBadge) {
				this.backendLoadBadge.textContent = "Load: --";
				this.backendLoadBadge.classList.remove(
					"load-normal",
					"load-busy",
					"load-hot",
				);
			}
		}
	}

	stopQueueStatusPolling() {
		if (this.queueStatusTimer) {
			window.clearInterval(this.queueStatusTimer);
			this.queueStatusTimer = null;
		}
	}

	async updateQueueStatus() {
		if (!this.queueStatusText) return;

		try {
			const res = await fetch("/api/queue/status", { cache: "no-store" });
			if (!res.ok) return;

			const data = await res.json();
			this.applyQueueSnapshot(data, true);
		} catch {
			// Silently ignore queue polling errors; primary request error handling remains.
		}
	}

	resetResults() {
		this.resultsSection.classList.add("d-none");
		this.humanizerSection.classList.add("d-none");
		this.modelResults.innerHTML = "";
		this.modelBreakdown.innerHTML = "";
		this.highlightedText.innerHTML = "";
	}

	normalizeScore(raw) {
		if (raw === null || raw === undefined || Number.isNaN(Number(raw)))
			return 0;
		const n = Number(raw);
		return n <= 1 ? n * 100 : n;
	}

	async handleAnalyze() {
		this.clearError();
		this.resetResults();

		const text = (this.inputText.value || "").trim();
		const file = this.inputFile.files && this.inputFile.files[0];
		const detailed = this.detailedToggle.checked;

		if (text.length < 50 && !file) {
			this.showError(
				"Provide at least 50 characters of text, or upload a supported file.",
			);
			return;
		}

		this.showLoading();

		try {
			let data;
			let mode = "text";

			if (text.length >= 50) {
				data = await this.analyzeText(text, detailed);
			} else {
				mode = "file";
				data = await this.analyzeFile(file, detailed);
			}

			if (data && data.processing_status === "pending" && data.request_id) {
				this.showToast(
					"Queued",
					"Heavy analysis is queued. Showing fast result and refreshing when complete.",
					"warning",
				);
				data = await this.waitForDelayedResult(data.request_id, data);
			}

			this.renderResults(data, text);
			this.addHistoryItem(data, mode);
			this.renderHistory();
		} catch (err) {
			this.showError(err.message || "Analysis failed. Try again.");
		} finally {
			this.hideLoading();
		}
	}

	async waitForDelayedResult(requestId, fallbackData) {
		const maxAttempts = 35;
		for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
			await new Promise((resolve) => window.setTimeout(resolve, 900));

			try {
				const res = await fetch(
					`/api/detect/result/${encodeURIComponent(requestId)}`,
					{
						cache: "no-store",
					},
				);
				if (!res.ok) continue;

				const payload = await res.json();
				if (payload.status === "completed" && payload.result) {
					this.showToast("Completed", "Heavy analysis finished.", "success");
					return payload.result;
				}
				if (payload.status === "failed") {
					this.showToast(
						"Heavy Analysis Failed",
						payload.error || "Using fast result.",
						"danger",
					);
					return fallbackData;
				}
			} catch {
				// Keep polling until timeout.
			}
		}

		this.showToast(
			"Still Processing",
			"Heavy analysis is still running. Showing the fast result for now.",
			"info",
		);
		return fallbackData;
	}

	async analyzeText(text, detailed) {
		const endpoint = detailed
			? "/api/detect/text/detailed"
			: "/api/detect/text";
		const body = new FormData();
		body.append("text", text);
		body.append("allow_delayed", "true");

		const res = await fetch(endpoint, { method: "POST", body });
		const data = await res.json();
		if (!res.ok) throw new Error(this.getFriendlyApiError(res, data));
		return data;
	}

	async analyzeFile(file, detailed) {
		if (!file) throw new Error("No file selected.");
		const endpoint = detailed
			? "/api/detect/file/detailed"
			: "/api/detect/file";
		const body = new FormData();
		body.append("file", file);
		body.append("allow_delayed", "true");

		const res = await fetch(endpoint, { method: "POST", body });
		const data = await res.json();
		if (!res.ok) throw new Error(this.getFriendlyApiError(res, data));
		return data;
	}

	getFriendlyApiError(response, data) {
		const detail = (data && data.detail) || "Request failed.";

		if (response.status === 429) {
			const retryAfter = response.headers.get("Retry-After");
			if (retryAfter) {
				this.showToast(
					"Rate Limited",
					`Retry in about ${retryAfter}s.`,
					"warning",
				);
				return `Too many requests from this IP. Retry in about ${retryAfter}s.`;
			}
			this.showToast(
				"Rate Limited",
				"Please wait before sending another request.",
				"warning",
			);
			return "Too many requests from this IP. Please wait and retry.";
		}

		if (
			response.status === 503 &&
			String(detail).toLowerCase().includes("queue is full")
		) {
			this.showToast(
				"Queue Full",
				"Server queue is full. Try again shortly.",
				"danger",
			);
			return "Server is busy and the queue is full. Please retry in a moment.";
		}

		if (
			response.status === 503 &&
			String(detail).toLowerCase().includes("queue wait timeout")
		) {
			this.showToast(
				"Queue Timeout",
				"Queue wait timed out. Retry or shorten input.",
				"warning",
			);
			return "Server queue wait timed out. Try again or reduce request size.";
		}

		if (response.status === 504) {
			this.showToast(
				"Processing Timeout",
				"Request took too long. Try smaller content.",
				"warning",
			);
			return "Request timed out during processing. Please try a shorter input or retry.";
		}

		return detail;
	}

	renderResults(data, fallbackText) {
		const base = data.standard_result || data;
		const score = this.normalizeScore(
			base.ensemble_score ?? data.overall_analysis?.ai_probability ?? 0,
		);
		const scoreFixed = Math.max(0, Math.min(score, 100));

		const predictionRaw =
			base.ensemble_prediction ||
			data.overall_analysis?.prediction ||
			"Unknown";
		const prediction =
			String(predictionRaw).toUpperCase() === "AI"
				? "Likely AI"
				: "Likely Human";

		this.bigScoreValue.textContent = `${Math.round(scoreFixed)}%`;
		this.ensembleProgress.style.width = `${scoreFixed}%`;
		this.ensembleProgress.setAttribute(
			"aria-valuenow",
			String(Math.round(scoreFixed)),
		);
		this.ensembleScore.textContent = `Confidence: ${scoreFixed.toFixed(2)}%`;
		this.ensemblePrediction.innerHTML = `<span class="prediction-badge ${prediction.includes("AI") ? "prediction-ai" : "prediction-human"}">${prediction}</span>`;

		this.renderModelCards(base.model_results || {});
		this.renderBreakdown(data.model_breakdown || {});
		this.renderHighlighted(data.highlighted_html, fallbackText);
		this.renderSuggestions(
			base.humanizer_suggestions || data.humanizer_suggestions || [],
		);

		this.resultsSection.classList.remove("d-none");
	}

	renderModelCards(modelResults) {
		const keys = Object.keys(modelResults || {});
		if (!keys.length) {
			this.modelResults.innerHTML =
				"<div class='text-muted'>No model detail available.</div>";
			return;
		}

		const html = keys
			.map((name) => {
				const r = modelResults[name];
				const score = this.normalizeScore(r.ai_probability);
				return `
          <div class="model-card">
            <div class="fw-semibold">${this.escapeHtml(name)}</div>
            <div class="small text-muted">AI probability: ${score.toFixed(2)}%</div>
            <div class="small">Prediction: <strong>${this.escapeHtml(r.prediction || "N/A")}</strong></div>
          </div>
        `;
			})
			.join("");

		this.modelResults.innerHTML = html;
	}

	renderBreakdown(breakdown) {
		if (!breakdown || !breakdown.by_model) {
			this.modelBreakdown.innerHTML = "<div>No advanced stats available.</div>";
			return;
		}

		const modelItems = Object.entries(breakdown.by_model)
			.map(([name, val]) => {
				return `<li><strong>${this.escapeHtml(name)}</strong>: ${Number(val.detection_rate || 0).toFixed(2)}% (${val.sentences_detected || 0} sentences)</li>`;
			})
			.join("");

		const c = breakdown.consensus_analysis || {};
		this.modelBreakdown.innerHTML = `
      <div class="mb-2"><strong>Consensus</strong></div>
      <div class="mb-2">
        All AI: ${Number(c.all_models_agree_ai_percentage || 0).toFixed(2)}%,
        All Human: ${Number(c.all_models_agree_human_percentage || 0).toFixed(2)}%,
        Mixed: ${Number(c.mixed_opinions_percentage || 0).toFixed(2)}%
      </div>
      <ul class="mb-0">${modelItems}</ul>
    `;
	}

	renderHighlighted(highlightedHtml, fallbackText) {
		// Dispose old tooltips before rerender
		if (window.bootstrap && typeof window.bootstrap.Tooltip === "function") {
			this.highlightedText
				.querySelectorAll('[data-bs-toggle="tooltip"]')
				.forEach((el) => {
					const instance = window.bootstrap.Tooltip.getInstance(el);
					if (instance) instance.dispose();
				});
		}

		if (highlightedHtml && typeof highlightedHtml === "string") {
			this.highlightedText.innerHTML = highlightedHtml;

			// Initialize tooltips on newly inserted spans
			if (window.bootstrap && typeof window.bootstrap.Tooltip === "function") {
				this.highlightedText
					.querySelectorAll('[data-bs-toggle="tooltip"]')
					.forEach((el) => {
						new window.bootstrap.Tooltip(el);
					});
			}
		} else {
			this.highlightedText.textContent =
				fallbackText || "No detailed highlights in this mode.";
		}
	}

	renderSuggestions(items) {
		if (!items || !items.length) {
			this.humanizerSection.classList.add("d-none");
			this.humanizerSuggestions.innerHTML = "";
			return;
		}

		this.humanizerSuggestions.innerHTML = items
			.map((x) => `<li>${this.escapeHtml(x.suggestion || String(x))}</li>`)
			.join("");
		this.humanizerSection.classList.remove("d-none");
	}

	addHistoryItem(data, mode) {
		const base = data.standard_result || data;
		const score = this.normalizeScore(
			base.ensemble_score ?? data.overall_analysis?.ai_probability ?? 0,
		);
		const prediction =
			base.ensemble_prediction ||
			data.overall_analysis?.prediction ||
			"Unknown";

		const history = this.getHistory();
		history.unshift({
			timestamp: new Date().toISOString(),
			mode,
			score: Number(score.toFixed(2)),
			prediction,
		});

		localStorage.setItem(
			this.historyKey,
			JSON.stringify(history.slice(0, this.maxHistory)),
		);
	}

	getHistory() {
		try {
			const parsed = JSON.parse(localStorage.getItem(this.historyKey) || "[]");
			return Array.isArray(parsed) ? parsed : [];
		} catch {
			return [];
		}
	}

	renderHistory() {
		const history = this.getHistory();
		if (!history.length) {
			this.historyList.innerHTML =
				"<div class='history-empty'>No checks yet.</div>";
			return;
		}

		this.historyList.innerHTML = history
			.map((h) => {
				const dt = new Date(h.timestamp);
				return `
          <div class="history-item">
            <div class="d-flex justify-content-between">
              <strong>${this.escapeHtml(h.prediction)}</strong>
              <span>${Number(h.score).toFixed(1)}%</span>
            </div>
            <div class="small text-muted">${this.escapeHtml(h.mode)} • ${dt.toLocaleString()}</div>
          </div>
        `;
			})
			.join("");
	}

	clearHistory() {
		localStorage.removeItem(this.historyKey);
		this.renderHistory();
	}

	escapeHtml(value) {
		return String(value)
			.replaceAll("&", "&amp;")
			.replaceAll("<", "&lt;")
			.replaceAll(">", "&gt;")
			.replaceAll('"', "&quot;")
			.replaceAll("'", "&#39;");
	}
}

window.addEventListener("DOMContentLoaded", () => {
	new DetectAIApp();
});
