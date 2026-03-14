class DetectAIApp {
	constructor() {
		this.historyKey = "detectai_history";
		this.themeKey = "detectai_theme";
		this.maxHistory = 10;
		this.loadingTimer = null;
		this.progressSim = 12;
		this.bindElements();
		this.bindEvents();
		this.applySavedTheme();
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

		this.loadingSpinner = document.getElementById("loadingSpinner");
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

	toggleTheme() {
		document.body.classList.toggle("dark-theme");
		const value = document.body.classList.contains("dark-theme")
			? "dark"
			: "light";
		localStorage.setItem(this.themeKey, value);
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
		if (this.loadingTimer) {
			window.clearInterval(this.loadingTimer);
			this.loadingTimer = null;
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

			this.renderResults(data, text);
			this.addHistoryItem(data, mode);
			this.renderHistory();
		} catch (err) {
			this.showError(err.message || "Analysis failed. Try again.");
		} finally {
			this.hideLoading();
		}
	}

	async analyzeText(text, detailed) {
		const endpoint = detailed
			? "/api/detect/text/detailed"
			: "/api/detect/text";
		const body = new FormData();
		body.append("text", text);

		const res = await fetch(endpoint, { method: "POST", body });
		const data = await res.json();
		if (!res.ok) throw new Error(data.detail || "Text analysis failed.");
		return data;
	}

	async analyzeFile(file, detailed) {
		if (!file) throw new Error("No file selected.");
		const endpoint = detailed
			? "/api/detect/file/detailed"
			: "/api/detect/file";
		const body = new FormData();
		body.append("file", file);

		const res = await fetch(endpoint, { method: "POST", body });
		const data = await res.json();
		if (!res.ok) throw new Error(data.detail || "File analysis failed.");
		return data;
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
