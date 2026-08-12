let API_BASE = window.API_BASE || "";

let currentChart = null;
let currentPredictions = [];
let currentRecordingId = null;

const elements = {
    recordingSelect: document.getElementById('recording-select'),
    modelRadios: document.querySelectorAll('input[name="model-version"]'),
    toggleBtns: document.querySelectorAll('.toggle-btn'),
    ahiValue: document.getElementById('ahi-value'),
    severityBand: document.getElementById('severity-band'),
    severityPanel: document.getElementById('severity-panel'),
    explanationPrompt: document.getElementById('explanation-prompt'),
    explanationContent: document.getElementById('explanation-content'),
    explanationImage: document.getElementById('explanation-image'),
    explanationFeatures: document.getElementById('explanation-features')
};

// Initialize
async function init() {
    try {
        const res = await fetch(`${API_BASE}/recordings/`);
        const recordings = await res.json();
        
        elements.recordingSelect.innerHTML = '<option value="" disabled selected>Select a Recording...</option>';
        recordings.forEach(rec => {
            const opt = document.createElement('option');
            opt.value = rec.id;
            opt.textContent = `Record ${rec.id.toUpperCase()} (${rec.split_role})`;
            elements.recordingSelect.appendChild(opt);
        });

        elements.recordingSelect.addEventListener('change', (e) => {
            currentRecordingId = e.target.value;
            loadDashboard();
        });

        elements.modelRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                // Update active class for styling
                elements.toggleBtns.forEach(btn => btn.classList.remove('active'));
                e.target.closest('.toggle-btn').classList.add('active');
                if (currentRecordingId) {
                    loadDashboard();
                }
            });
        });
    } catch (err) {
        console.error("Failed to load recordings", err);
        elements.recordingSelect.innerHTML = '<option value="" disabled>Error loading API</option>';
    }
}

function getSelectedModel() {
    return document.querySelector('input[name="model-version"]:checked').value;
}

async function loadDashboard() {
    if (!currentRecordingId) return;
    const modelVersion = getSelectedModel();

    // 1. Fetch Severity
    try {
        const sevRes = await fetch(`${API_BASE}/recordings/${currentRecordingId}/severity`);
        if (sevRes.ok) {
            const severity = await sevRes.json();
            elements.ahiValue.textContent = severity.ahi.toFixed(1);
            elements.severityBand.textContent = severity.severity_band;
            
            // Update color class
            elements.severityPanel.className = 'glass-panel severity-panel'; // reset
            elements.severityPanel.classList.add(`severity-${severity.severity_band.toLowerCase()}`);
        } else {
            // Need to run predict first if severity isn't cached
            await fetch(`${API_BASE}/recordings/${currentRecordingId}/predict?model_version=${modelVersion}`, { method: 'POST' });
            // Retry severity
            const retryRes = await fetch(`${API_BASE}/recordings/${currentRecordingId}/severity`);
            const severity = await retryRes.json();
            elements.ahiValue.textContent = severity.ahi.toFixed(1);
            elements.severityBand.textContent = severity.severity_band;
            elements.severityPanel.className = 'glass-panel severity-panel';
            elements.severityPanel.classList.add(`severity-${severity.severity_band.toLowerCase()}`);
        }
    } catch (err) {
        console.error("Failed to fetch severity", err);
    }

    // 2. Fetch Predictions to build the trace timeline
    try {
        const predRes = await fetch(`${API_BASE}/recordings/${currentRecordingId}/predict?model_version=${modelVersion}`, { method: 'POST' });
        currentPredictions = await predRes.json();
        renderChart(currentPredictions);
    } catch (err) {
        console.error("Failed to fetch predictions", err);
    }

    // Reset Explanation
    hideExplanation();
}

function renderChart(predictions) {
    const ctx = document.getElementById('prediction-chart').getContext('2d');
    
    if (currentChart) {
        currentChart.destroy();
    }

    // Prepare data
    // For visual simplicity, since we don't have the raw ECG in the payload (too large),
    // we'll plot the Apnea Probability as a line, highlighting areas > 0.5
    
    const labels = predictions.map(p => `Min ${p.minute_index}`);
    const data = predictions.map(p => p.probability);
    
    // Create red background zones for predictions > 0.5
    const backgroundColors = predictions.map(p => p.is_apnea ? 'rgba(239, 68, 68, 0.4)' : 'rgba(59, 130, 246, 0.1)');

    currentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Apnea Probability',
                data: data,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                pointBackgroundColor: predictions.map(p => p.is_apnea ? '#ef4444' : '#3b82f6'),
                pointRadius: 4,
                pointHoverRadius: 6,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            color: '#94a3b8',
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#94a3b8', maxTicksLimit: 20 }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#f8fafc' }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Probability: ${(context.raw * 100).toFixed(1)}%`;
                        }
                    }
                }
            },
            onClick: (event, elementsAtEvent) => {
                if (elementsAtEvent.length > 0) {
                    const idx = elementsAtEvent[0].index;
                    const pred = currentPredictions[idx];
                    if (pred.is_apnea) {
                        loadExplanation(pred.id);
                    } else {
                        elements.explanationPrompt.textContent = "Window is not flagged as apnea. Only flagged windows have explanations.";
                        elements.explanationPrompt.classList.remove('hidden');
                        elements.explanationContent.classList.add('hidden');
                    }
                }
            }
        }
    });
}

async function loadExplanation(predictionId) {
    elements.explanationPrompt.textContent = "Loading explanation...";
    elements.explanationPrompt.classList.remove('hidden');
    elements.explanationContent.classList.add('hidden');

    try {
        const res = await fetch(`${API_BASE}/predictions/${predictionId}/explanation`);
        if (res.ok) {
            const exp = await res.json();
            // Assuming plot_path is served, but we might need to fetch the image bytes or base64. 
            // In a real app we'd serve the images statically. 
            // For this project, if plot_path is an absolute path, it might not render directly in browser.
            // But let's assume there's a static route for it, or we just show the top features.
            
            elements.explanationPrompt.classList.add('hidden');
            elements.explanationContent.classList.remove('hidden');
            
            // Just display the text features if image isn't readily servable via web URL
            // Since we didn't add a static mount for the docs/ folder in main.py, we'll display features.
            const features = exp.top_features ? JSON.stringify(exp.top_features, null, 2) : "No features array returned.";
            elements.explanationFeatures.innerHTML = `<strong>Method:</strong> ${exp.method}<br><br><strong>Top Features:</strong><br><pre>${features}</pre>`;
            
            // If we have a plot path, we could try to show it (will break if it's C:/...)
            // elements.explanationImage.src = exp.plot_path; 
            elements.explanationImage.style.display = 'none'; // Hide image for now since it's a local file path
        } else {
            elements.explanationPrompt.textContent = "Explanation not available.";
        }
    } catch (err) {
        console.error("Failed to fetch explanation", err);
        elements.explanationPrompt.textContent = "Error fetching explanation.";
    }
}

function hideExplanation() {
    elements.explanationPrompt.textContent = "Select a flagged red window on the chart to inspect the model's explanation.";
    elements.explanationPrompt.classList.remove('hidden');
    elements.explanationContent.classList.add('hidden');
}

// Start
init();
