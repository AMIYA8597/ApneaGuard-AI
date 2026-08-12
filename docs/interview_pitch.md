# Interview Pitch Script: ApneaGuard AI

*(Targeted at Health-Tech/ResMed Engineering & ML Roles)*

**[0:00 - 0:30] The Problem & The Solution**
"Hi, I'd love to walk you through ApneaGuard AI. Sleep apnea is a massive, under-diagnosed issue globally, and polysomnography scoring is incredibly time-intensive for clinicians. I wanted to build an end-to-end, portfolio-grade system to automate this. I ingested the public PhysioNet ECG dataset, extracted raw physiological signals, and built a full machine-learning pipeline to detect per-minute apnea events and output an AASM-standard severity score."

**[0:30 - 1:00] The Engineering Discipline (The Hook)**
"But the most critical part of this project wasn't just throwing a model at data—it was structural validation. The most common pitfall in physiological ML is random data leakage across time-series windows, where a model essentially memorizes a patient's resting heart rate instead of learning generalized apnea signatures. I explicitly engineered a strict **subject-level k-fold split**. I even built an adversarial regression test in my CI/CD pipeline that mathematically proves my split algorithm restricts models from 'cheating' by isolating test subjects entirely from the training data."

**[1:00 - 1:30] The Honest Model Comparison & Explainability**
"With a mathematically sound baseline, I ran an honest comparison between a classical XGBoost model running on engineered Heart Rate Variability features, and a PyTorch 1D-CNN running on the raw waveform. The CNN won out on held-out patients, jumping from a 0.61 PR-AUC to 0.69 PR-AUC. But because clinical trust requires transparency, I didn't stop at accuracy. I implemented Saliency and SHAP explainability pipelines, so a clinician reviewing the web dashboard can click a flagged window and physically see exactly which waveform anomalies triggered the model."

**[1:30 - 2:00] The Full-Stack Production Reality**
"Finally, I didn't leave it as a Jupyter Notebook. I operationalized it. I wrapped the inference engine in a FastAPI backend with a PostgreSQL database, managed by Alembic migrations. I built a modern, interactive vanilla-JS web dashboard, containerized the entire stack into a multi-stage Docker build, and fully automated the testing and deployment through GitHub Actions. It’s a complete vertical slice of how I'd approach a production health-tech application at a company like ResMed—from rigorous data science down to resilient DevOps."
