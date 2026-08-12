# Interview Pitch Script: ApneaGuard AI

*(Targeted at Health-Tech/ResMed Engineering & ML Roles)*

**[0:00 - 0:30] The Problem & The Solution**
"Hi, I'd love to walk you through ApneaGuard AI. Sleep apnea is a massive, under-diagnosed issue globally, and polysomnography scoring is incredibly time-intensive for clinicians. I wanted to build an end-to-end, portfolio-grade system to automate this. I ingested the public PhysioNet ECG dataset, extracted raw physiological signals, and built a full machine-learning pipeline to detect per-minute apnea events and output an AASM-standard severity score."

**[0:30 - 1:00] The Engineering Discipline (The Hook)**
"But the most critical part of this project wasn't just throwing a model at data—it was structural validation. The most common pitfall in physiological ML is random data leakage across time-series windows, where a model essentially memorizes a patient's resting heart rate instead of learning generalized apnea signatures. I explicitly engineered a strict **subject-level k-fold split**. I even built an adversarial regression test in my CI/CD pipeline that mathematically proves my split algorithm restricts models from 'cheating' by isolating test subjects entirely from the training data."

**[1:00 - 1:30] The Honest Model Comparison & Explainability**
"With a mathematically sound baseline, I ran an honest comparison between a classical XGBoost model running on engineered Heart Rate Variability features, and a PyTorch 1D-CNN running on the raw waveform. The CNN won out on held-out patients, jumping from a 0.50 PR-AUC to a 0.60 PR-AUC. But because clinical trust requires transparency, I didn't stop at accuracy. I implemented Saliency and SHAP explainability pipelines, so a clinician reviewing the web dashboard can click a flagged window and physically see exactly which waveform anomalies triggered the model."

**[1:30 - 2:00] The Full-Stack Production Reality & Remediation**
"Finally, I didn't leave it as a Jupyter Notebook. I operationalized it into a modern, decoupled serverless architecture on Render, Vercel, and Neon. More importantly, during a rigorous pre-deployment code review, I caught a critical issue: I found that my own generated codebase's model-serving layer was a placeholder heuristic instead of real inference. I traced it to a missing model-persistence step in the training pipeline, and methodically fixed the entire pipeline end to end. With fully automated GitHub Actions CI/CD pipelines, it’s a complete vertical slice of how I'd approach a resilient, modern health-tech application at a company like ResMed—from rigorous data science down to production DevOps."

