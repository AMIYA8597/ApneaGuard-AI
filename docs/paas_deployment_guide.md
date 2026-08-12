# PaaS Deployment Guide: Vercel + Render + Neon

This project is architected to run entirely on modern Serverless and Platform-as-a-Service (PaaS) providers, taking full advantage of generous free tiers to host the Clinical Dashboard completely independently of the inference backend.

## Architecture Topology
- **Database**: Neon (Free-tier Serverless Postgres)
- **Backend API**: Render (Free-tier Docker Web Service)
- **Frontend Dashboard**: Vercel (Free-tier Static Deployment)

---

## 1. Neon Database Setup (Storage Layer)
Neon provides a scale-to-zero serverless PostgreSQL database. It wakes up in <1 second, meaning there is zero risk of the manual-unpause required by other providers (e.g., Supabase free tier).

1. Create a free account at [Neon.tech](https://neon.tech).
2. Create a new project (e.g., `apneaguard-db`) using **Postgres 16**.
3. Copy the pooled connection string (looks like `postgresql://[user]:[password]@[host]/[dbname]?sslmode=require`).
4. Keep this string ready for the Render configuration.

---

## 2. Render Setup (Inference Backend)
Render will host the FastAPI application and execute the Heavy ML inference (XGBoost / 1D-CNN) using our multi-stage Docker container.

1. Create a free account at [Render.com](https://render.com).
2. Create a new **Web Service**.
3. Connect your GitHub repository.
4. Set the runtime environment to **Docker**.
5. Under Environment Variables, add:
   - `DATABASE_URL`: *(Paste the Neon connection string here)*
6. Deploy. Render will automatically build the `Dockerfile` and expose the FastAPI endpoints.
7. Copy the public Render URL (e.g., `https://apneaguard-api.onrender.com`). 

> **Important**: Upon first deployment, you must run Alembic migrations. You can do this by SSHing into the Render container via the Render Dashboard shell and running `alembic upgrade head`, followed by `python db/seed.py` if you wish to seed dummy data.

---

## 3. Vercel Setup (Frontend)
Vercel hosts the ultra-fast Vanilla JS frontend dashboard, communicating with Render over CORS.

1. Create a free account at [Vercel.com](https://vercel.com).
2. Create a new project and import the GitHub repository.
3. Configure the Build Settings:
   - **Framework Preset**: Other (Static)
   - **Build Command**: `cd frontend && bash build.sh`
   - **Output Directory**: `frontend`
4. Under Environment Variables, add:
   - `NEXT_PUBLIC_API_BASE`: *(Paste your Render URL here, e.g., `https://apneaguard-api.onrender.com`)*
5. Deploy. Vercel will execute the build script to inject the Render URL directly into the static dashboard JavaScript.

---

## Summary of Costs
This entire architecture falls completely within the perpetual free tiers of all three providers. You will not incur any monthly costs running this portfolio project indefinitely.
