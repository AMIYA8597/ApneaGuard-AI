# ApneaGuard AI - Deployment Notes

The ApneaGuard AI platform is deployed across three specialized, free-tier Platform-as-a-Service (PaaS) providers to minimize costs while providing a robust, modern architecture.

## 1. Database: Neon (Serverless Postgres)

Neon provides a scale-to-zero serverless PostgreSQL database. It wakes up in under a second upon receiving a query, making it perfect for our low-traffic portfolio deployment.

**Steps to Provision:**
1. Sign up at [Neon.tech](https://neon.tech) and create a new project (Free Tier: 0.5 GB storage).
2. Once the database is created, copy the **Postgres Connection String** from the dashboard.
3. Locally, set this connection string as your `DATABASE_URL` environment variable:
   - *Note: You may need to append `?sslmode=require` to the string if Neon requires it.*
4. Run the Alembic migrations to build the schema:
   ```bash
   export DATABASE_URL="postgresql://user:pass@ep-host.region.aws.neon.tech/neondb?sslmode=require"
   alembic upgrade head
   ```
5. Seed the database with demo recordings:
   ```bash
   python -m db.seed
   ```

## 2. Backend API: Render (Docker Web Service)

The FastAPI backend is deployed as a Docker Web Service on Render's free tier.

**Important Caveat for Live Demos:**
> [!WARNING]
> Render's free tier spins down the web service after ~15 minutes of inactivity and takes roughly a minute to wake on the next request. If this is being demoed live (e.g., in an interview), send a warm-up request to the backend 2-3 minutes before the demo starts to avoid a visible cold-start delay.

**Steps to Deploy:**
1. Go to [Render](https://render.com) and create a new **Web Service**.
2. Connect your GitHub repository.
3. **Language/Environment**: Select `Docker`. Render will automatically detect the `Dockerfile` at the root.
4. **Environment Variables**:
   - `DATABASE_URL`: Set to the Neon connection string.
   - `ALLOWED_ORIGINS`: Temporarily set to `*` or leave blank, but **update this to your Vercel frontend URL** once step 3 is complete.
   - `MODEL_RELEASE_URL`: Set to the raw download URL of the `models.tar.gz` artifact from your GitHub Releases. (Create a GitHub Release on your repo and upload a tarball of the `models/artifacts/` directory).
5. Deploy the service.

## 3. Frontend: Vercel (Next.js)

The frontend is a decoupled Next.js (App Router) dashboard hosted on Vercel's Edge Network.

**Steps to Deploy:**
1. Go to [Vercel](https://vercel.com) and select **Add New Project**.
2. Connect your GitHub repository.
3. **Framework Preset**: Next.js.
4. **Root Directory**: `frontend/` (since the project is structured as a monorepo).
5. **Environment Variables**:
   - `NEXT_PUBLIC_API_BASE`: Set to the public Render URL of your backend API (e.g., `https://apneaguard-api.onrender.com`).
6. Deploy the project.
7. Finally, take the resulting Vercel domain (e.g., `https://apneaguard-frontend.vercel.app`) and add it to the `ALLOWED_ORIGINS` environment variable in your Render backend settings, then redeploy the backend to secure CORS.

---

## Appendix: Alternative Local/AWS Deployment Strategy

*The following notes detail how this project would be deployed to AWS in a traditional monolithic setup for production, preserving the original EC2 cost-estimation strategy.*

### AWS Budget Alerts & Free-Tier Tracking
**Important Dates:**
- **6-Month Free Credit Expiration Date**: February 12, 2027

> [!WARNING]
> **Action Required Before Provisioning:** An AWS Budget Alert MUST be created in the AWS Billing Console, bound to the account email, triggering at a low threshold (e.g., $15.00). Ensure this is active before launching the EC2 instance below to prevent unexpected overage charges.

### AWS Architecture & Provisioning
For demonstration and interview purposes, the deployment is orchestrated as a single-node monolithic container stack using `docker-compose.prod.yml`. This prevents any managed-DB cold-start risks or excessive networking costs.

**Target AWS Footprint:**
- **Service**: 1x EC2 Instance (Amazon Linux 2023 or Ubuntu 24.04 LTS).
- **Instance Type**: `t3.micro` (or `t2.micro` depending on regional availability).
- **Storage**: 8 GB EBS General Purpose (gp3) root volume.
- **Estimated On-Demand Cost** *(if free tier lapses in us-east-1)*: 
  - `t3.micro`: ~$7.60 / month.
  - `8GB gp3 EBS`: ~$0.64 / month.
  - **Total**: ~$8.24 / month (excluding data transfer out).

### Security Group Configuration
To strictly limit the attack surface, the EC2 Security Group must be configured as follows:
- **Inbound Port 80 (HTTP)**: Allow `0.0.0.0/0` (For public web traffic to the dashboard).
- **Inbound Port 443 (HTTPS)**: Allow `0.0.0.0/0` (For SSL termination if configured).
- **Inbound Port 22 (SSH)**: Allow **ONLY YOUR SPECIFIC IP** (e.g., `203.0.113.45/32`). Do not leave this open to `0.0.0.0/0`.
- **Outbound**: Allow all traffic.
