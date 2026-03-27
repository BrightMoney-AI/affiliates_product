# RPU Projection Pipeline — Deployment Guide (No S3)

## How It Works

```
Metabase (Card 2821 + 2820)
        │
        ▼
  Airflow DAG (daily 07:00 UTC)
  ├── Task 1: Pull Metabase data
  ├── Task 2: Run RPU model
  └── Task 3: Git commit rpu_output.json
        │
        ▼
  GitHub Repo (rpu-dashboard)
        │  (auto-trigger on push)
        ▼
  Vercel deploys static dashboard
```

No S3, no APIs, no servers. The dashboard is a single HTML file that reads
a bundled JSON file. Airflow updates the JSON daily via git push, and Vercel
auto-deploys on every commit.

---

## Step 1: Create the GitHub Repo

```bash
# Create a new repo on GitHub (can be private)
# Name: rpu-dashboard  (under BrightMoney-AI org)

# Clone it locally
git clone git@github.com:BrightMoney-AI/rpu-dashboard.git
cd rpu-dashboard

# Copy the dashboard files into it
cp -r vercel-dashboard/* .

# Your repo should look like:
# rpu-dashboard/
# ├── vercel.json
# └── public/
#     ├── index.html
#     └── data/
#         └── rpu_output.json    ← updated daily by Airflow

git add .
git commit -m "Initial dashboard setup"
git push origin main
```

---

## Step 2: Connect to Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# From the repo root, deploy
cd rpu-dashboard
vercel              # preview deploy — check it looks right
vercel --prod       # production deploy
```

Vercel will ask:
- **Link to existing project?** → No, create new
- **Project name?** → `rpu-dashboard`
- **Framework?** → Other
- **Output directory?** → `public`

After this, **every push to `main` auto-deploys**. No manual deploys needed.

### Optional: Custom domain
```bash
vercel domains add rpu.brightmoney.co
# Then add the CNAME record Vercel shows
```

---

## Step 3: Create a GitHub Personal Access Token

The Airflow DAG needs to push to the repo. Create a token:

1. Go to **github.com → Settings → Developer settings → Personal Access Tokens → Fine-grained tokens**
2. **Repository access**: Select `rpu-dashboard` only
3. **Permissions**: Contents → Read and Write
4. Copy the token

---

## Step 4: Set Airflow Variables

Go to **Airflow UI → Admin → Variables** and add:

| Variable | Value | Notes |
|----------|-------|-------|
| `COSMOS_METABASE_USERNAME` | your.email@brightmoney.co | Metabase login |
| `COSMOS_METABASE_PASSWORD` | ●●●●●●●● | Metabase password |
| `GITHUB_PAT` | ghp_xxxxxxxxxxxx | From Step 3 |
| `rpu_dashboard_repo` | BrightMoney-AI/rpu-dashboard | Optional (this is the default) |

---

## Step 5: Deploy the Airflow DAG

```bash
cd brightmoney-airflow-platform

# Copy the DAG
cp rpu_projection_pipeline.py dags_v2/

# Push
git add dags_v2/rpu_projection_pipeline.py
git commit -m "Add RPU projection pipeline DAG"
git push origin master
```

---

## Step 6: Test

1. Go to **Airflow UI** → find `rpu_projection_pipeline` → toggle ON
2. Click **Trigger DAG** to run manually
3. Watch the 3 tasks complete:
   - `pull_metabase_data` → pulls from Metabase cards 2821 + 2820
   - `run_rpu_model` → runs the projection model
   - `commit_to_repo` → pushes `rpu_output.json` to GitHub
4. Vercel auto-deploys within ~60 seconds
5. Open your Vercel URL — dashboard shows fresh data

---

## Files

| File | Purpose |
|------|---------|
| `dags/rpu_projection_pipeline.py` | Airflow DAG (copy to `dags_v2/`) |
| `vercel-dashboard/public/index.html` | Dashboard UI |
| `vercel-dashboard/public/data/rpu_output.json` | Data file (updated daily by Airflow) |
| `vercel-dashboard/vercel.json` | Vercel config |
| `rpu_model.py` | Standalone model for local testing |

---

## Daily Operation

Once set up, everything is automatic:

- **07:00 UTC daily** → Airflow pulls Metabase → runs model → pushes JSON to GitHub
- **~07:02 UTC** → Vercel detects the push → rebuilds → dashboard is live with new data
- **You do nothing** — just open the dashboard URL anytime

### Manual refresh anytime:
```bash
# Option A: Trigger the DAG manually in Airflow UI

# Option B: Run locally and push
python3 rpu_model.py enroll.csv non_api.csv
cp rpu_output.csv public/data/rpu_output.json  # convert to JSON first
git add . && git commit -m "Manual update" && git push
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Metabase auth fails | Check username/password in Airflow Variables |
| Git push fails | Check GITHUB_PAT has write access to the repo |
| Vercel doesn't update | Check GitHub → repo → commits — is the push there? |
| Dashboard shows old data | Check Airflow task logs for errors |
| Card returns empty | Verify cards 2821/2820 are active with correct params |
