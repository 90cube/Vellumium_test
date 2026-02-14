# Vellumium Node-RED Generation Pipeline

Node-RED flow that orchestrates AI image generation requests between Supabase, the local app_engine.py GPU server, and Cloudflare R2 storage.

## Architecture

```
Supabase (generation_queue table)
  --> Node-RED polls for pending jobs (every 5s)
  --> Calls local app_engine.py Gradio API (http://localhost:7860)
  --> Uploads result image to Cloudflare R2
  --> Updates Supabase with result URL + layer content_url
```

## Prerequisites

- **Node-RED v4.x** running at http://127.0.0.1:1880
- **app_engine.py** running at http://localhost:7860 (with a model loaded)
- **Supabase** project with `generation_queue` and `layers` tables
- **Cloudflare R2** bucket for image storage

## Required Node-RED Plugins

**None required.** This flow uses only built-in Node-RED nodes:
- `inject` (timer)
- `function` (JavaScript logic, including AWS SigV4 signing)
- `http request` (REST API calls)
- `http in` / `http response` (health check endpoints)
- `switch` (conditional routing)
- `catch` (error handling)
- `debug` (test output)
- `comment` (documentation)

The AWS Signature V4 signing for R2 uploads is implemented directly in a function node using Node.js built-in `crypto` module, so no additional npm packages like `aws4` or `node-red-contrib-aws` are needed.

## How to Import

1. Open Node-RED at http://127.0.0.1:1880
2. Click the hamburger menu (top-right) -> **Import**
3. Select **"select a file to import"**
4. Browse to `E:\Vellumium\node-red\generation-flow.json`
5. Click **Import**
6. Three flow tabs will appear: **Generation Pipeline**, **Health Check**, **Manual Test**
7. Click **Deploy** (top-right)

## Configuring Credentials

Credentials are stored as **flow-level environment variables** on each tab. This keeps them out of the flow JSON and avoids hardcoding secrets.

### Setting Environment Variables

1. **Double-click the "Generation Pipeline" tab header** (the tab name at the top)
2. In the dialog, find the **Environment Variables** section
3. Fill in the values:

| Variable | Value | Description |
|----------|-------|-------------|
| `SUPABASE_URL` | `https://lxhtjxonftnzlknnpzza.supabase.co` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | `(your service role key)` | Service role key (bypasses RLS) |
| `APP_ENGINE_URL` | `http://localhost:7860` | Local Gradio server (default) |
| `R2_ENDPOINT` | `https://60b1c001df7e87e1a2bf946b244bb33c.r2.cloudflarestorage.com` | R2 S3-compatible endpoint |
| `R2_BUCKET` | `vellumium-storage` | R2 bucket name |
| `R2_ACCESS_KEY_ID` | `(your R2 access key)` | R2 API token access key |
| `R2_SECRET_ACCESS_KEY` | `(your R2 secret key)` | R2 API token secret key |
| `R2_PUBLIC_URL` | *(optional)* | Public URL prefix for result images (e.g., custom domain) |

4. Click **Done**, then **Deploy**
5. Repeat for the **Health Check** and **Manual Test** tabs if you want to use those features

**Important:** The `SUPABASE_SERVICE_ROLE_KEY` is used (not the anon key) because the Node-RED worker needs to bypass Row Level Security to read and update any user's generation jobs.

## Flow Details

### Flow 1: Generation Pipeline

The main pipeline runs on the **"Generation Pipeline"** tab:

1. **Poll Timer** - Fires every 5 seconds
2. **Concurrency Lock** - Ensures only 1 GPU job runs at a time (prevents VRAM conflicts)
3. **Fetch Pending Job** - GETs the oldest `status=pending` row from `generation_queue`
4. **Extract Job** - If no pending jobs, stops. Otherwise extracts job data
5. **Acquire Lock** - Sets `flow.processing = true`
6. **Update to Processing** - PATCHes job `status` to `'processing'`
7. **Build Gradio Request** - Constructs the `/api/predict` POST body from job params
8. **Call App Engine** - POSTs to the Gradio API (120s timeout for GPU generation)
9. **Extract Image** - Handles three Gradio response formats: base64, file URL, or file path
10. **Fetch Image** (if needed) - Downloads the image file if it was returned as a URL/path
11. **R2 Upload** - Signs the request with AWS SigV4 and PUTs the PNG to R2
12. **Update Completed** - PATCHes job `status` to `'completed'` with `result_url`
13. **Update Layer** - If `layer_id` is present, PATCHes the layer's `content_url`
14. **Release Lock** - Sets `flow.processing = false`

**Error handling:** Any error at any stage triggers the error path, which:
- PATCHes the job `status` to `'failed'` with `error_message`
- Releases the concurrency lock
- A global `catch` node handles uncaught exceptions

### Flow 2: Health Check

The **"Health Check"** tab provides HTTP endpoints:

- **`GET http://127.0.0.1:1880/api/health`** - Returns Node-RED status, config check, and pipeline lock state
- **`GET http://127.0.0.1:1880/api/health/app-engine`** - Actively pings `app_engine.py /health` and reports if it's up or down

Example response from `/api/health`:
```json
{
  "node_red": { "status": "ok", "uptime_seconds": 3600 },
  "generation_pipeline": { "processing": false },
  "config": {
    "supabase_url": "configured",
    "supabase_key": "configured",
    "app_engine_url": "http://localhost:7860",
    "r2_endpoint": "configured",
    "r2_bucket": "vellumium-storage"
  },
  "timestamp": "2026-02-08T12:00:00.000Z"
}
```

### Flow 3: Manual Test

The **"Manual Test"** tab provides two inject buttons:

1. **"Test: Insert sample job"** - Inserts a test generation job into Supabase with a sample prompt. The Generation Pipeline will pick it up on its next poll cycle.
2. **"Test: Check queue status"** - Queries the last 10 jobs from `generation_queue` and displays them in the debug sidebar.

**Note:** The test job insert omits `user_id`. If your `generation_queue` table has a NOT NULL constraint on `user_id`, edit the "Build test job insert" function node to include a valid UUID.

## Testing the Flow

### Step 1: Verify Configuration
```
GET http://127.0.0.1:1880/api/health
```
All config values should show `"configured"`.

### Step 2: Check App Engine
```
GET http://127.0.0.1:1880/api/health/app-engine
```
Should return `"status": "ok"`. If it shows `"down"`, start app_engine.py first.

### Step 3: Load a Model
Make sure a model is loaded in app_engine.py. Check via:
```
GET http://localhost:7860/status
```
The response should show a loaded model.

### Step 4: Insert a Test Job
Click the "Test: Insert sample job" inject button in the Manual Test tab, or insert manually:
```sql
INSERT INTO generation_queue (user_id, prompt, params, status)
VALUES (
  'your-user-uuid',
  'A beautiful mountain landscape at sunset, photorealistic',
  '{"width": 1024, "height": 1024, "steps": 8}',
  'pending'
);
```

### Step 5: Watch the Pipeline
- Open the Node-RED debug sidebar (right panel, bug icon)
- Watch for `[Generation]` log messages
- The job should transition: `pending` -> `processing` -> `completed`
- Check the `result_url` in the debug output or query the table

### Step 6: Verify R2 Upload
Check the `result_url` returned in the completed job. If `R2_PUBLIC_URL` is configured, the URL will use that domain. Otherwise it will be the raw R2 endpoint URL.

## Gradio API Notes

The flow calls the Gradio `/api/predict` endpoint with `fn_index: 0`, which corresponds to the first `.click()` handler registered in the Gradio Blocks interface (the generate button).

The `data` array matches the input order from `gen_btn.click()` in `app_engine.py`:
```
[prompt, input_image, steps, height, width, control_image, control_type,
 control_scale, mask_image, seed, scheduler, mask_blur, denoise_strength]
```

If LoRA sliders are configured in app_engine.py, the Gradio API may expect additional values at the end of the `data` array. The current implementation omits LoRA values (all default to 0), which should work for basic text-to-image generation. If LoRA support is needed from the queue, update the "Build Gradio API request" function node to append LoRA scale values from `job.params`.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No jobs picked up | Missing config | Check `/api/health` for `"MISSING"` values |
| Jobs stay `pending` | Lock stuck | Check if `flow.processing` is stuck `true`; restart flows |
| Generation fails | Model not loaded | Load a model via the Gradio UI first |
| R2 upload fails | Wrong credentials | Verify R2 access key and endpoint |
| `fn_index` error | Gradio API changed | Check `/api/` on the Gradio server for correct fn_index |
| Timeout on generation | Slow GPU / large image | Increase timeout in the "POST app_engine generate" node |

## File Structure

```
E:\Vellumium\node-red\
  generation-flow.json   -- Node-RED flow (import this)
  README.md              -- This file
```
