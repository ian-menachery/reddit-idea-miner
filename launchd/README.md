# launchd setup (Mac Mini prod only)

The dashboard runs locally; the pipeline runs daily via `launchd` and produces fresh data the dashboard reads. Discovery (v2) will run weekly via the same mechanism — the `weekly.plist` is shipped but currently `Disabled=true`.

## One-time install

1. **Determine your paths.** The plists ship with two placeholders:
   - `PROJECT_PATH` — absolute path to this repo (e.g., `/Users/ian/projects/reddit`)
   - `UV_BIN` — absolute path to `uv` (e.g., `/opt/homebrew/bin/uv` on Apple Silicon, or `/usr/local/bin/uv` on Intel)

   Find them:
   ```bash
   pwd                # while inside the repo: this is PROJECT_PATH
   which uv           # this is UV_BIN
   ```

2. **Render the plists** into `~/Library/LaunchAgents/`:
   ```bash
   PROJECT="$(pwd)"
   UV="$(which uv)"
   mkdir -p "$HOME/Library/LaunchAgents"

   sed -e "s|PROJECT_PATH|$PROJECT|g" -e "s|UV_BIN|$UV|g" \
     launchd/com.reddit-miner.daily.plist \
     > "$HOME/Library/LaunchAgents/com.reddit-miner.daily.plist"

   sed -e "s|PROJECT_PATH|$PROJECT|g" -e "s|UV_BIN|$UV|g" \
     launchd/com.reddit-miner.weekly.plist \
     > "$HOME/Library/LaunchAgents/com.reddit-miner.weekly.plist"
   ```

3. **Load the agents:**
   ```bash
   launchctl load "$HOME/Library/LaunchAgents/com.reddit-miner.daily.plist"
   launchctl load "$HOME/Library/LaunchAgents/com.reddit-miner.weekly.plist"
   ```

4. **Verify:**
   ```bash
   launchctl list | grep reddit-miner
   # Should show two entries (PID column is "-" until they fire).
   ```

5. **Test fire (one-shot, don't wait until 4am):**
   ```bash
   launchctl start com.reddit-miner.daily
   tail -f data/logs/launchd-daily.out.log
   ```

## Schedule

- **Daily pipeline**: every day at **04:00 local**. Ingests new posts, embeds, reduces, clusters, extracts competitor mentions, fires alerts.
- **Weekly discovery**: every Sunday at **05:00 local**. **Disabled in v1** — `discover` command not implemented.

## Logs

- `data/logs/launchd-daily.{out,err}.log` — stdout/stderr from launchd-invoked runs.
- `data/logs/miner.log` — rotating application log (10MB × 5 backups).

## Dashboard over Tailscale

Streamlit is local-only by default. To reach it from your laptop while the Mac Mini runs the cron:

1. **Tailscale** on both machines (one-time): `https://tailscale.com/download`. Note the Mac Mini's Tailscale IP, e.g. `100.x.y.z`.
2. **Start Streamlit bound to the Tailscale interface:**
   ```bash
   uv run streamlit run app.py --server.address=0.0.0.0 --server.port=8501
   ```
   (`0.0.0.0` is fine because Tailscale firewalls the rest of the network; only nodes in your tailnet can reach it.)
3. **Visit** `http://100.x.y.z:8501` from your laptop. Bookmark it.

To make Streamlit run as another launchd agent so it survives reboots, add a third plist along the same lines (omitted from v1 — `streamlit run` in a tmux is fine for personal use).

## Disable / reload / uninstall

```bash
launchctl unload "$HOME/Library/LaunchAgents/com.reddit-miner.daily.plist"
launchctl unload "$HOME/Library/LaunchAgents/com.reddit-miner.weekly.plist"
rm "$HOME/Library/LaunchAgents/com.reddit-miner."*.plist
```

After editing a plist: `unload` then `load` again (launchd doesn't hot-reload).
