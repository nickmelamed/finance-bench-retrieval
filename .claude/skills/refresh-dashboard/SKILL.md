---
name: refresh-dashboard
description: Regenerate dashboard/data.js from outputs/runs and check the README table against it.
---

1. Run `make dashboard`. It reads every folder in `outputs/runs/` and rewrites
   `dashboard/data.js`.
2. Run `make numbers` and show the result.
3. Report which run ids are now in `data.js` and whether the latest run
   matches the README table. Do not edit the README. If the table is stale,
   say which cells differ and ask the owner.
