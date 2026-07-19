# Nightly email previews

Real-data renders of the k4Bench nightly regression email, produced by the
**production** renderer (`k4bench.regression.email.to_html`) — the same code path
`k4bench.regression.notify` uses to send the mail. Nothing here is hand-written
or hand-edited; regenerate with the command below.

Each file is the **full nightly email** covering every run group in the night's
`report.json`. The ALLEGRO / `Z → bb` section is the targeted validation scope
called out in the implementation brief; the email itself is not filtered to it.

## Regenerate

```bash
python email-previews/generate_previews.py 2026-06-27 2026-06-28
```

The script fetches the published artifacts over HTTP, fails clearly on any HTTP
error, and never sends email. It reuses first-confirmation `blame.json` sidecars
for same-release reconfirmations exactly as `notify` does in production.

## Inputs

| Night | report.json | blame.json |
|---|---|---|
| 2026-06-27 | <https://k4bench-data.web.cern.ch/_reports/2026-06-27/report.json> | <https://k4bench-data.web.cern.ch/_reports/2026-06-27/blame.json> |
| 2026-06-28 | <https://k4bench-data.web.cern.ch/_reports/2026-06-28/report.json> | <https://k4bench-data.web.cern.ch/_reports/2026-06-28/blame.json> |

The 2026-06-28 render additionally reuses the 2026-06-27 `blame.json` sidecar to
attribute its same-release reconfirmations.

## Dashboard reference scopes (ALLEGRO / Z → bb)

- Preview A (27 Jun): <https://k4bench-dashboard.app.cern.ch/?detector=ALLEGRO_o1_v03&platform=x86_64-almalinux9-gcc14.2.0-opt&sample=p8_ee_Zbb_ecm91&stack=key4hep-2026-06-27&range=Last+7+days&tab=Regressions&tmetric=mean_time_s&mmetric=mean_rss_mb&from=2026-07-17&to=2026-07-18&reg_all=0&report=2026-06-27>
- Preview B (28 Jun): <https://k4bench-dashboard.app.cern.ch/?detector=ALLEGRO_o1_v03&platform=x86_64-almalinux9-gcc14.2.0-opt&sample=p8_ee_Zbb_ecm91&stack=key4hep-2026-06-27&range=Last+7+days&tab=Regressions&tmetric=mean_time_s&mmetric=mean_rss_mb&from=2026-07-17&to=2026-07-18&reg_all=0&report=2026-06-28>

## Generated files

| File | Size | Subject |
|---|---|---|
| [`k4bench-nightly-2026-06-27.html`](k4bench-nightly-2026-06-27.html) | 49,144 bytes (≈48 KiB) | `[k4Bench][ACTION] 2026-06-27 — 318 new regressions` |
| [`k4bench-nightly-2026-06-28.html`](k4bench-nightly-2026-06-28.html) | 64,235 bytes (≈63 KiB) | `[k4Bench][ACTION] 2026-06-28 — 15 new, 304 reconfirmed` |

Both are comfortably under the 100 KiB target.

## Observed classifications (full report)

| Night | New | Reconfirmed | Watch | Failures | Reliable groups |
|---|---|---|---|---|---|
| 2026-06-27 | 318 | 0 | 33 | 0 | 7/7 |
| 2026-06-28 | 15 | 304 | 5 | 0 | 7/7 |

These match the expected classifications in the brief exactly (engine field
semantics unchanged; nothing in the classifier was altered to fit the data).

### ALLEGRO / `Z → bb` scope (`k4h_release == key4hep-2026-06-27`)

| Night | New | Reconfirmed | Watch | Notes |
|---|---|---|---|---|
| 2026-06-27 | 175 | 0 | 2 | Top ranked candidate `key4hep/k4geo#607` at 95%. |
| 2026-06-28 | 1 | 169 | 4 | Still the same release. The one New row is `median_time_s` / `without_ScreenSol` (≈+6.5%), first confirmed 2026-06-28. Reconfirmed rows say "First confirmed 27 Jun 2026" and reuse `key4hep/k4geo#607` (95%); the night's new change window ranks `HEP-FCC/k4RecCalorimeter#265` (85%) and `HEP-FCC/FCC-config#362` (80%). |

## Notes / limitations

- These are static HTML files; the tappable actions and links point at the
  live dashboard and require the dashboard/data hosts to be reachable.
- Each file wraps the production email body in a minimal
  `<!doctype html><meta charset="utf-8">…` document so a browser opening the
  `file://` preview decodes UTF-8 correctly. The real email needs no such
  wrapper — its MIME part declares the charset — so the renderer emits the body
  alone and only `generate_previews.py` adds the wrapper.
- Screenshots (PNG) are optional and not included. The HTML previews were
  visually checked at 700 px and 390 px viewport widths.
- Rendering uses inline styles only, no JavaScript and no external images or
  fonts, so the previews render the same in a conservative mail client as in a
  browser.
