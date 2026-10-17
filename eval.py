"""Bin-Sight accuracy test: run labelled photos through the classifier and build a confusion matrix.

1. Put your photos in folders named after the CORRECT bin id, e.g.
       eval_photos/hydrocarbon/acetone_bottle.jpg
       eval_photos/sharps/needle.jpg
       eval_photos/aqueous/...
   Valid folder names are the bin ids in waste_categories.json, e.g. hydrocarbon, chlorinated,
   halogenated, aqueous, acid, silver_nitrate, heavy_metal, silica, cmr_solid, autoclave, sharps,
   broken_glass, contaminated_solids, recycling, general (the script lists them all if one is wrong)

   OR make a list (Excel, CSV or tab-separated) with columns
       Photo | Bin_Classification | Gemini_Classification | Eval_Correct
   fill in Photo (e.g. photo_1) and Bin_Classification (e.g. Hydrocarbon), and run
       python eval.py --list data/photo_list.xlsx      (photos are read from data/photos/)
   It writes photo_list_results.xlsx with the last two columns filled in.

2. Run:   python eval.py eval_photos
   Options:
     --bins default    offer the default lab's bins (default; same as the app's starting selection)
     --bins all        offer every bin type, all 32 (the hardest test)
     --bins folders    only offer the bins you have folders for (like a lab that ticked just those)
     --sleep 1.0       seconds to wait between Gemini calls (raise it if you hit rate limits)
     --fresh           ignore cached results and re-run every photo

3. Open eval_report.html in your browser. Results are cached in eval_cache.json,
   so if it stops halfway you can just run it again and it carries on.
"""
import argparse
import csv
import html
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from classify import CATEGORIES, CATEGORY_BY_ID, classify_image  # noqa: E402

MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".heic": "image/heic", ".heif": "image/heif"}
CACHE_PATH = Path("eval_cache.json")
NONE = "none"            # Gemini said "none of your bins fit"
NOT_WASTE = "not_waste"  # Gemini said "this isn't waste"
_ORDER = list(CATEGORY_BY_ID) + [NOT_WASTE, NONE]


def _is_any_chemical(cid: str) -> bool:
    return "|" in cid and set(cid.split("|")) == set(CHEMICAL)


def short(cid: str) -> str:
    if _is_any_chemical(cid):
        return "Any chem"
    if "|" in cid:
        return " / ".join(short(c) for c in cid.split("|"))
    return {NONE: "None fit", NOT_WASTE: "Not waste"}.get(cid) or CATEGORY_BY_ID.get(cid, {}).get("label", cid)


def full(cid: str) -> str:
    if _is_any_chemical(cid):
        return "Any chemical waste bin"
    if "|" in cid:
        return " or ".join(full(c) for c in cid.split("|"))
    return {NONE: "No suitable bin", NOT_WASTE: "Not waste"}.get(cid) or CATEGORY_BY_ID.get(cid, {}).get("name", cid)


def is_ok(r) -> bool:
    """Correct if the prediction is any of the acceptable answers ('aqueous|hydrocarbon')."""
    return r["pred"] in r["truth"].split("|")


def order_key(cid: str):
    first = cid.split("|")[0]
    return (_ORDER.index(first) if first in _ORDER else 999, cid)


def collect(folder: Path) -> list[tuple[Path, str]]:
    photos, unknown = [], []
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        if sub.name not in CATEGORY_BY_ID:
            unknown.append(sub.name)
            continue
        photos += [(f, sub.name) for f in sorted(sub.iterdir()) if f.suffix.lower() in MIME]
    if unknown:
        print(f"⚠️  Skipping folders that aren't bin ids: {', '.join(unknown)}")
        print(f"   Valid names: {', '.join(CATEGORY_BY_ID)}\n")
    return photos


def run(photos, bins, sleep, fresh) -> list[dict]:
    cache = {} if fresh or not CACHE_PATH.exists() else json.loads(CACHE_PATH.read_text())
    rows, failed = [], []
    for i, (path, truth) in enumerate(photos, 1):
        tax = sum(Path(f).stat().st_mtime for f in ("taxonomy.json", "classify.py") if Path(f).exists())
        key = f"{path}|{path.stat().st_mtime}|{tax}|{','.join(b['id'] for b in bins)}"
        if key in cache:
            r = cache[key]
        else:
            try:
                out = None
                for attempt in range(5):
                    try:
                        out = classify_image(path.read_bytes(), MIME[path.suffix.lower()], bins)
                        break
                    except Exception as e:
                        msg = str(e)
                        if attempt < 4 and any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "overloaded")):
                            wait = 20 * (attempt + 1)
                            print(f"  [{i}/{len(photos)}] {path.name}: Gemini busy / rate-limited, waiting {wait}s and retrying…")
                            time.sleep(wait)
                            continue
                        raise
                r = {
                    "pred": NOT_WASTE if out.get("not_waste") else out["category"]["id"] if out["category"] else NONE,
                    "confidence": round(out["confidence"], 2),
                    "low_confidence": out["low_confidence"],
                    "alternative": out["alternative"]["id"] if out["alternative"] else "",
                    "missing_bin": out["missing_bin"],
                    "decided_by": out.get("decided_by", "ai"),
                    "matched_item": (out.get("policy") or {}).get("item_id", ""),
                    "matched_name": (out.get("policy") or {}).get("item_name", ""),
                    "description": out["item_description"],
                    "reasoning": out["reasoning"],
                }
            except Exception as e:
                print(f"  [{i}/{len(photos)}] {path.name}: ERROR {e}")
                failed.append(path.name)
                continue
            cache[key] = r
            CACHE_PATH.write_text(json.dumps(cache, indent=1))
            time.sleep(sleep)
        r = {**r, "file": str(path), "name": path.name, "truth": truth}
        mark = "✓" if is_ok(r) else "✗"
        via = f"policy {r['matched_item']}" if r.get("decided_by") == "policy" else "AI"
        print(f"  [{i}/{len(photos)}] {mark} {path.name}: {short(truth)} → {short(r['pred'])} ({r['confidence']:.0%}, {via})")
        rows.append(r)
    if failed:
        print(f"\n⚠️  {len(failed)} photos failed and are NOT in the results: {', '.join(failed[:10])}{' …' if len(failed) > 10 else ''}")
        print("   Run the same command again (without --fresh) to retry just those.")
    return rows


def outcome(r) -> str:
    if is_ok(r):
        return "correct"
    if r["pred"] == NONE:
        return "flagged"           # said no bin fits: safe, but unhelpful
    if r["low_confidence"]:
        return "wrong_warned"      # wrong, but told the student to check the label
    return "wrong_confident"       # the dangerous one


# ---------- colour helpers (sequential ramps: blue = correct diagonal, orange = errors) ----------
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"]
ORANGE = ["#fde4d8", "#f8c3aa", "#f39f7b", "#eb6834", "#c9501f", "#a33f16", "#7f300f"]


def ramp(colours, t):
    """t in (0, 1] -> a step on the ramp (light = few, dark = many)."""
    return colours[min(len(colours) - 1, max(0, round(t * (len(colours) - 1))))]


def report(rows, bins_mode, out_path: Path, excluded: int = 0):
    truths = sorted({r["truth"] for r in rows}, key=order_key)
    singles = [t for t in truths if "|" not in t]
    for t in truths:  # give each answer of a short combined label ("aqueous/hydrocarbon") a column
        if not _is_any_chemical(t) and len(t.split("|")) <= 3:
            singles += [c for c in t.split("|") if c not in singles]
    cols = sorted(set(singles), key=order_key)
    cols += sorted({r["pred"] for r in rows} - set(cols), key=order_key)

    cells = defaultdict(list)
    for r in rows:
        cells[(r["truth"], r["pred"])].append(r)
    row_tot = Counter(r["truth"] for r in rows)

    n = len(rows)
    oc = Counter(outcome(r) for r in rows)
    acc = oc["correct"] / n if n else 0
    pol = [r for r in rows if r.get("decided_by") == "policy"]
    pol_acc = sum(is_ok(r) for r in pol) / len(pol) if pol else 0
    ai = [r for r in rows if r.get("decided_by") != "policy"]
    ai_acc = sum(is_ok(r) for r in ai) / len(ai) if ai else 0
    sure = [r for r in rows if not r.get("uncertain")]
    unsure = [r for r in rows if r.get("uncertain")]
    sure_acc = sum(is_ok(r) for r in sure) / len(sure) if sure else 0
    unsure_acc = sum(is_ok(r) for r in unsure) / len(unsure) if unsure else 0

    # Matrix: each cell coloured by share of its ROW (so small classes aren't washed out).
    head = "".join(f'<th class="col" title="{html.escape(full(c))}"><span>{html.escape(short(c))}</span></th>' for c in cols)
    body = ""
    for t in truths:
        tds = ""
        for c in cols:
            items = cells.get((t, c), [])
            k = len(items)
            if not k:
                tds += '<td class="cell empty" aria-label="0"></td>'
                continue
            share = k / row_tot[t]
            bg = ramp(BLUE if c in t.split("|") else ORANGE, share)
            ink = "#ffffff" if share > 0.45 else "var(--text-primary)"
            tip = (f"<b>{k} of {row_tot[t]}</b> {html.escape(full(t))} photos → {html.escape(full(c))}<br>"
                   + "<br>".join(html.escape(i['name']) for i in items[:8])
                   + ("<br>…" if k > 8 else ""))
            tds += (f'<td class="cell" style="background:{bg};color:{ink}" data-tip="{html.escape(tip)}" '
                    f'tabindex="0" aria-label="{k} {html.escape(full(t))} predicted as {html.escape(full(c))}">{k}</td>')
        recall = sum(len(cells.get((t, c), [])) for c in t.split("|")) / row_tot[t]
        body += (f'<tr><th class="rowh" title="{html.escape(full(t))}">{html.escape(full(t))}</th>{tds}'
                 f'<td class="recall">{recall:.0%}</td></tr>')

    mistakes = sorted((r for r in rows if not is_ok(r)),
                      key=lambda r: ["wrong_confident", "wrong_warned", "flagged"].index(outcome(r)))
    label = {"wrong_confident": "⚠️ Wrong bin, confident", "wrong_warned": "Wrong, but warned to check label",
             "flagged": "Said no bin fits"}
    mistake_rows = "".join(
        f"<tr><td>{label[outcome(r)]}</td><td>{html.escape(r['name'])}</td><td>{html.escape(full(r['truth']))}</td>"
        f"<td>{html.escape(full(r['pred']))}</td><td>{r['confidence']:.0%}</td>"
        f"<td>{'Policy ' + html.escape(r.get('matched_item', '')) + ': ' + html.escape(r.get('matched_name', '')) if r.get('decided_by') == 'policy' else 'Gemini alone'}</td>"
        f"<td class='muted'>{html.escape(r['reasoning'])}</td></tr>" for r in mistakes
    ) or '<tr><td colspan="7" class="muted">No mistakes 🎉</td></tr>'

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Bin-Sight Accuracy</title>
<style>
:root {{ --surface:#fcfcfb; --card:#ffffff; --text-primary:#0b0b0b; --text-secondary:#52514e;
        --border:#e6e5e1; --empty:#f3f2ef; }}
@media (prefers-color-scheme: dark) {{ :root {{ --surface:#1a1a19; --card:#222220; --text-primary:#f0efec;
        --text-secondary:#c3c2b7; --border:#383835; --empty:#2a2a28; }} }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--surface); color:var(--text-primary); font-family:system-ui,-apple-system,sans-serif; }}
main {{ max-width:1000px; margin:0 auto; padding:28px 16px 60px; }}
h1 {{ margin:0 0 4px; font-size:26px }} h2 {{ font-size:18px; margin:32px 0 10px }}
.muted {{ color:var(--text-secondary) }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:20px 0 }}
.tile {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:16px }}
.tile .v {{ font-size:34px; font-weight:700; letter-spacing:-.02em }}
.tile .k {{ color:var(--text-secondary); font-size:14px; margin-top:2px }}
.scroll {{ overflow-x:auto; background:var(--card); border:1px solid var(--border); border-radius:14px; padding:16px }}
table.cm {{ border-collapse:separate; border-spacing:2px; font-variant-numeric:tabular-nums }}
.cm th {{ font-weight:600; font-size:12px; color:var(--text-secondary) }}
.cm .col {{ height:74px; vertical-align:bottom; padding:0 }}
.cm .col span {{ display:inline-block; writing-mode:vertical-rl; transform:rotate(180deg); padding:4px 0; }}
.cm .rowh {{ text-align:right; padding-right:10px; white-space:nowrap; max-width:220px; overflow:hidden; text-overflow:ellipsis }}
.cm .cell {{ width:44px; height:44px; min-width:44px; text-align:center; font-weight:700; border-radius:4px; cursor:default; outline-offset:2px }}
.cm .empty {{ background:var(--empty) }}
.cm .recall {{ padding-left:12px; font-size:13px; color:var(--text-secondary) }}
.axis {{ font-size:12px; color:var(--text-secondary); margin:6px 0 }}
.legend {{ display:flex; gap:18px; flex-wrap:wrap; font-size:13px; color:var(--text-secondary); margin-top:12px }}
.legend i {{ display:inline-block; width:14px; height:14px; border-radius:3px; vertical-align:-2px; margin-right:6px }}
table.list {{ width:100%; border-collapse:collapse; font-size:14px; background:var(--card); border-radius:14px; overflow:hidden }}
.list th, .list td {{ text-align:left; padding:10px; border-bottom:1px solid var(--border); vertical-align:top }}
.list th {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--text-secondary) }}
#tip {{ position:fixed; pointer-events:none; background:var(--card); color:var(--text-primary); border:1px solid var(--border);
        border-radius:10px; padding:8px 10px; font-size:13px; box-shadow:0 6px 20px rgba(0,0,0,.15); display:none; max-width:280px; z-index:9 }}
</style></head><body><main>
<h1>Bin-Sight accuracy test</h1>
<p class="muted">{n} labelled photos · Gemini model from classify.py · bins offered: { {"all": "every bin type", "default": "the default lab bins", "folders": "only the bins in the test folders"}[bins_mode] }</p>

<div class="tiles">
  <div class="tile"><div class="v">{acc:.0%}</div><div class="k">Correct bin ({oc['correct']} of {n})</div></div>
  <div class="tile"><div class="v">{oc['wrong_confident']}</div><div class="k">Wrong bin <i>and</i> confident (the risky case)</div></div>
  <div class="tile"><div class="v">{oc['wrong_warned']}</div><div class="k">Wrong, but told to check the label</div></div>
  <div class="tile"><div class="v">{oc['flagged']}</div><div class="k">Said no bin fits</div></div>
</div>
<div class="tiles">
  <div class="tile"><div class="v">{sure_acc:.0%}</div><div class="k">Accuracy on labels you were sure of ({len(sure)} photos)</div></div>
  <div class="tile"><div class="v">{unsure_acc:.0%}</div><div class="k">Accuracy on uncertain labels ({len(unsure)} photos, marked ? or several answers)</div></div>
  <div class="tile"><div class="v">{excluded}</div><div class="k">Photos left out (no label, or "idk")</div></div>
</div>
<div class="tiles">
  <div class="tile"><div class="v">{len(pol)}</div><div class="k">Photos matched to your waste policy ({len(pol) / n:.0%})</div></div>
  <div class="tile"><div class="v">{pol_acc:.0%}</div><div class="k">Accuracy when the policy decided</div></div>
  <div class="tile"><div class="v">{ai_acc:.0%}</div><div class="k">Accuracy when Gemini decided alone ({len(ai)} photos)</div></div>
</div>

<h2>Confusion matrix</h2>
<div class="scroll">
  <div class="axis">Rows: the correct bin · Columns: what Bin-Sight chose · Right-hand %: share of that row's photos it got right · Rows with "/" accept either answer (both count as correct)</div>
  <table class="cm"><thead><tr><th></th>{head}<th></th></tr></thead><tbody>{body}</tbody></table>
  <div class="legend"><span><i style="background:{BLUE[4]}"></i>Correct (diagonal), darker = more of the row</span>
  <span><i style="background:{ORANGE[3]}"></i>Mistake, darker = more of the row</span>
  <span><i style="background:var(--empty)"></i>None</span></div>
</div>

<h2>Every mistake</h2>
<table class="list"><thead><tr><th>Type</th><th>Photo</th><th>Correct bin</th><th>Bin-Sight chose</th><th>Conf.</th><th>Decided by</th><th>Its reasoning</th></tr></thead>
<tbody>{mistake_rows}</tbody></table>
</main><div id="tip" role="tooltip"></div>
<script>
const tip = document.getElementById("tip");
function show(el, x, y) {{ tip.innerHTML = el.dataset.tip; tip.style.display = "block";
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(x + 14, innerWidth - w - 8) + "px"; tip.style.top = Math.min(y + 14, innerHeight - h - 8) + "px"; }}
document.querySelectorAll("[data-tip]").forEach(el => {{
  el.addEventListener("mousemove", e => show(el, e.clientX, e.clientY));
  el.addEventListener("mouseleave", () => tip.style.display = "none");
  el.addEventListener("focus", () => {{ const r = el.getBoundingClientRect(); show(el, r.right, r.top); }});
  el.addEventListener("blur", () => tip.style.display = "none");
}});
</script></body></html>"""
    out_path.write_text(page)


# ---------------------------------------------------------------- list mode
# A table (Excel, CSV or tab-separated) with columns:
#   Photo | Bin_Classification | Gemini_Classification | Eval_Correct
# You fill Photo + Bin_Classification; this script fills the other two.

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def match_bin(text: str) -> str | None:
    """Turn whatever was typed in Bin_Classification into a bin id ('Hydrocarbon', 'hydrocarbon', 'HC', 'Silica waste'...)."""
    t = _norm(text)
    if not t:
        return None
    for c in CATEGORIES:
        if t in (_norm(c["id"]), _norm(c["name"]), _norm(c.get("label", ""))):
            return c["id"]
    hits = [c["id"] for c in CATEGORIES if t in _norm(c["name"]) or t in _norm(c["id"]) or _norm(c["id"]) in t]
    return hits[0] if len(set(hits)) == 1 else None


CHEMICAL = ["hydrocarbon", "chlorinated", "halogenated", "aqueous", "acid", "alkaline", "silver_nitrate",
            "heavy_metal", "contaminated_solids", "guanidinium", "cmr_liquid", "cmr_solid"]
ALIASES = {  # normalised phrase -> acceptable bins (broad ones count as "uncertain")
    "notwaste": [NOT_WASTE], "nowaste": [NOT_WASTE], "keep": [NOT_WASTE],
    "solidwaste": ["contaminated_solids"], "solid": ["contaminated_solids"], "solids": ["contaminated_solids"],
    "chemicalsolidwaste": ["contaminated_solids"],
    "glassbottlesempty": ["broken_glass"], "emptyglassbottles": ["broken_glass"], "glassbottles": ["broken_glass"],
    "emptybottles": ["broken_glass"], "glass": ["broken_glass"], "brokenglass": ["broken_glass"],
    "hazardouschemicalwaste": CHEMICAL, "chemicalwaste": CHEMICAL, "hazardouswaste": CHEMICAL,
    "biological": ["autoclave"], "bio": ["autoclave"], "sink": ["drain"], "drain": ["drain"],
}
BROAD = {"hazardouschemicalwaste", "chemicalwaste", "hazardouswaste"}
SKIP_WORDS = ["idk", "dontknow", "unknown", "notsure", "unsure", "nolabel"]


def _one(part: str):
    """One label fragment -> list of bin ids, or None."""
    p = part.strip(" ?.,;:()").strip()
    n = _norm(p)
    if not n:
        return None
    if n in ALIASES:
        return ALIASES[n]
    if n.endswith("waste") and n[:-5] in ALIASES:
        return ALIASES[n[:-5]]
    hit = match_bin(p)
    return [hit] if hit else None


def match_truth(text: str):
    """Free-text label -> (acceptable bin ids, uncertain?) ; ('skip', reason) ; or None if unrecognised."""
    raw = str(text).strip()
    n = _norm(raw)
    if not n:
        return ("skip", "no label")
    if any(w in n for w in SKIP_WORDS):
        return ("skip", "label says unsure")
    uncertain = "?" in raw or "depend" in n or bool(re.search(r"\bor\b", raw.lower()))
    ids = _one(raw)
    if ids is None:
        parts = re.split(r"/|\bor\b|,|&|\+", raw, flags=re.I)
        found = [_one(x) for x in parts]
        if len(parts) > 1 and all(found):
            ids = [i for f in found for i in f]
            uncertain = True
    if ids is None:  # last resort: look for any bin name or alias inside the sentence
        phrases = {**{_norm(c["name"]): [c["id"]] for c in CATEGORIES},
                   **{_norm(c["id"]): [c["id"]] for c in CATEGORIES}, **ALIASES}
        rest, ids = n, []
        for ph in sorted(phrases, key=len, reverse=True):
            if len(ph) >= 5 and ph in rest:
                ids += phrases[ph]
                rest = rest.replace(ph, " ")
        ids = ids or None
        uncertain = True
    if ids is None:
        return None
    if n.rstrip("waste") in BROAD or n in BROAD or len(set(ids)) > 1:
        uncertain = True
    seen = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    return seen, uncertain


def find_photo(name: str, photo_dir: Path) -> Path | None:
    """'photo_1' finds photo_1.jpg / .jpeg / .png / .heic …; a full filename works too. Case-insensitive."""
    name = str(name).strip()
    if not name:
        return None
    exact = photo_dir / name
    if exact.is_file():
        return exact
    stem = Path(name).stem.lower()
    for f in photo_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in MIME and (f.name.lower() == name.lower() or f.stem.lower() == stem):
            return f
    return None


def read_table(path: Path) -> tuple[list[str], list[list]]:
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        import openpyxl
        ws = openpyxl.load_workbook(path, data_only=True).active
        data = [["" if v is None else v for v in r] for r in ws.iter_rows(values_only=True)]
    else:
        text = path.read_text(encoding="utf-8-sig")
        dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",\t;")
        data = list(csv.reader(text.splitlines(), dialect))
    data = [r for r in data if any(str(v).strip() for v in r)]
    header = [str(h).strip() for h in data[0]]
    return header, [list(r) + [""] * (len(header) - len(r)) for r in data[1:]]


def col(header: list[str], name: str) -> int:
    for i, h in enumerate(header):
        if _norm(h) == _norm(name):
            return i
    raise SystemExit(f"Column '{name}' not found. Your columns are: {', '.join(header)}")


def run_list(list_path: Path, photo_dir: Path, bins_mode: str, sleep: float, fresh: bool):
    header, table = read_table(list_path)
    ci_photo, ci_bin = col(header, "Photo"), col(header, "Bin_Classification")
    for extra in ["Gemini_Classification", "Eval_Correct", "Decided_By", "Confidence"]:
        if not any(_norm(h) == _norm(extra) for h in header):
            header.append(extra)
            for r in table:
                r.append("")
    ci_gem, ci_ok = col(header, "Gemini_Classification"), col(header, "Eval_Correct")
    ci_by, ci_conf = col(header, "Decided_By"), col(header, "Confidence")

    photos, problems, row_of, uncertain_of, interp = [], [], {}, {}, []
    excluded, single_ids = 0, set()
    for idx, r in enumerate(table):
        if not str(r[ci_photo]).strip():
            continue
        parsed = match_truth(r[ci_bin])
        path = find_photo(r[ci_photo], photo_dir)
        if parsed and parsed[0] == "skip":
            excluded += 1
            r[ci_ok] = f"SKIPPED ({parsed[1]})"
            interp.append((r[ci_photo], r[ci_bin], f"left out: {parsed[1]}"))
        elif parsed is None:
            problems.append(f"{r[ci_photo]}: don't recognise bin '{r[ci_bin]}'")
            r[ci_ok] = "UNKNOWN BIN"
            excluded += 1
        elif path is None:
            problems.append(f"{r[ci_photo]}: photo not found in {photo_dir}")
            r[ci_ok] = "PHOTO NOT FOUND"
            excluded += 1
        else:
            ids, unsure = parsed
            truth = "|".join(ids)
            photos.append((path, truth))
            row_of[str(path)] = idx
            uncertain_of[str(path)] = unsure
            if len(ids) == 1 and ids[0] != NOT_WASTE:
                single_ids.add(ids[0])
            interp.append((r[ci_photo], r[ci_bin], full(truth) + ("   [uncertain]" if unsure else "")))
    print("How each label was read (check these!):")
    for ph, lab, meaning in interp:
        print(f"   {str(ph):<10} {str(lab)[:45]:<46} → {meaning}")
    print()
    if problems:
        print("⚠️  Skipping these rows:\n   " + "\n   ".join(problems))
        print(f"   Bin names you can use: {', '.join(c['name'] for c in CATEGORIES)}\n")
    if not photos:
        sys.exit("No usable rows. Check the Photo names and the photo folder (--photos).")

    # Offer the default lab bins plus any bin that is the single clear answer for some photo.
    bins = (CATEGORIES if bins_mode == "all"
            else [CATEGORY_BY_ID[i] for i in sorted(single_ids)] if bins_mode == "folders"
            else [c for c in CATEGORIES if c.get("default") or c["id"] in single_ids])
    print(f"Testing {len(photos)} photos against {len(bins)} bins…\n")
    rows = run(photos, bins, sleep, fresh)

    for r in rows:
        r["uncertain"] = uncertain_of.get(r["file"], False)
        t = table[row_of[r["file"]]]
        t[ci_gem] = full(r["pred"]) if r["pred"] != NONE else f"No suitable bin (needs: {r.get('missing_bin', '')})"
        t[ci_ok] = "TRUE" if is_ok(r) else "FALSE"
        t[ci_by] = f"Policy {r.get('matched_item', '')}" if r.get("decided_by") == "policy" else "Gemini alone"
        t[ci_conf] = f"{r['confidence']:.0%}"
    out = write_table(list_path, header, table, rows)
    return rows, out, excluded


def write_table(src: Path, header: list[str], table: list[list], rows: list[dict]) -> Path:
    n = len(rows)
    correct = sum(is_ok(r) for r in rows)
    if src.suffix.lower() in (".xlsx", ".xlsm"):
        import openpyxl
        from openpyxl.styles import Font, PatternFill
        out = src.with_name(src.stem + "_results.xlsx")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Results"
        ws.append(header)
        for r in table:
            ws.append(r)
        for c in ws[1]:
            c.font = Font(bold=True)
        ok_col = header.index(next(h for h in header if _norm(h) == "evalcorrect")) + 1
        green, red = PatternFill("solid", fgColor="D9F2E3"), PatternFill("solid", fgColor="FBE0DC")
        for row in ws.iter_rows(min_row=2, min_col=ok_col, max_col=ok_col):
            for c in row:
                c.fill = green if c.value == "TRUE" else red if c.value else PatternFill()
        last = ws.max_row
        ws.append([])
        rng = f"{ws.cell(2, ok_col).column_letter}2:{ws.cell(2, ok_col).column_letter}{last}"
        ws.append(["Accuracy", f'=COUNTIF({rng},"TRUE")/(COUNTIF({rng},"TRUE")+COUNTIF({rng},"FALSE"))'])
        ws.cell(ws.max_row, 2).number_format = "0%"
        ws.cell(ws.max_row, 1).font = Font(bold=True)
        for i, h in enumerate(header, 1):
            width = max(len(str(h)), *(len(str(r[i - 1])) for r in table)) if table else len(str(h))
            ws.column_dimensions[ws.cell(1, i).column_letter].width = min(48, width + 2)
        ws.freeze_panes = "A2"
        wb.save(out)
    else:
        out = src.with_name(src.stem + "_results.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(table)
            w.writerow([])
            w.writerow(["Accuracy", f"{correct}/{n} = {correct / n:.0%}" if n else "n/a"])
    return out


def main():
    ap = argparse.ArgumentParser(description="Bin-Sight accuracy test")
    ap.add_argument("folder", type=Path, nargs="?", help="folder of sub-folders named after the correct bin")
    ap.add_argument("--list", type=Path, help="Excel/CSV/TSV with Photo and Bin_Classification columns")
    ap.add_argument("--photos", type=Path, help="where the photos in --list are (default: data/photos)")
    ap.add_argument("--bins", choices=["default", "all", "folders"], default="default")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()

    filled, excluded = None, 0
    if a.list:
        if not a.list.is_file():
            sys.exit(f"List not found: {a.list}")
        default_dir = Path("data/photos")
        photo_dir = a.photos or (default_dir if default_dir.is_dir() else a.list.parent)
        if not photo_dir.is_dir():
            sys.exit(f"Photo folder not found: {photo_dir}")
        rows, filled, excluded = run_list(a.list, photo_dir, a.bins, a.sleep, a.fresh)
    else:
        if not a.folder or not a.folder.is_dir():
            sys.exit("Give a photo folder (python eval.py eval_photos) or a list (python eval.py --list photos.xlsx --photos photos/)")
        photos = collect(a.folder)
        if not photos:
            sys.exit("No photos found. Put them in sub-folders named after the correct bin id (see top of eval.py).")
        folder_ids = {t for _, t in photos}
        bins = [c for c in CATEGORIES if c.get("default") or c["id"] in folder_ids] if a.bins == "default" else CATEGORIES if a.bins == "all" else [CATEGORY_BY_ID[i] for i in sorted(folder_ids)]
        print(f"Testing {len(photos)} photos against {len(bins)} bins…\n")
        rows = run(photos, bins, a.sleep, a.fresh)
    if not rows:
        sys.exit("Every photo failed. Check your GEMINI_API_KEY and internet connection.")

    with open("eval_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "truth", "pred", "decided_by", "matched_item", "matched_name", "confidence", "low_confidence",
                                          "alternative", "missing_bin", "description", "reasoning", "file"])
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in w.fieldnames} for r in rows)

    report(rows, a.bins, Path("eval_report.html"), excluded)
    oc = Counter(outcome(r) for r in rows)
    print(f"\nAccuracy: {oc['correct']}/{len(rows)} = {oc['correct'] / len(rows):.0%}")
    print(f"Wrong & confident: {oc['wrong_confident']} · Wrong but warned: {oc['wrong_warned']} · No bin fits: {oc['flagged']}")
    sure = [r for r in rows if not r.get("uncertain")]
    if len(sure) != len(rows) and sure:
        print(f"Accuracy on labels you were sure of: {sum(is_ok(r) for r in sure)}/{len(sure)} = {sum(is_ok(r) for r in sure) / len(sure):.0%}")
    if filled:
        print(f"Filled-in list: {filled}")
    print("Report: eval_report.html   Raw results: eval_results.csv")


if __name__ == "__main__":
    main()
