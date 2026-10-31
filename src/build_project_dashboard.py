"""Build an offline, self-contained dashboard from verified research artifacts."""

import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def build(run_dir, output, sessions=()):
    if output.exists():
        raise FileExistsError("Preserving existing dashboard")
    metrics = pd.read_csv(run_dir/"test/strategy_comparison.csv")
    daily = pd.read_csv(run_dir/"test/daily_equity.csv")
    coverage = pd.read_csv(run_dir/"input_coverage.csv")
    check = json.loads((run_dir/"verification.json").read_text())
    now = datetime.now(timezone.utc)
    paper = []
    for directory in sessions:
        directory = Path(directory)
        entry = {"session": directory.name, "alerts": [], "alert_source": None}
        status = directory/"status.json"
        if status.exists():
            entry.update(json.loads(status.read_text()))
            entry["status_age_seconds"] = now.timestamp()-status.stat().st_mtime
        else:
            entry["missing_status"] = True       # a requested session that is not exporting must be visible
        for name in ("health.json", "monitor.json"):
            report = directory/name
            if report.exists():
                entry["alert_source"] = name
                entry["alerts"] = [a if isinstance(a, dict) else {"code": str(a), "severity": "warning", "detail": ""}
                                   for a in json.loads(report.read_text()).get("alerts", [])]
                entry["alerts_age_seconds"] = now.timestamp()-report.stat().st_mtime
                break
        paper.append(entry)
    # Recompute lineage for the artifacts this page was built from rather than echoing a saved flag.
    frozen_file = run_dir/"frozen_before_historical_test.json"
    lineage = {"generated_at_utc": now.isoformat(), "run": run_dir.name, "frozen_manifest_present": frozen_file.exists(),
               "artifact_ids": {name: hashlib.sha256((run_dir/name).read_bytes()).hexdigest()[:16]
                                for name in ("test/strategy_comparison.csv", "test/daily_equity.csv", "input_coverage.csv")}}
    if frozen_file.exists():
        expected = json.loads(frozen_file.read_text())["checkpoint_sha256"]
        mismatched = [name for name, value in expected.items()
                      if not (run_dir/name).exists() or hashlib.sha256((run_dir/name).read_bytes()).hexdigest() != value]
        lineage.update(checkpoints_checked=len(expected), checkpoints_matching=len(expected)-len(mismatched), mismatched=mismatched)
    test_sessions = int(pd.to_datetime(daily["trade_date"]).dt.date.nunique()) if "trade_date" in daily else None
    data = {"metrics": metrics.replace([np.inf, -np.inf], np.nan).astype(object).where(metrics.notna(), None).to_dict("records"),
            "daily": daily.to_dict("records"), "coverage": coverage.to_dict("records"), "verification": check, "paper": paper, "lineage": lineage, "meta": {"test_sessions": test_sessions}}
    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NIFTY hedge research</title><style>
body{font:16px system-ui;background:#101923;color:#e8edf2;max-width:1180px;margin:32px auto;padding:0 22px}h1{font-size:30px}h2{font-size:21px}.note{background:#233448;padding:18px;border-left:4px solid #e7b65e;border-radius:8px}section{margin:28px 0;padding:20px;background:#172432;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:right;padding:11px;border-bottom:1px solid #354351}td:first-child,th:first-child{text-align:left}select,button{padding:10px;background:#263d50;color:white;border:1px solid #698094;border-radius:5px}svg{width:100%;height:280px;background:#101923}a{color:#7ac9f3}.scroll{overflow:auto}.muted{color:#b8c5d0}.good{color:#7fd4a4}
</style><h1>NIFTY options · hedging research</h1><p class="muted">Historical comparison and forward paper-session monitor · offline report</p>
<p class="note">Retrospective research using configured execution and margin assumptions. ML hedging superiority and live profitability have not been established. Synthetic demonstrations are labelled separately from forward observations.</p>
<section><h2>Verification and input coverage</h2><div id="checks"></div><div id="coverage"></div></section>
<section><h2>Historical strategy comparison</h2><p>Same option positions, separate sequential accounts. Lower loss CVaR is better. Ratios annualized from <span id="nsessions">?</span> sessions are descriptive and unstable.</p><div class="scroll" id="metrics"></div><p><button id="download">Download metrics CSV</button></p></section>
<section><h2>Equity and drawdown</h2><label>Strategy <select id="strategy"></select></label> <label>Measure <select id="measure"><option value="equity">Equity (INR)</option><option value="drawdown_fraction">Daily drawdown fraction</option></select></label><p id="chartlabel"></p><svg id="chart" viewBox="0 0 1000 280" role="img" aria-label="Historical equity or drawdown chart"></svg></section>
<section><h2>Paper-session status</h2><p>Sessions require frozen contracts, completed-bar state and fresh bid/ask depth. A rejected liquidation leaves exposure open and requires reconciliation.</p><div id="paper"></div></section>
<script>const data=__DATA__;
const fmt=(x)=>typeof x==='number'?x.toLocaleString('en-IN',{maximumFractionDigits:3}):String(x??'—');
function table(id,rows,cols){const el=document.getElementById(id),t=document.createElement('table'),head=t.createTHead().insertRow();cols.forEach(c=>{const th=document.createElement('th');th.textContent=c;head.append(th)});const b=t.createTBody();rows.forEach(r=>{let tr=b.insertRow();cols.forEach(c=>{tr.insertCell().textContent=fmt(r[c])})});el.replaceChildren(t)}
document.getElementById('nsessions').textContent=data.meta.test_sessions??'unknown';
const L=data.lineage;document.getElementById('checks').textContent=`Saved verification file (as of its creation, not re-run): ${data.verification.tests_passed} tests passed. Lineage recomputed at build ${L.generated_at_utc}: ${L.frozen_manifest_present?`${L.checkpoints_matching}/${L.checkpoints_checked} frozen checkpoints match`:'no frozen manifest found'}${L.mismatched&&L.mismatched.length?' · MISMATCH: '+L.mismatched.join(', '):''} · run ${L.run} · artifacts ${Object.values(L.artifact_ids).join('/')} · forward superiority established: false`;
table('coverage',data.coverage,['split','decision_rows','ml_available','valid_pricing_rows']);
table('metrics',data.metrics,['strategy','episodes','net_pnl','fees','loss_cvar_95','daily_sharpe_annualized','max_daily_drawdown_fraction']);
const selector=document.getElementById('strategy');data.metrics.forEach(r=>{const o=document.createElement('option');o.value=r.strategy;o.textContent=r.strategy;selector.append(o)});
function chart(){const field=document.getElementById('measure').value,rows=data.daily.filter(r=>r.strategy===selector.value),v=rows.map(r=>r[field]);let lo=Math.min(...v),hi=Math.max(...v);if(hi===lo)hi=lo+1;const points=v.map((n,i)=>`${60+i*900/Math.max(v.length-1,1)},${240-(n-lo)*200/(hi-lo)}`).join(' ');document.getElementById('chart').innerHTML='<polyline fill="none" stroke="#72c9ec" stroke-width="3" points="'+points+'"/>';document.getElementById('chartlabel').textContent=`${rows[0]?.trade_date} to ${rows.at(-1)?.trade_date} · range ${fmt(lo)} to ${fmt(hi)} · ${rows.length} sessions`;}
selector.onchange=chart;document.getElementById('measure').onchange=chart;chart();
const paper=document.getElementById('paper');if(!data.paper.length)paper.textContent='No paper sessions requested. No forward performance results are available.';else data.paper.forEach(r=>{let p=document.createElement('p');if(r.missing_status){p.textContent=`${r.session} · NO STATUS FILE: session missing or not exporting`;paper.append(p);}else{p.textContent=`${r.session} · ${r.mode} · ${r.strategy} · accepted ${r.accepted_events}, rejected ${r.rejected_events} · equity P&L is not a live return · open option units ${r.state.q}, futures units ${r.state.h} · halted ${r.state.halted} · status age ${Math.round(r.status_age_seconds)}s`;paper.append(p);}(r.alerts||[]).forEach(a=>{const q=document.createElement('p');q.className='note';q.textContent=`ALERT [${a.severity}] ${a.code} ${a.detail||''} (${r.alert_source}, ${Math.round(r.alerts_age_seconds)}s old)`;paper.append(q)});if(!r.alert_source){const q=document.createElement('p');q.className='note';q.textContent='No health/monitor report for this session: run the monitor or service cycle.';paper.append(q)}});
document.getElementById('download').onclick=()=>{const cols=Object.keys(data.metrics[0]);const csv=[cols,...data.metrics.map(r=>cols.map(c=>r[c]))].map(row=>row.map(x=>JSON.stringify(x??'')).join(',')).join('\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='hedging_metrics.csv';a.click();URL.revokeObjectURL(a.href)};
</script></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__DATA__", payload), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-session", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    print(build(args.run_dir, args.output, args.paper_session))
