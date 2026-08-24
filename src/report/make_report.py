"""세션 결과 HTML 보고서 생성기.

숫자는 전부 results/ 아래 결과 파일에서만 읽는다 (CLAUDE.md 절대 규칙 4).
사람이 쓰는 서술(오늘 무엇을 했는가)만 세션 JSON에서 읽는다.

실행: python -m src.report.make_report --session results/session_2026-08-24.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def iqm(x) -> float:
    """사분위평균 — 위아래 25%를 버린 가운데 50%의 평균."""
    x = np.sort(np.asarray(x, dtype=float))
    lo, hi = int(np.floor(len(x) * 0.25)), int(np.ceil(len(x) * 0.75))
    return float(np.mean(x[lo:hi]))


def iqm_ci(x, n_boot: int = 10000, seed: int = 0):
    """IQM의 95% 부트스트랩 신뢰구간 (에피소드 재표집).

    주의: 고정 규칙은 학습이 없어 '시드 = 에피소드'다. 학습 에이전트에서는
    시드 단위 계층 부트스트랩(rliable)으로 바꿔야 한다 (CLAUDE.md 절대 규칙 1).
    """
    x = np.asarray(x, dtype=float)
    if float(np.ptp(x)) == 0.0:  # 모든 값이 같으면 구간도 한 점
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    boots = np.array([iqm(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_episodes(csv_rel: str) -> dict:
    with (ROOT / csv_rel).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_lambda_map(curves: dict, lams) -> str:
    """λ-성능 지도 SVG (x = λ, y = 비용 차감 후 점수 IQM)."""
    W, H, PL, PR, PT, PB = 720, 380, 74, 180, 22, 48
    ys = [v for c in curves.values() for v in c]
    ymin, ymax = min(ys), max(ys)
    pad = (ymax - ymin) * 0.08 or 1.0
    ymin, ymax = ymin - pad, ymax + pad
    xmin, xmax = min(lams), max(lams)

    def X(v):
        return PL + (v - xmin) / (xmax - xmin) * (W - PL - PR)

    def Y(v):
        return PT + (ymax - v) / (ymax - ymin) * (H - PT - PB)

    colors = ["#c0392b", "#2c6fbb", "#7f8c8d", "#95a5a6", "#a9b0b6", "#5d6d7e"]
    mid = (PT + H - PB) / 2
    p = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" role="img" '
         'aria-label="람다에 따른 고정 규칙 성능 지도">' % (W, H)]
    for i in range(5):
        v = ymin + (ymax - ymin) * i / 4
        p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="var(--grid)" stroke-width="1"/>'
                 % (PL, Y(v), W - PR, Y(v)))
        p.append('<text x="%d" y="%.1f" text-anchor="end" font-size="11" fill="var(--muted)">%.0f</text>'
                 % (PL - 8, Y(v) + 4, v))
    for lam in lams:
        p.append('<text x="%.1f" y="%d" text-anchor="middle" font-size="11" fill="var(--muted)">%g</text>'
                 % (X(lam), H - PB + 18, lam))
    p.append('<text x="%.0f" y="%d" text-anchor="middle" font-size="12" fill="var(--muted)">'
             '행동 비용 &#955; (행동 1회당 빼는 점수)</text>' % ((PL + W - PR) / 2, H - 10))
    p.append('<text x="16" y="%.0f" font-size="12" fill="var(--muted)" text-anchor="middle" '
             'transform="rotate(-90 16 %.0f)">비용 차감 후 점수 (IQM)</text>' % (mid, mid))
    for i, (name, vals) in enumerate(curves.items()):
        color = colors[i % len(colors)]
        pts = " ".join("%.1f,%.1f" % (X(lam), Y(v)) for lam, v in zip(lams, vals))
        width = 3 if name in ("pump", "noop") else 1.6
        p.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="%s"/>' % (pts, color, width))
        p.append('<text x="%d" y="%.1f" font-size="12" fill="%s">%s</text>'
                 % (W - PR + 8, Y(vals[-1]) + 4, color, esc(name)))
    p.append("</svg>")
    return "".join(p)


CSS = """
:root{--bg:#ffffff;--fg:#1b1f24;--muted:#6a737d;--line:#e3e6ea;--grid:#eef1f4;
--card:#f7f9fb;--accent:#c0392b;--ok:#1f7a4d;--warn:#8a6100}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
--bg:#14171a;--fg:#e6e9ec;--muted:#9aa4ae;--line:#2a3038;--grid:#232830;
--card:#1b1f25;--accent:#ff7a6b;--ok:#4ec98a;--warn:#e0b355}}
:root[data-theme="dark"]{--bg:#14171a;--fg:#e6e9ec;--muted:#9aa4ae;--line:#2a3038;
--grid:#232830;--card:#1b1f25;--accent:#ff7a6b;--ok:#4ec98a;--warn:#e0b355}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font-family:"Pretendard","Malgun Gothic",system-ui,-apple-system,sans-serif;
line-height:1.75;font-size:16px}
.wrap{max-width:880px;margin:0 auto;padding:40px 20px 80px}
h1{font-size:27px;line-height:1.35;margin:0 0 6px;letter-spacing:-0.01em}
h2{font-size:19px;margin:46px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.date{color:var(--muted);font-size:14px;margin-bottom:26px}
.lead{background:var(--card);border-left:4px solid var(--accent);
padding:16px 18px;border-radius:0 8px 8px 0;margin:22px 0}
.lead b{color:var(--accent)}
.note{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:14px 16px;margin:16px 0;font-size:15px}
table{width:100%;border-collapse:collapse;font-size:14px;margin:12px 0}
.scroll{overflow-x:auto}
th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:13px;white-space:nowrap}
td.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
th.num{text-align:right}
tr.win td{background:rgba(31,122,77,0.12);font-weight:600}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:13px}
.muted{color:var(--muted)}
.ok{color:var(--ok);font-weight:600}
.bad{color:var(--accent);font-weight:600}
.warn{color:var(--warn);font-weight:600}
ul{padding-left:20px}li{margin:6px 0}
figure{margin:18px 0;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:14px}
figcaption{font-size:13px;color:var(--muted);margin-top:8px}
svg{width:100%;height:auto;display:block}
footer{margin-top:56px;padding-top:16px;border-top:1px solid var(--line);
font-size:13px;color:var(--muted);line-height:1.8}
code{background:var(--card);padding:2px 5px;border-radius:4px;font-size:13px;
font-family:ui-monospace,Consolas,monospace}
"""


def kv_table(rows, headers, num_cols=()) -> str:
    head = "<tr>" + "".join(
        '<th class="num">%s</th>' % esc(h) if i in num_cols else "<th>%s</th>" % esc(h)
        for i, h in enumerate(headers)) + "</tr>"
    body = ""
    for row in rows:
        body += "<tr>" + "".join(
            '<td class="num">%s</td>' % c if i in num_cols else "<td>%s</td>" % c
            for i, c in enumerate(row)) + "</tr>"
    return "<table>%s%s</table>" % (head, body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    args = ap.parse_args()

    S = json.loads(Path(args.session).read_text(encoding="utf-8"))
    summary = json.loads((ROOT / S["summary_json"]).read_text(encoding="utf-8"))
    eps = {r["policy"]: load_episodes(r["csv"]) for r in summary}

    # λ 재계산: 고정 규칙은 λ와 무관하게 같은 행동을 내므로
    # r'(λ) = r − λ×행동횟수 로 정확히 다시 계산할 수 있다 (근사가 아님).
    lam_grid = [0.0, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
    curves = {pid: [iqm(d["raw_return"] - lam * d["n_actions"]) for lam in lam_grid]
              for pid, d in eps.items()}

    cross = None
    if "pump" in curves and "noop" in curves:
        diff = [a - b for a, b in zip(curves["pump"], curves["noop"])]
        for i in range(len(lam_grid) - 1):
            if diff[i] > 0 >= diff[i + 1]:
                t = diff[i] / (diff[i] - diff[i + 1])
                cross = lam_grid[i] + t * (lam_grid[i + 1] - lam_grid[i])
                break

    rows = []
    for r in summary:
        lo, hi = iqm_ci(eps[r["policy"]]["raw_return"])
        rows.append(dict(r, ci_lo=lo, ci_hi=hi))
    best = max(rows, key=lambda r: r["raw_return_iqm"])

    res_head = ("<tr><th>규칙</th><th class='num'>원래 점수 r (IQM)</th><th class='num'>95% 신뢰구간</th>"
                "<th class='num'>행동 횟수(평균)</th><th class='num'>에피소드 길이</th>"
                "<th class='num'>목표 도달률</th></tr>")
    res_body = ""
    for r in rows:
        cls = ' class="win"' if r is best else ""
        res_body += ('<tr%s><td class="mono">%s</td><td class="num">%.2f</td>'
                     '<td class="num muted">[%.2f, %.2f]</td><td class="num">%.1f</td>'
                     '<td class="num">%.1f</td><td class="num">%.0f%%</td></tr>'
                     % (cls, esc(r["policy"]), r["raw_return_iqm"], r["ci_lo"], r["ci_hi"],
                        r["n_actions_mean"], r["steps_mean"], r["solved_rate"] * 100))
    res_table = "<table>%s%s</table>" % (res_head, res_body)

    lam_table = kv_table(
        [['<span class="mono">%s</span>' % esc(p)] + ["%.1f" % v for v in vals]
         for p, vals in curves.items()],
        ["규칙"] + ["λ=%g" % lam for lam in lam_grid],
        num_cols=tuple(range(1, len(lam_grid) + 1)))

    cross_txt = "%.3f" % cross if cross is not None else "격자 밖"
    lambda_read = S["lambda_read"].replace("{cross}", cross_txt)

    def li(items):
        return "<ul>" + "".join("<li>%s</li>" % x for x in items) + "</ul>"

    html = """<title>%s</title>
<style>%s</style>
<div class="wrap">
<h1>%s</h1>
<div class="date">%s · when-to-act-rl · 1주차 (기준선과 배관)</div>

<div class="lead">%s</div>

<h2>1. 오늘 한 일</h2>
%s

<h2>2. 환경 세팅</h2>
%s
<p class="muted">전체 설치 로그: <code>%s</code></p>

<h2>3. 비용 래퍼 λ=0 자가 점검</h2>
<p>비용 래퍼는 이 연구의 심장이다. 점수를 <code>r' = r − λ × (행동한 횟수)</code>로 바꾸는데,
<b>λ=0이면 원래 게임과 한 치도 달라지면 안 된다.</b> 그것부터 확인했다.</p>
%s

<h2>4. 스모크 테스트 — MountainCar, 고정 규칙, λ=0</h2>
<p>%s</p>
<div class="scroll">%s</div>
<p class="muted">IQM = 위아래 25%%를 버린 가운데 50%%의 평균(운 좋은/나쁜 판에 덜 휘둘림).
신뢰구간은 에피소드 %d개를 다시 뽑아(부트스트랩 10,000회) 구한 IQM의 95%% 구간이다.
값이 전부 같으면(−200) 구간도 한 점이 된다.</p>
%s

<h2>5. 덤 — λ를 올리면 어떻게 되는가 (고정 규칙만)</h2>
<p>고정 규칙은 <b>λ가 얼마든 똑같은 행동을 낸다.</b> 그래서 위에서 기록한 에피소드를
다시 계산하는 것만으로 λ별 점수를 <b>정확히</b> 얻을 수 있다 (추측이 아니라 재계산).</p>
<figure>%s
<figcaption>λ-성능 지도 v0 — 고정 규칙만. 학습 에이전트(DQN·TempoRL·Lazy-MDP)는 아직 없다.</figcaption></figure>
<div class="scroll">%s</div>
%s

<h2>6. 깃허브</h2>
%s

<h2>7. 다음에 할 일</h2>
%s

<h2>8. 확인 필요 (아직 모르는 것)</h2>
%s

<footer>
숫자 출처: <code>%s</code> · <code>results/raw/MountainCar-v0/&lt;규칙&gt;/lam0.0/episodes.csv</code><br>
이 보고서는 <code>src/report/make_report.py</code>가 위 파일들을 읽어 자동 생성했다 — 손으로 옮겨 적은 숫자는 없다.<br>
%s
</footer>
</div>""" % (
        esc(S["title"]), CSS, esc(S["title"]), esc(S["date"]), S["lead"],
        kv_table([[esc(a), b, esc(c)] for a, b, c in S["todo"]], ["할 일", "결과", "비고"]),
        kv_table([['<span class="mono">%s</span>' % esc(a), b, esc(c)] for a, b, c in S["setup"]],
                 ["항목", "결과", "비고"]),
        esc(S["install_log"]),
        kv_table([[esc(a), b] for a, b in S["selfcheck"]], ["점검 내용", "결과"]),
        S["smoke_intro"], res_table, rows[0]["n_episodes"], S["smoke_read"],
        svg_lambda_map(curves, lam_grid), lam_table, lambda_read,
        S["github"], li(S["next"]), li(S["unknown"]),
        esc(S["summary_json"]), esc(S["footer"]),
    )

    out = ROOT / "results" / "reports" / S["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print("보고서 생성: %s" % out)
    if cross is not None:
        print("(참고) 임계값 규칙이 무행동에 따라잡히는 λ ≈ %.3f" % cross)


if __name__ == "__main__":
    main()
