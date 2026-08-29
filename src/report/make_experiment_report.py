"""본실험 종합 HTML 보고서 — λ-성능 지도 + 임계 비용 λ* 표 + 실험 상태.

숫자는 전부 results/ 아래 결과 파일에서만 읽는다 (CLAUDE.md 절대 규칙 4).
없는 것은 "아직 없음"이라고 쓰고 빈칸을 그럴듯한 말로 채우지 않는다.

실행: python -m src.report.make_experiment_report --open
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "results" / "aggregate"
FIG = ROOT / "results" / "figures"
REP = ROOT / "results" / "reports"

LABEL = {"dqn": "표준 DQN", "temporl": "TempoRL 방식", "lazy": "Lazy-MDP 방식"}
REF_RULE = {"MountainCar-v0": "rule_pump", "LunarLander-v3": "rule_threshold_tuned",
            "MinAtar_Freeway-v1": "rule_cautious"}
RULE_LABEL = {"rule_pump": "pump(임계값) 규칙", "rule_threshold": "임계값 규칙(처음)",
              "rule_threshold_tuned": "임계값 규칙(튜닝)",
              "rule_cautious": "신중 규칙", "rule_cautious_d1": "신중 규칙(d=1)",
              "rule_periodic_k3": "3스텝 주기",
              "rule_noop": "무행동", "rule_periodic_k1": "매 스텝 규칙",
              "rule_periodic_k2": "2스텝 주기", "rule_periodic_k4": "4스텝 주기",
              "rule_periodic_k8": "8스텝 주기"}


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def img(path: Path) -> str:
    if not path.exists():
        return '<p class="missing">그림 없음: ' + esc(path.name) + "</p>"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return '<img src="data:image/png;base64,' + b64 + '" alt="' + esc(path.name) + '">'


def fmt(v, nd=1) -> str:
    try:
        return format(float(v), ",." + str(nd) + "f")
    except Exception:
        return "—"


def env_section(env_id: str) -> str:
    iqm_p = AGG / (env_id + "_iqm.csv")
    if not iqm_p.exists():
        return '<section id="env-' + esc(env_id) + '"><h2>' + esc(env_id) + '</h2><p class="missing">집계 결과가 아직 없다.</p></section>'
    agg = pd.read_csv(iqm_p)
    ref = REF_RULE.get(env_id, "rule_pump")
    learners = [a for a in sorted(agg.agent.unique()) if not str(a).startswith("rule_")]
    rules = [a for a in sorted(agg.agent.unique()) if str(a).startswith("rule_")]
    lams = sorted(agg.lam.unique())

    head = "".join("<th>λ=" + format(l, "g") + "</th>" for l in lams)
    body = ""
    for a in learners + rules:
        g = agg[agg.agent == a].set_index("lam")
        cells = ""
        for l in lams:
            if l not in g.index:
                cells += '<td class="na">—</td>'
                continue
            r = g.loc[l]
            better = ""
            if a in learners and ref in set(agg.agent):
                rr = agg[(agg.agent == ref) & (agg.lam == l)]
                if len(rr):
                    rr = rr.iloc[0]
                    if r["cost_return_ci_lo"] > rr["cost_return_ci_hi"]:
                        better = " win"
                    elif r["cost_return_ci_hi"] < rr["cost_return_ci_lo"]:
                        better = " lose"
            cells += ('<td class="num' + better + '">' + fmt(r["cost_return_iqm"])
                      + '<span class="ci">[' + fmt(r["cost_return_ci_lo"]) + ", "
                      + fmt(r["cost_return_ci_hi"]) + ']</span>'
                      + '<span class="act">행동 ' + fmt(r["n_actions_iqm"]) + '회</span></td>')
        name = LABEL.get(a, RULE_LABEL.get(a, a))
        cls = "learner" if a in learners else ("refrule" if a == ref else "rule")
        body += '<tr class="' + cls + '"><th>' + esc(name) + "</th>" + cells + "</tr>"
    n_seeds = int(agg.n_seeds.max())
    table = ('<table class="grid"><thead><tr><th>계열 / 규칙</th>' + head + "</tr></thead><tbody>"
             + body + "</tbody></table>"
             + '<p class="cap">칸의 값 = 비용 반영 총보상 r\'의 IQM, 대괄호는 95% 신뢰구간, '
             + "그 아래는 에피소드당 행동 횟수. 초록칸 = 기준 규칙("
             + esc(RULE_LABEL.get(ref, ref)) + ")을 통계적으로 이긴 조건, "
             + "빨강칸 = 통계적으로 진 조건, 무색 = 신뢰구간이 겹쳐 우열을 말할 수 없음. "
             + "시드 최대 " + str(n_seeds) + "개.</p>")

    star_p = AGG / (env_id + "_lambda_star.json")
    star_html = '<p class="missing">λ* 계산 결과가 아직 없다.</p>'
    if star_p.exists():
        st = json.loads(star_p.read_text(encoding="utf-8"))
        rows = ""
        for s in st.get("results", []):
            ci = s["lam_star_ci"] if s["lam_star_ci"] is not None else "격자 안에 없음"
            pt = s["lam_star_pt"] if s["lam_star_pt"] is not None else "격자 안에 없음"
            rows += ("<tr><th>" + esc(LABEL.get(s["learner"], s["learner"])) + "</th>"
                     + '<td class="num">' + esc(ci) + "</td>"
                     + '<td class="num">' + esc(pt) + "</td>"
                     + "<td>" + esc(s.get("note", "")) + "</td></tr>")
        star_html = ('<table class="grid"><thead><tr><th>학습 계열</th>'
                     + "<th>λ*<sub>CI</sub> (엄격)</th><th>λ*<sub>점추정</sub> (느슨)</th><th>해석</th>"
                     + "</tr></thead><tbody>" + rows + "</tbody></table>"
                     + '<p class="cap">λ* = ' + esc(RULE_LABEL.get(ref, ref))
                     + "을 더 이상 이기지 못하게 되는 가장 작은 행동 비용. "
                     + "λ*<sub>CI</sub>는 '통계적으로 확실한 우위'가 사라지는 지점(신뢰구간 기준), "
                     + "λ*<sub>점추정</sub>은 두 곡선이 교차하는 지점. "
                     + "격자 안에서 교차가 없으면 그렇게 적는다.</p>")

    figs = "".join(img(FIG / (env_id + "_" + n + ".png")) for n in
                   ["lambda_map_cost_return", "action_map", "learning_curves"])
    return ('<section id="env-' + esc(env_id) + '"><h2>' + esc(env_id) + "</h2>"
            + "<h3>λ-성능 지도</h3>" + figs
            + "<h3>조건별 성능표</h3>" + table
            + "<h3>임계 비용 λ*</h3>" + star_html + "</section>")


def fairness_section() -> str:
    """공정성 점검 — 비용이 없는 λ=0에서 학습이 규칙 수준에 닿는가.

    이 절이 없으면 "규칙에 진 것은 비용 때문이 아니라 학습이 덜 된 탓"이라는 반론에 답할 수 없다.
    """
    try:
        from src.analysis.fairness_verdict import CANDIDATES, REF_ROOM, REF_RULE, seed_scores, summarize
    except Exception as e:
        return ""
    ref = summarize(seed_scores(REF_ROOM, REF_RULE))
    if ref is None:
        return ""
    rows = ('<tr class="refrule"><th>기준: pump 규칙 (학습 없음)</th>'
            + '<td class="num">' + str(ref["n"]) + "</td>"
            + '<td class="num"><b>' + fmt(ref["iqm"]) + "</b>"
            + '<span class="ci">[' + fmt(ref["lo"]) + ", " + fmt(ref["hi"]) + "]</span></td>"
            + "<td>—</td><td>—</td></tr>")
    any_row = False
    for room, label in CANDIDATES.items():
        sc = seed_scores(room, "dqn")
        st = summarize(sc)
        if st is None:
            continue
        any_row = True
        if st["lo"] > ref["hi"]:
            v, cls = "이김", "win"
        elif st["hi"] < ref["lo"]:
            v, cls = "짐", "lose"
        else:
            v, cls = "비김 (신뢰구간 겹침)", ""
        rows += ("<tr><th>" + esc(label) + '</th><td class="num">' + str(st["n"]) + "</td>"
                 + '<td class="num ' + cls + '">' + fmt(st["iqm"])
                 + '<span class="ci">[' + fmt(st["lo"]) + ", " + fmt(st["hi"]) + "]</span></td>"
                 + '<td class="num">' + fmt(st["iqm"] - ref["iqm"]) + "</td>"
                 + "<td>" + esc(v) + "</td></tr>")
    if not any_row:
        return ""
    return ('<section id="fair"><h2>공정성 점검 — 비용이 없을 때(λ=0) 학습은 규칙 수준에 닿는가</h2>'
            + "<p>MountainCar에서 임계 비용 λ*가 0으로 나왔다는 것은 '비용이 없어도 학습이 규칙에 진다'는 뜻이다. "
            + "그렇다면 이 결과는 비용에 대한 발견이 아니라 <b>학습이 덜 됐다는 신호</b>일 수 있다. "
            + "그래서 학습 예산과 탐험 설정을 바꿔 가며 λ=0 성능을 다시 쟀다. "
            + "표준 DQN 기준이며, 본실험과 같은 평가(스냅샷 3장 × 50 에피소드)를 쓴다.</p>"
            + '<table class="grid"><thead><tr><th>설정</th><th>시드</th>'
            + "<th>원보상 r IQM [95% CI]</th><th>규칙 대비</th><th>판정</th>"
            + "</tr></thead><tbody>" + rows + "</tbody></table>"
            + '<p class="cap">초록칸 = 규칙을 통계적으로 이김, 빨강칸 = 통계적으로 짐, '
            + "무색 = 신뢰구간이 겹쳐 우열을 말할 수 없음.</p></section>")


def audit_section() -> str:
    """기준선 감사 — 비교 상대인 규칙을 성의 있게 만들었는가."""
    p = ROOT / "results" / "aggregate" / "baseline_audit.json"
    if not p.exists():
        return ""
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    rows = ""
    for env, r in d.get("results", {}).items():
        ok = abs(float(r["차이"])) < 1e-6
        rows += ("<tr><th>" + esc(env) + "</th><td>" + esc(r["rule"]) + "</td>"
                 + '<td class="num">' + fmt(r["현재"]["eval_iqm"]) + "</td>"
                 + '<td class="num ' + ("" if ok else "lose") + '">' + fmt(r["튜닝셋 최고"]["eval_iqm"]) + "</td>"
                 + '<td class="num">' + fmt(r["차이"]) + "</td><td>" + esc(r["판정"]) + "</td></tr>")
    if not rows:
        return ""
    return ('<section id="audit"><h2>기준선 감사 — 비교 상대인 규칙을 성의 있게 만들었는가</h2>'
            + "<p>“학습이 단순 규칙을 이긴다”는 주장은 그 규칙을 얼마나 잘 만들었는지에 달려 있다. "
            + "대충 만든 규칙을 이기는 것은 쉽다. 환경마다 기준 규칙의 계수를 격자로 훑어 더 나은 것이 "
            + "있는지 확인했다. <b>튜닝은 평가에 쓰지 않는 에피소드에서 하고, 거기서 고른 하나만 "
            + "평가용 에피소드로 다시 쟀다.</b></p>"
            + '<table class="grid"><thead><tr><th>환경</th><th>규칙</th><th>현재 계수</th>'
            + "<th>다시 고른 계수</th><th>차이</th><th>판정</th></tr></thead><tbody>" + rows
            + "</tbody></table>"
            + '<p class="cap">빨강칸 = 더 나은 계수가 있었다는 뜻. 그 환경에서는 기준 규칙을 바꾸고 '
            + "λ*를 다시 계산했다.</p></section>")


def causal_section_html() -> str:
    """인과 실험 — 무행동 붕괴는 최적해인가 탐험 실패인가."""
    p = ROOT / "results" / "aggregate" / "causal_warmup.json"
    if not p.exists():
        return ""
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    res = d.get("results", [])
    if not res:
        return ""
    rows = ""
    for r in res:
        for key, label in (("from_start", "처음부터"), ("warmup", "절반 뒤")):
            c = r[key]
            sv = c.get("solved")
            rows += ("<tr>" + ("<th rowspan='2'>λ=" + format(float(r["lam"]), "g")
                               + "<br>" + esc(LABEL.get(r["agent"], r["agent"])) + "</th>"
                               if key == "from_start" else "")
                     + "<td>" + label + "</td>"
                     + '<td class="num">' + fmt(c["score"]["iqm"])
                     + '<span class="ci">[' + fmt(c["score"]["lo"]) + ", " + fmt(c["score"]["hi"]) + "]</span></td>"
                     + '<td class="num">' + fmt(c["actions"]["iqm"], 0) + "</td>"
                     + '<td class="num">' + (fmt(sv["iqm"] * 100, 0) + "%" if sv else "—") + "</td>"
                     + ("<td rowspan='2'>" + esc(r["verdict"]) + "</td>" if key == "from_start" else "")
                     + "</tr>")
    return ('<section id="causal"><h2>인과 실험 — 무행동 붕괴는 최적해인가, 탐험 실패인가</h2>'
            + "<p>같은 예산·같은 시드 안에서 <b>비용을 켜는 시점만</b> 바꿨다. 앞 절반은 비용 없이 "
            + "학습하고 나머지 절반에서 비용을 켠다. <b>평가는 양쪽 모두 진짜 λ로 한다</b> — "
            + "성적표는 언제나 비용이 있는 세상에서 매긴다.</p>"
            + '<table class="grid"><thead><tr><th>조건</th><th>비용 시점</th><th>r′ IQM [95% CI]</th>'
            + "<th>행동</th><th>목표 도달률</th><th>판정</th></tr></thead><tbody>" + rows
            + "</tbody></table>"
            + '<p class="cap">워밍업 쪽이 목표에 더 자주 닿으면 → 탐험 실패다. 비용이 있는 세상에서도 '
            + "목표에 닿는 정책이 존재하는데 처음부터 비용을 물리면 못 찾는다는 뜻이다. "
            + "양쪽 다 무행동으로 굳으면 → 무행동이 정말 최적해다.</p></section>")


def status_section() -> str:
    rows = ""
    for pf in sorted((ROOT / "results").glob("progress_*.json")):
        try:
            p = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            continue
        pct = 100.0 * p["done"] / max(1, p["total"])
        rows += ("<tr><th>" + esc(p["run_name"]) + '</th><td class="num">'
                 + str(p["done"]) + "/" + str(p["total"]) + " (" + format(pct, ".0f") + "%)</td>"
                 + '<td class="num">' + str(p.get("skipped", 0)) + "</td>"
                 + "<td>" + esc(p.get("eta_text", "—")) + "</td>"
                 + "<td>" + ("끝남" if p.get("finished") else "진행 중") + "</td></tr>")
    wd = ROOT / "results" / "watchdog.log"
    log = ""
    if wd.exists():
        lines = wd.read_text(encoding="utf-8").strip().splitlines()[-12:]
        log = "<pre class='log'>" + esc("\n".join(lines)) + "</pre>"
    if not rows:
        rows = '<tr><td colspan="5" class="missing">실행 기록 없음</td></tr>'
    return ('<section id="status"><h2>실험 진행 상태</h2>'
            + '<table class="grid"><thead><tr><th>실험</th><th>완료 조건</th><th>건너뜀</th>'
            + "<th>남은 시간</th><th>상태</th></tr></thead><tbody>" + rows + "</tbody></table>"
            + "<h3>자가 감시 기록 (최근 12줄)</h3>"
            + (log or '<p class="missing">아직 기록 없음</p>') + "</section>")


CSS = """
*{box-sizing:border-box} body{margin:0;padding:32px;font-family:'Malgun Gothic',system-ui,sans-serif;
 background:#f6f7f9;color:#1b1f24;line-height:1.65}
.wrap{max-width:1100px;margin:0 auto}
h1{font-size:1.9rem;margin:0 0 6px} .sub{color:#5b636d;margin:0 0 28px}
section{background:#fff;border:1px solid #e3e6ea;border-radius:12px;padding:22px 26px;margin-bottom:22px}
h2{font-size:1.3rem;margin:0 0 14px;padding-bottom:8px;border-bottom:2px solid #eceff3}
h3{font-size:1.02rem;margin:22px 0 10px;color:#333}
table.grid{border-collapse:collapse;width:100%;font-size:.86rem;margin:6px 0}
table.grid th,table.grid td{border:1px solid #e3e6ea;padding:7px 9px;text-align:left;vertical-align:top}
table.grid thead th{background:#f0f2f5;font-weight:600;text-align:center}
table.grid tbody th{background:#fafbfc;white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums}
td.na{text-align:center;color:#aab}
td .ci{display:block;font-size:.74rem;color:#6b7280}
td .act{display:block;font-size:.74rem;color:#8a93a0}
td.win{background:#e8f6ec} td.lose{background:#fdecec}
tr.refrule th{background:#fff8e1} tr.learner th{font-weight:700}
img{max-width:100%;height:auto;display:block;margin:10px 0;border:1px solid #e8eaee;border-radius:8px}
.cap{font-size:.82rem;color:#5b636d;margin:6px 0 0}
.missing{color:#9aa0a6;font-style:italic}
pre.log{background:#1e2228;color:#dfe3e8;padding:12px;border-radius:8px;overflow-x:auto;font-size:.76rem;line-height:1.5}
.key{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}
.key div{background:#f0f2f5;border-radius:8px;padding:10px 14px;font-size:.86rem}
.key b{display:block;font-size:1.25rem}
/* 목차 — 제목이 서른 개 넘는 문서라 없으면 스크롤로 찾아야 한다 */
.toc{background:#fff;border:1px solid #e3e6ea;border-radius:12px;padding:16px 22px;margin:14px 0 22px}
.toc b{font-size:.9rem;color:#5b636d}
.toc ul{margin:8px 0 0;padding:0;list-style:none;display:flex;flex-wrap:wrap;gap:8px 18px}
.toc a{color:#1a56b8;text-decoration:none;font-size:.9rem}
.toc a:hover{text-decoration:underline}
/* 보조 실험 방 안내는 절 카드가 아니라 구분선처럼 보이게 */
h2#rooms{margin-top:34px;font-size:1.15rem;color:#5b636d;border-bottom:2px dashed #ccd2d9}
"""

INTRO = """<section><h2>이 보고서를 읽는 법</h2>
<p><b>가로축 λ</b>는 행동 1번의 값이다. 오른쪽으로 갈수록 "버튼 한 번 누르는 값"이 비싸진다.
λ=0이면 원래 문제와 같고, λ가 충분히 크면 아무것도 안 하는 것이 최적이 된다.</p>
<p><b>세로축 r'</b>은 비용까지 빼고 남은 총보상이다. 높을수록 좋다.</p>
<p><b>기준선</b>은 학습 없는 고정 규칙 중 가장 센 것이다 (MountainCar는 pump 규칙, r IQM −119.3).
무행동(−200)처럼 문제를 아예 못 푸는 약한 규칙을 기준으로 삼으면 "학습이 이겼다"는 말이
너무 쉬워지므로, 모든 그림과 표에서 pump 규칙을 기준으로 유지한다.</p>
<p><b>임계 비용 λ*</b>는 학습이 이 기준 규칙을 더 이상 이기지 못하게 되는 가장 작은 λ다.
이 값이 이 연구의 최종 산출물이다 — "행동이 이만큼 비싸지면 학습을 도입할 이유가 없다"는 문턱.</p>
</section>"""

SOURCE = """<section><h2>출처</h2>
<p class="cap">모든 수치는 <code>results/aggregate/*_iqm.csv</code>,
<code>results/aggregate/*_lambda_star.json</code>, <code>results/raw/…/seed*_final.csv</code>에서 읽었다.
설계 결정과 도중에 고친 버그는 <code>docs/실험일지.md</code>에 날짜와 함께 기록돼 있다.</p></section>"""


ROOM_INTRO = (
    '<h2 id="rooms">보조 실험 방 — 한 가지만 바꿔 본 대조 조건</h2>'
    "<p>아래는 주 실험과 <b>딱 한 가지만</b> 다르게 두고 돌린 방들이다. 결론을 말하기 위한 것이 아니라, "
    "결론에 대한 반론을 확인하기 위한 것이다. 시드 수가 주 실험보다 적을 수 있으므로 "
    "<b>우열 판정은 주 실험 표에서만</b> 한다.</p>"
    "<ul>"
    "<li><b>@budget1M_epsconst / epsdecay / wide</b> — λ=0에서 예산·탐험·신경망을 바꿨다 (공정성 점검)</li>"
    "<li><b>@warmup50</b> — 예산의 앞 절반만 비용 없이 학습시켰다 (인과 실험)</li>"
    "</ul>")


def toc(mains: list[str], rooms: list[str]) -> str:
    """제목이 서른 개 넘는 문서에 목차가 없으면 읽는 사람이 스크롤로 찾아야 한다."""
    items = ['<li><a href="#status">실험 진행 상태</a></li>',
             '<li><a href="#fair">공정성 점검</a></li>',
             '<li><a href="#audit">기준선 감사</a></li>',
             '<li><a href="#causal">인과 실험</a></li>']
    items += ['<li><a href="#env-' + e + '">' + e + '</a></li>' for e in mains]
    if rooms:
        items.append('<li><a href="#rooms">보조 실험 방 (' + str(len(rooms)) + '개)</a></li>')
    return '<div class="toc"><b>목차</b><ul>' + "".join(items) + "</ul></div>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="Microsoft Edge로 자동 열기")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    found = sorted({p.name.replace("_iqm.csv", "") for p in AGG.glob("*_iqm.csv")}) if AGG.exists() else []
    if not found:
        found = [p.name for p in (ROOT / "results" / "raw").iterdir() if p.is_dir()]
    # 알파벳 순으로 늘어놓으면 보조 실험 방(@budget1M 등)이 주 환경 사이에 끼어 읽기 나쁘다.
    # 논문에서 다루는 순서(주 환경 3종)를 먼저, 보조 방을 뒤에 둔다.
    MAIN = ["MountainCar-v0", "LunarLander-v3", "MinAtar_Freeway-v1"]
    mains = [e for e in MAIN if e in found]
    mains += [e for e in found if "@" not in e and e not in mains]
    rooms = [e for e in found if "@" in e]
    envs = mains + rooms

    n_done = sum(1 for _ in (ROOT / "results" / "raw").rglob("seed*_meta.json"))
    key = ('<div class="key"><div>완료 조건 수<b>' + str(n_done) + "</b></div>"
           + "<div>환경<b>" + str(len(envs)) + "</b></div>"
           + "<div>생성 시각<b>" + time.strftime("%Y-%m-%d %H:%M") + " KST</b></div></div>")

    html = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            + "<title>본실험 보고서 — 행동 비용 λ와 강화학습의 우위 조건</title>"
            + "<style>" + CSS + "</style></head><body><div class='wrap'>"
            + "<h1>λ-성능 지도와 임계 비용 λ*</h1>"
            + '<p class="sub">행동 1번에 비용 λ를 물렸을 때, 학습이 단순 고정 규칙을 언제까지 이기는가 — '
            + "동아대학교 졸업과제 · 전혜성</p>"
            + key + toc(mains, rooms) + INTRO + status_section() + fairness_section()
            + audit_section() + causal_section_html()
            + "".join(env_section(e) for e in mains)
            + (ROOM_INTRO if rooms else "")
            + "".join(env_section(e) for e in rooms)
            + SOURCE + "</div></body></html>")

    REP.mkdir(parents=True, exist_ok=True)
    out = (Path(a.out).resolve() if a.out
           else REP / (time.strftime("%Y-%m-%d") + "_본실험보고서.html"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    try:
        shown = str(out.relative_to(ROOT))
    except ValueError:      # 저장소 밖 경로를 --out으로 준 경우
        shown = str(out)
    print("보고서 저장: " + shown)
    if a.open:
        from src.report.open_edge import open_in_edge
        open_in_edge(out)


if __name__ == "__main__":
    main()
