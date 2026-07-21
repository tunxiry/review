import json, subprocess

rows=[]
out=subprocess.run(["python3","analyze.py"],capture_output=True,text=True,cwd=".")
for line in out.stdout.strip().split("\n"):
    rows.append(json.loads(line))

def sc(v):
    return "0" if v==0 else f"{v:+d}"

def rsi_state(r):
    return "극과매도" if r<=25 else "과매도" if r<=32 else "약과매도" if r<=42 else "중립" if r<=57 else "약과매수" if r<=67 else "과매수" if r<=74 else "극과매수"
def rsi_imp(v,r):
    if v<0: return "반등 여력" if v==-1 else "낙폭 과대"
    if v>0: return "과열 근접" if v==1 else "과매수 부담"
    return "방향 혼조"
def bb_pos(pb):
    return "하단이탈" if pb<=0.05 else "하단근접" if pb<=0.25 else "중단권" if pb<0.75 else "상단이탈" if pb>=0.95 else "상단근접"
def bb_imp(v):
    return "낙폭 과대" if v<0 else "과매수권" if v>0 else "중앙 위치"
def dev_expr(dev):
    return "20일선 대폭이탈" if dev<=-15 else "20일선 아래 과대" if dev<=-7 else "20일선 부근" if dev<7 else "20일선 위 과열" if dev<15 else "20일선 대폭초과"
def p52_zone(p):
    return "1년 최저권" if p<=10 else "하단권" if p<=25 else "중하단권" if p<=45 else "중앙권" if p<55 else "중상단권" if p<75 else "상단권" if p<90 else "1년 최고권"
def vol_txt(vr,v):
    if vr>=130:
        if v<0: return "투매 급증","바닥 신호"
        if v>0: return "과열 급증","고점 신호"
        return "급증","방향 혼조"
    return "평이","신호 미약"

def hdr(b):
    v=b['verdict']; e=b['emoji']; t=sc(b['total']); c=f"{b['close']:.2f}"
    line=f"{e} {b['ticker']} | {v} ({t}) | {c}"
    if width(line)>28:
        line=f"{e} {b['ticker']}|{v} ({t})|{c}"
    return line

def width(s):
    return len(s)

lines=["📊 기술적 분석 (7/21)",""]
order=["QQQ","SOXX","AAPL","MSTR","BTC"]
bym={b.get('ticker'):b for b in rows}
for tk in order:
    b=bym.get(tk)
    if b is None or 'error' in b:
        lines.append(f"⚠️ {tk} | 데이터 조회 실패"); lines.append(""); continue
    blk=[hdr(b)]
    blk.append(f"- RSI {b['rsi']} {rsi_state(b['rsi'])}, {rsi_imp(b['rsi_s'],b['rsi'])} ({sc(b['rsi_s'])})")
    blk.append(f"- 볼린저밴드 {bb_pos(b['pb'])}, {bb_imp(b['bb_s'])} ({sc(b['bb_s'])})")
    blk.append(f"- 이격도 {b['dev']}% {dev_expr(b['dev'])} ({sc(b['dev_s'])})")
    blk.append(f"- 52주 {b['pos52']}% {p52_zone(b['pos52'])} ({sc(b['p52_s'])})")
    vt,vi=vol_txt(b['vratio'],b['vol_s'])
    blk.append(f"- 거래량 {b['vratio']}% {vt}, {vi} ({sc(b['vol_s'])})")
    if b['div_s']!=0:
        ds="강세" if b['div_s']<0 else "약세"
        di="저점 신호" if b['div_s']<0 else "고점 신호"
        blk.append(f"- 다이버전스 {ds}, {di} ({sc(b['div_s'])})")
    if b['pattern_s']!=0:
        pn=b['pattern_desc'].split(" (")[0].strip()
        nk="넥라인 돌파" if b['pattern_broke'] else "넥라인 미돌파"
        blk.append(f"- {pn} {nk} ({sc(b['pattern_s'])})")
    if b['candle_s']!=0:
        cn="모닝스타" if b['candle_s']<0 else "이브닝스타"
        ci="상승 반전" if b['candle_s']<0 else "하락 반전"
        blk.append(f"- {cn}, {ci} ({sc(b['candle_s'])})")
    if b.get('risk_desc') and b.get('risk_flag'):
        blk.append(f"⚠️ 목표 {b['target']} ({b['gap']}% 여지)")
    if b.get('verdict_note'):
        blk.append(f"↳ {b['verdict_note']}")
    lines.extend(blk)
    lines.append("")

msg="\n".join(lines).rstrip()
print(msg)
print("\n=== WIDTH CHECK (>28) ===")
for ln in msg.split("\n"):
    if width(ln)>28:
        print(f"[{width(ln)}] {ln}")
open("message.txt","w").write(msg)
