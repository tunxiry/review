import json
rows = {}
for line in open("results.jsonl"):
    line=line.strip()
    if not line: continue
    o=json.loads(line)
    rows[o['ticker']]=o
order=["SPY","SOXX","MSTR"]
# latest date for title
dates=[o['date'] for o in rows.values() if 'date' in o]
latest=max(dates)
import datetime
dt=datetime.date.fromisoformat(latest)
title_md=f"{dt.month}/{dt.day}"

def sc(x):
    return '+%d'%x if x>0 else str(x)
def rsi_word(r):
    return "극과매도" if r<=25 else "과매도" if r<=32 else "약과매도" if r<=42 else "중립" if r<=57 else "약과매수" if r<=67 else "과매수" if r<=74 else "극과매수"
def bb_word(pb):
    return "하단이탈" if pb<=0.05 else "하단근접" if pb<=0.25 else "중단권" if pb<0.75 else "상단이탈" if pb>=0.95 else "상단근접"
def dev_word(dev):
    return "20일선 대폭이탈" if dev<=-15 else "20일선 아래 과대" if dev<=-7 else "20일선 부근" if dev<7 else "20일선 위 과열" if dev<15 else "20일선 대폭초과"
def p52_word(p):
    return "1년 최저권" if p<=10 else "하단권" if p<=25 else "중하단권" if p<45 else "중앙권" if p<55 else "중상단권" if p<75 else "상단권" if p<90 else "1년 최고권"

def rsi_imp(s):
    return {-3:"강한바닥권",-2:"반등여력",-1:"반등여력",0:"방향 혼조",1:"상승우위",2:"과열조짐",3:"과열심화"}[s]
def bb_imp(s):
    return {-2:"낙폭과대",-1:"과대낙폭",0:"방향중립",1:"상단부담",2:"과열이탈"}[s]
def vol_word(o):
    vr=o['vratio']; s=o['vol_s']
    if vr>=130:
        if s==-2: return "투매급증, 바닥신호"
        if s==2: return "과열급증, 고점경계"
        return "급증하나 방향약"
    return "평이, 신호미약"

lines=[]
lines.append(f"📊 기술적 분석 ({title_md})")
for t in order:
    o=rows.get(t)
    lines.append("")
    if o is None or 'error' in o:
        lines.append(f"⚠️ {t} | 데이터 조회 실패")
        continue
    head=f"{o['emoji']} {t} | {o['verdict']} ({sc(o['total'])}) | {o['close']:.2f}"
    if len(head)>28:
        head=f"{o['emoji']} {t}|{o['verdict']} ({sc(o['total'])})|{o['close']:.2f}"
    lines.append(head)
    lines.append(f"- RSI {o['rsi']} {rsi_word(o['rsi'])}, {rsi_imp(o['rsi_s'])} ({sc(o['rsi_s'])})")
    lines.append(f"- 볼린저밴드 {bb_word(o['pb'])}, {bb_imp(o['bb_s'])} ({sc(o['bb_s'])})")
    lines.append(f"- 이격도 {o['dev']}% {dev_word(o['dev'])} ({sc(o['dev_s'])})")
    lines.append(f"- 52주 {o['pos52']}% {p52_word(o['pos52'])} ({sc(o['p52_s'])})")
    lines.append(f"- 거래량 {o['vratio']}% {vol_word(o)} ({sc(o['vol_s'])})")
    if o['div_s']!=0:
        dw="강세" if o['div_s']<0 else "약세"
        imp="저점신호" if o['div_s']<0 else "고점신호"
        lines.append(f"- 다이버전스 {dw}, {imp} ({sc(o['div_s'])})")
    if o['pattern_s']!=0:
        # pattern name from desc first token
        pname=o['pattern_desc'].split(" ")[0] if o['pattern_desc'] else "패턴"
        neck="넥라인 돌파" if o['pattern_confirmed'] else "넥라인 미돌파"
        lines.append(f"- {pname} {neck} ({sc(o['pattern_s'])})")
    if o['candle_s']!=0:
        nm="모닝스타" if o['candle_s']<0 else "이브닝스타"
        imp="상승반전" if o['candle_s']<0 else "하락반전"
        lines.append(f"- {nm}, {imp} ({sc(o['candle_s'])})")
    if o.get('risk_desc') and o.get('target') is not None:
        lines.append(f"⚠️ 목표 {o['target']} ({o['gap']}% 여지)")
    if o.get('verdict_note'):
        lines.append(f"↳ {o['verdict_note']}")

msg="\n".join(lines)
print(msg)
print("\n===== LINE LENGTH CHECK =====")
for ln in lines:
    mark="  <<OVER" if len(ln)>28 else ""
    print(f"{len(ln):2d} | {ln}{mark}")
open("message.txt","w").write(msg)
