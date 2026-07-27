import json
R=json.load(open("results.json"))
def sc(x):
    return "0" if x==0 else f"{x:+d}"

def rsi_state(r):
    return "극과매도" if r<=25 else "과매도" if r<=32 else "약과매도" if r<=42 else "중립" if r<=57 else "약과매수" if r<=67 else "과매수" if r<=74 else "극과매수"
def rsi_imp(s):
    return {-3:"강한 반등 기대",-2:"반등 여력",-1:"반등 소지",0:"방향 혼조",1:"과열 초기",2:"조정 주의",3:"조정 임박"}[s]
def bb_pos(pb):
    return "하단이탈" if pb<=0.05 else "하단근접" if pb<=0.25 else "중단권" if pb<0.75 else "상단이탈" if pb>=0.95 else "상단근접"
def bb_imp(s):
    return {-2:"낙폭과대",-1:"과대낙폭",0:"정상범위",1:"과열조짐",2:"급등과열"}[s]
def dev_lbl(dev):
    return "20일선 대폭이탈" if dev<=-15 else "20일선 아래 과대" if dev<=-7 else "20일선 부근" if dev<7 else "20일선 대폭초과" if dev>=15 else "20일선 위 과열"
def p52_lbl(p):
    return "1년 최저권" if p<=10 else "하단권" if p<=25 else "중하단권" if p<=45 else "중앙권" if p<55 else "중상단권" if p<75 else "상단권" if p<90 else "1년 최고권"
def vol_txt(s,vr):
    if s==-2: return "투매급증, 바닥신호"
    if s==2: return "매수급증, 과열신호"
    if vr>=130: return "급증, 방향중립"
    return "평이, 신호미약"

lines=[]
# 제목 날짜 = 가장 늦은 date
dates=[b["date"] for b in R if "date" in b]
md=max(dates)
y,m,dd=md.split("-")
title=f"📊 기술적 분석 ({int(m)}/{int(dd)})"
blocks=[]
for b in R:
    if "error" in b:
        blocks.append([f"⚠️ {b['ticker']} | 데이터 조회 실패"])
        continue
    L=[]
    hdr=f"{b['emoji']} {b['ticker']} | {b['verdict']} ({sc(b['total'])}) | {b['close']:.2f}"
    if len(hdr)>28:
        hdr=f"{b['emoji']} {b['ticker']}|{b['verdict']} ({sc(b['total'])})|{b['close']:.2f}"
    L.append(hdr)
    L.append(f"- RSI {b['rsi']} {rsi_state(b['rsi'])}, {rsi_imp(b['rsi_s'])} ({sc(b['rsi_s'])})")
    L.append(f"- 볼린저밴드 {bb_pos(b['pb'])}, {bb_imp(b['bb_s'])} ({sc(b['bb_s'])})")
    L.append(f"- 이격도 {b['dev']}% {dev_lbl(b['dev'])} ({sc(b['dev_s'])})")
    L.append(f"- 52주 {b['pos52']}% {p52_lbl(b['pos52'])} ({sc(b['p52_s'])})")
    L.append(f"- 거래량 {b['vratio']}% {vol_txt(b['vol_s'],b['vratio'])} ({sc(b['vol_s'])})")
    if b['div_s']!=0:
        dv="강세, 저점신호" if b['div_s']<0 else "약세, 고점신호"
        L.append(f"- 다이버전스 {dv} ({sc(b['div_s'])})")
    if b['pattern_s']!=0:
        nm=b['pattern_desc'].split(' (')[0]
        nk="넥라인 돌파" if b['pattern_broke'] else "넥라인 미돌파"
        L.append(f"- {nm} {nk} ({sc(b['pattern_s'])})")
    if b['candle_s']!=0:
        cs="모닝스타, 상승반전" if b['candle_s']<0 else "이브닝스타, 하락반전"
        L.append(f"- {cs} ({sc(b['candle_s'])})")
    if b.get('risk_flag'):
        L.append(f"⚠️ 목표 {b['target']} ({b['gap']}% 여지)")
    if b.get('verdict_adj'):
        L.append(f"↳ {b['verdict_adj']}")
    blocks.append(L)

msg=title+"\n\n"+"\n\n".join("\n".join(bl) for bl in blocks)
print(msg)
print("=====CHARCHECK=====")
for ln in msg.split("\n"):
    if len(ln)>28:
        print(f"[{len(ln)}] {ln}")
open("message.txt","w").write(msg)
