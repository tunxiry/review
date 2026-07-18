import json, pandas as pd
def analyze(t, data):
    df = pd.DataFrame(data)
    df['t'] = pd.to_datetime(df['t'])
    df = df.sort_values('t').reset_index(drop=True)
    df['close'] = df.apply(lambda r: r['a'] if ('a' in r and pd.notna(r['a'])) else r['c'], axis=1)
    df['h'] = df['h'].astype(float); df['l'] = df['l'].astype(float); df['o'] = df['o'].astype(float)
    df['ma20'] = df['close'].rolling(20).mean()
    sd = df['close'].rolling(20).std()
    df['bb_up'] = df['ma20'] + 2*sd
    df['bb_lo'] = df['ma20'] - 2*sd
    d = df['close'].diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al = l.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['rsi'] = 100 - 100/(1+ag/al)
    v20 = df['v'].rolling(20).mean()
    L, P = df.iloc[-1], df.iloc[-2]
    chg = (L.close-P.close)/P.close*100
    chg3 = (L.close-df['close'].iloc[-4])/df['close'].iloc[-4]*100
    vr = L.v/v20.iloc[-1] if v20.iloc[-1] else 1

    # 음수=저점방향, 0=중립, 양수=고점방향
    r = float(L.rsi)
    rsi_s = -3 if r<=25 else -2 if r<=32 else -1 if r<=42 else 3 if r>=75 else 2 if r>=68 else 1 if r>=58 else 0

    pb = (L.close-L.bb_lo)/(L.bb_up-L.bb_lo) if L.bb_up!=L.bb_lo else 0.5
    bb_s = -2 if pb<=0.05 else -1 if pb<=0.25 else 2 if pb>=0.95 else 1 if pb>=0.75 else 0

    dev = (L.close-L.ma20)/L.ma20*100
    dev_s = -2 if dev<=-15 else -1 if dev<=-7 else 2 if dev>=15 else 1 if dev>=7 else 0

    hi52 = df['close'].tail(252).max(); lo52 = df['close'].tail(252).min()
    pos52 = (L.close-lo52)/(hi52-lo52)*100 if hi52!=lo52 else 50
    p52_s = -2 if pos52<=10 else -1 if pos52<=25 else 2 if pos52>=90 else 1 if pos52>=75 else 0

    if vr>=1.3 and (chg<=-2 or chg3<=-5): vol_s=-2
    elif vr>=1.3 and (chg>=2 or chg3>=5): vol_s=2
    else: vol_s=0

    pos_score = rsi_s+bb_s+dev_s+p52_s
    base = dict(ticker=t, date=str(L.t.date()), close=round(float(L.close),2),
        rsi=round(r), rsi_s=rsi_s, pb=round(float(pb),2), bb_s=bb_s,
        dev=round(float(dev),1), dev_s=dev_s, pos52=round(float(pos52)), p52_s=p52_s,
        vratio=round(float(vr)*100), chg=round(float(chg),1), chg3=round(float(chg3),1), vol_s=vol_s,
        pos_score=pos_score)
    return base, df

# ---------------- [2단계-B] ----------------
def find_swings(df, lookback=90, k=3):
    n = len(df)
    hi = df['h'].values; lo = df['l'].values
    start = max(k, n-lookback)
    sh=[]; sl=[]
    for i in range(start, n-k):
        wl = range(i-k, i); wr = range(i+1, i+k+1)
        if all(hi[i] > hi[j] for j in wl) and all(hi[i] > hi[j] for j in wr):
            sh.append(i)
        if all(lo[i] < lo[j] for j in wl) and all(lo[i] < lo[j] for j in wr):
            sl.append(i)
    return sh, sl

def process(base, df):
    n = len(df)
    last = n-1
    rsi = df['rsi'].values; hi = df['h'].values; lo = df['l'].values; cl = df['close'].values
    sh, sl = find_swings(df)

    # 2) RSI 다이버전스
    div_s = 0; div_desc = None
    if len(sl) >= 2:
        a,b = sl[-2], sl[-1]
        if lo[b] < lo[a] and rsi[b] > rsi[a]:
            div_s = -2; div_desc = "RSI 강세 다이버전스, 저점 신호"
    if len(sh) >= 2:
        a,b = sh[-2], sh[-1]
        if hi[b] > hi[a] and rsi[b] < rsi[a]:
            # 더 최근 스윙 기준으로 우선
            if div_s == 0 or (sh[-1] > sl[-1] if sl else True):
                div_s = 2; div_desc = "RSI 약세 다이버전스, 고점 신호"

    # 3) 차트패턴(raw)
    cands = []  # (abs, raw, desc, neck, breakout, extreme_price)
    # 더블탑: 최근 두 스윙고점 3% 이내 + 최근스윙 25일 이내
    if len(sh) >= 2:
        h1,h2 = sh[-2], sh[-1]
        if abs(hi[h1]-hi[h2])/hi[h1] <= 0.03 and (last-h2) <= 25:
            mids = [j for j in sl if h1 < j < h2]
            neck = min(lo[j] for j in mids) if mids else min(lo[h1:h2+1])
            peak = max(hi[h1], hi[h2])
            brk = cl[last] < neck
            raw = 2 if brk else 1
            cands.append((abs(raw), raw, "더블탑", neck, brk, peak))
    # 더블바텀
    if len(sl) >= 2:
        l1,l2 = sl[-2], sl[-1]
        if abs(lo[l1]-lo[l2])/lo[l1] <= 0.03 and (last-l2) <= 25:
            mids = [j for j in sh if l1 < j < l2]
            neck = max(hi[j] for j in mids) if mids else max(hi[l1:l2+1])
            trough = min(lo[l1], lo[l2])
            brk = cl[last] > neck
            raw = -2 if brk else -1
            cands.append((abs(raw), raw, "더블바텀", neck, brk, trough))
    # 헤드앤숄더: 스윙고점 3개, 가운데가 극단(최고), 양어깨 6% 이내 + 25일 이내
    if len(sh) >= 3:
        s1,s2,s3 = sh[-3], sh[-2], sh[-1]
        if hi[s2] > hi[s1] and hi[s2] > hi[s3] and abs(hi[s1]-hi[s3])/hi[s1] <= 0.06 and (last-s3) <= 25:
            mids = [j for j in sl if s1 < j < s3]
            neck = (sum(lo[j] for j in mids)/len(mids)) if mids else min(lo[s1:s3+1])
            head = hi[s2]
            brk = cl[last] < neck
            raw = 3 if brk else 1
            cands.append((abs(raw), raw, "헤드앤숄더", neck, brk, head))
    # 역헤드앤숄더
    if len(sl) >= 3:
        s1,s2,s3 = sl[-3], sl[-2], sl[-1]
        if lo[s2] < lo[s1] and lo[s2] < lo[s3] and abs(lo[s1]-lo[s3])/lo[s1] <= 0.06 and (last-s3) <= 25:
            mids = [j for j in sh if s1 < j < s3]
            neck = (sum(hi[j] for j in mids)/len(mids)) if mids else max(hi[s1:s3+1])
            head = lo[s2]
            brk = cl[last] > neck
            raw = -3 if brk else -1
            cands.append((abs(raw), raw, "역헤드앤숄더", neck, brk, head))

    pattern_raw = 0; pattern_desc = None; neck=None; brk=False; extreme=None; pname=None
    if cands:
        cands.sort(reverse=True)
        _, pattern_raw, pname, neck, brk, extreme = cands[0]
        pattern_desc = f"{pname} {'넥라인 돌파' if brk else '넥라인 미돌파'}"

    pos_score = base['pos_score']; close = base['close']

    # 4-1) 목표가 (하락반전형 확정 패턴에만)
    target=None; gap=None; risk_flag=False; risk_desc=None
    if pattern_raw > 0 and brk:  # 확정 하락반전형
        target = neck - (extreme - neck)
        gap = (close - target)/close*100
        pre = "더블탑" if pname=="더블탑" else "H&S"
        if gap > 3:
            risk_flag = True
            risk_desc = f"{pre} 목표가 {round(target,1)} (추가 {round(gap,1)}% 여지)"
        else:
            risk_flag = False
            risk_desc = "패턴 목표가 도달, 하락 목표 달성"
    elif pattern_raw < 0 and brk:  # 상승반전형 확정 (risk_flag 미사용)
        target = neck + (neck - extreme)
        gap = (target - close)/close*100

    # 4) 패턴 소진 보정
    exhaust=""
    if pattern_raw > 0 and pos_score <= -2:
        pattern_s = 1; exhaust=" (하락 상당부분 진행)"
    elif pattern_raw < 0 and pos_score >= 2:
        pattern_s = -1; exhaust=" (상승 상당부분 진행)"
    else:
        pattern_s = pattern_raw
    # 넥라인 돌파 후 5% 이상 추가 진행 시 소진
    if brk and neck:
        prog = abs(close - neck)/neck*100
        if prog >= 5 and pattern_s != 0:
            pattern_s = 1 if pattern_s>0 else -1
            exhaust += " (신호 소진)"
    if pattern_desc:
        pattern_desc += exhaust

    # 5) 캔들패턴 (최근 3거래일 시가/종가)
    candle_s = 0; candle_desc = None
    o = df['o'].values; c = df['close'].values
    if n >= 3:
        o1,c1 = o[-3], c[-3]; o2,c2 = o[-2], c[-2]; o3,c3 = o[-1], c[-1]
        body1 = abs(c1-o1); body2 = abs(c2-o2); body3 = abs(c3-o3)
        mid1 = (o1+c1)/2
        avgbody = pd.Series(abs(c-o)).tail(10).mean()
        big1 = body1 >= avgbody
        small2 = body2 <= body1*0.5
        if big1 and small2 and c1<o1 and c3>o3 and c3 > mid1:
            candle_s = -1; candle_desc = "모닝스타 출현, 상승반전 신호"
        elif big1 and small2 and c1>o1 and c3<o3 and c3 < mid1:
            candle_s = 1; candle_desc = "이브닝스타 출현, 하락반전 신호"

    signal_score = base['vol_s']+div_s+pattern_s+candle_s
    total = pos_score + signal_score

    items = [base['rsi_s'],base['bb_s'],base['dev_s'],base['p52_s'],base['vol_s'],div_s,pattern_s,candle_s]
    low_cnt = sum(1 for x in items if x<0)
    high_cnt = sum(1 for x in items if x>0)
    mid_cnt = sum(1 for x in items if x==0)
    strong_low = sum(1 for x in items if x<=-2)
    strong_high = sum(1 for x in items if x>=2)

    if total<=-6 and low_cnt>=4 and strong_high==0:
        vemoji,vtext="🟢","저점권"
    elif total<=-3 and low_cnt>=3 and low_cnt>high_cnt:
        vemoji,vtext="🔵","저점 근접"
    elif total>=6 and high_cnt>=4 and strong_low==0:
        vemoji,vtext="🔴","고점권"
    elif total>=3 and high_cnt>=3 and high_cnt>low_cnt:
        vemoji,vtext="🟠","고점 근접"
    else:
        vemoji,vtext="⚪","중립 구간"

    # 4-2) verdict 보정
    vsuffix = ""
    if risk_flag and vtext in ("저점권","저점 근접"):
        vsuffix = "(반등 시도 구간, 구조적 하락 목표 미달)"

    base.update(div_s=div_s, div_desc=div_desc, pattern_raw=pattern_raw, pattern_s=pattern_s,
        pattern_name=pname, pattern_desc=pattern_desc, breakout=bool(brk),
        candle_s=candle_s, candle_desc=candle_desc,
        target=round(target,1) if target is not None else None,
        gap=round(gap,1) if gap is not None else None,
        risk_flag=risk_flag, risk_desc=risk_desc,
        signal_score=signal_score, total=total,
        low_cnt=low_cnt, high_cnt=high_cnt, mid_cnt=mid_cnt,
        strong_low=strong_low, strong_high=strong_high,
        vemoji=vemoji, vtext=vtext, vsuffix=vsuffix)
    return base

tickers = ["QQQ","SOXX","MSTR"]
results = []
for t in tickers:
    try:
        raw = json.load(open(f"{t}.json"))["data"]
        base, df = analyze(t, raw)
        base = process(base, df)
        results.append(base)
        print(json.dumps(base, ensure_ascii=False))
    except Exception as e:
        results.append({"ticker": t, "error": str(e)})
        print(json.dumps({"ticker": t, "error": str(e)}, ensure_ascii=False))

# ---------------- [3단계] 메시지 작성 ----------------
def rsi_state(r):
    return "극과매도" if r<=25 else "과매도" if r<=32 else "약과매도" if r<=42 else "약과매수" if r>=75 else "과매수" if r>=68 else "약과매수" if r>=58 else "중립"
def rsi_impl(s):
    return {-3:"반등임박",-2:"반등임박",-1:"반등여력",0:"방향 혼조",1:"과열초입",2:"조정주의",3:"조정주의"}[s]
def bb_pos(pb):
    return "하단이탈" if pb<=0.05 else "하단근접" if pb<=0.25 else "상단이탈" if pb>=0.95 else "상단근접" if pb>=0.75 else "중단권"
def bb_impl(s):
    return {-2:"낙폭과대",-1:"과대낙폭",0:"밴드중앙",1:"상승과열",2:"과열이탈"}[s]
def dev_state(dv):
    return "20일선 대폭이탈" if dv<=-15 else "20일선 아래 과대" if dv<=-7 else "20일선 위 과열" if dv>=15 else "20일선 위 과열" if dv>=7 else "20일선 부근"
def dev_state2(dv):
    if dv<=-15: return "20일선 대폭이탈"
    if dv<=-7: return "20일선 아래 과대"
    if dv>=15: return "20일선 대폭초과"
    if dv>=7: return "20일선 위 과열"
    return "20일선 부근"
def p52_zone(p):
    return "1년 최저권" if p<=10 else "하단권" if p<=25 else "중하단권" if p<=45 else "중앙권" if p<55 else "중상단권" if p<75 else "상단권" if p<90 else "1년 최고권"
def vol_txt(s,vr):
    if s==-2: return "투매급증","바닥신호"
    if s==2: return "과열급증","고점신호"
    if vr>=130: return "급증","방향미정"
    return "평이","신호미약"
def sf(x):
    return "0" if x==0 else f"{x:+d}"

# 기준 거래일: base date 중 가장 늦은 날짜
dates = [pd.to_datetime(b['date']) for b in results if 'date' in b]
ref = max(dates)
title = f"{ref.month}/{ref.day}"

lines = [f"📊 기술적 분석 ({title})", ""]
order = ["QQQ","SOXX","MSTR"]
bmap = {b['ticker']:b for b in results}
blocks = []
for t in order:
    b = bmap[t]
    if 'error' in b:
        blocks.append(f"⚠️ {t} | 데이터 조회 실패")
        continue
    blk = []
    header = f"{b['vemoji']} {t} | {b['vtext']} ({sf(b['total'])}) | {b['close']:.2f}"
    if len(header) > 28:
        header = f"{b['vemoji']} {t}|{b['vtext']} ({sf(b['total'])})|{b['close']:.2f}"
    blk.append(header)
    blk.append(f"- RSI {b['rsi']} {rsi_state(b['rsi'])}, {rsi_impl(b['rsi_s'])} ({sf(b['rsi_s'])})")
    blk.append(f"- 볼린저밴드 {bb_pos(b['pb'])}, {bb_impl(b['bb_s'])} ({sf(b['bb_s'])})")
    blk.append(f"- 이격도 {b['dev']}% {dev_state2(b['dev'])} ({sf(b['dev_s'])})")
    blk.append(f"- 52주 {b['pos52']}% {p52_zone(b['pos52'])} ({sf(b['p52_s'])})")
    vs,vi = vol_txt(b['vol_s'], b['vratio'])
    blk.append(f"- 거래량 {b['vratio']}% {vs}, {vi} ({sf(b['vol_s'])})")
    if b['div_s'] != 0:
        dv = "강세" if b['div_s']<0 else "약세"
        di = "저점신호" if b['div_s']<0 else "고점신호"
        blk.append(f"- 다이버전스 {dv}, {di} ({sf(b['div_s'])})")
    if b['pattern_s'] != 0:
        brtxt = "넥라인 돌파" if b['breakout'] else "넥라인 미돌파"
        blk.append(f"- {b['pattern_name']} {brtxt} ({sf(b['pattern_s'])})")
    if b['candle_s'] != 0:
        cn = "모닝스타" if b['candle_s']<0 else "이브닝스타"
        ci = "상승반전" if b['candle_s']<0 else "하락반전"
        blk.append(f"- {cn}, {ci} ({sf(b['candle_s'])})")
    if b['risk_flag'] and b['target'] is not None:
        blk.append(f"⚠️ 목표 {b['target']} ({b['gap']}% 여지)")
    if b['vsuffix']:
        blk.append("↳ 반등 시도, 하락목표 미달")
    blocks.append("\n".join(blk))

msg = "\n".join(lines) + "\n".join(blocks) if False else (lines[0] + "\n\n" + "\n\n".join(blocks))
print("=====MESSAGE=====")
print(msg)
print("=====LEN=====", len(msg))
open("message.txt","w").write(msg)
