import json, pandas as pd, numpy as np
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

# ===== 2단계-B =====
def swings(df, win=90, k=3):
    """좌우 k거래일보다 고가 모두 높으면 스윙고점, 저가 모두 낮으면 스윙저점"""
    n = len(df)
    start = max(k, n-win)
    highs=[]; lows=[]
    for i in range(start, n-k):
        hh = df['h'].iloc[i]; ll = df['l'].iloc[i]
        lft_h = df['h'].iloc[i-k:i]; rgt_h = df['h'].iloc[i+1:i+1+k]
        lft_l = df['l'].iloc[i-k:i]; rgt_l = df['l'].iloc[i+1:i+1+k]
        if (hh > lft_h).all() and (hh > rgt_h).all():
            highs.append((i, hh))
        if (ll < lft_l).all() and (ll < rgt_l).all():
            lows.append((i, ll))
    return highs, lows

def build(t, data):
    base, df = analyze(t, data)
    n = len(df)
    L = df.iloc[-1]
    highs, lows = swings(df, 90, 3)

    # 2) RSI 다이버전스
    div_s = 0; div_desc = None
    if len(lows) >= 2:
        (i1,p1),(i2,p2) = lows[-2], lows[-1]
        r1 = df['rsi'].iloc[i1]; r2 = df['rsi'].iloc[i2]
        if p2 < p1 and pd.notna(r1) and pd.notna(r2) and r2 > r1:
            div_s = -2; div_desc = "RSI 강세 다이버전스, 저점 신호"
    if len(highs) >= 2 and div_s == 0:
        (i1,p1),(i2,p2) = highs[-2], highs[-1]
        r1 = df['rsi'].iloc[i1]; r2 = df['rsi'].iloc[i2]
        if p2 > p1 and pd.notna(r1) and pd.notna(r2) and r2 < r1:
            div_s = 2; div_desc = "RSI 약세 다이버전스, 고점 신호"

    # 3) 차트패턴(raw)
    cands = []  # (abs, raw, desc, neckline, confirmed, head_high or peak, is_dt/is_hs, kind)
    close_last = float(L.close)
    # --- 더블탑 / 더블바텀 (최근 두 스윙 3% 이내 + 25거래일 이내) ---
    if len(highs) >= 2:
        (i1,h1),(i2,h2) = highs[-2], highs[-1]
        if abs(h2-h1)/max(h1,h2) <= 0.03 and (n-1-i2) <= 25:
            # 넥라인 = 두 고점 사이 최저 저가
            neck = df['l'].iloc[i1:i2+1].min()
            confirmed = close_last < neck
            raw = 2 if confirmed else 1
            cands.append((abs(raw), raw, "더블탑", float(neck), confirmed, max(h1,h2), "dt_top"))
    if len(lows) >= 2:
        (i1,l1),(i2,l2) = lows[-2], lows[-1]
        if abs(l2-l1)/max(l1,l2) <= 0.03 and (n-1-i2) <= 25:
            neck = df['h'].iloc[i1:i2+1].max()
            confirmed = close_last > neck
            raw = -2 if confirmed else -1
            cands.append((abs(raw), raw, "더블바텀", float(neck), confirmed, min(l1,l2), "dt_bot"))
    # --- 헤드앤숄더 / 역H&S (스윙 3개, 가운데 극단, 어깨 6% 이내 + 25거래일 이내) ---
    if len(highs) >= 3:
        (ia,ha),(ib,hb),(ic,hc) = highs[-3], highs[-2], highs[-1]
        if hb > ha and hb > hc and abs(hc-ha)/max(ha,hc) <= 0.06 and (n-1-ic) <= 25:
            neck = df['l'].iloc[ia:ic+1].min()
            confirmed = close_last < neck
            raw = 3 if confirmed else 1
            cands.append((abs(raw), raw, "헤드앤숄더", float(neck), confirmed, float(hb), "hs_top"))
    if len(lows) >= 3:
        (ia,la),(ib,lb),(ic,lc) = lows[-3], lows[-2], lows[-1]
        if lb < la and lb < lc and abs(lc-la)/max(la,lc) <= 0.06 and (n-1-ic) <= 25:
            neck = df['h'].iloc[ia:ic+1].max()
            confirmed = close_last > neck
            raw = -3 if confirmed else -1
            cands.append((abs(raw), raw, "역헤드앤숄더", float(neck), confirmed, float(lb), "hs_bot"))

    pattern_raw = 0; pattern_desc = None; chosen = None
    if cands:
        cands.sort(key=lambda x: x[0], reverse=True)
        chosen = cands[0]
        pattern_raw = chosen[1]
        pattern_desc = chosen[2]

    # 4) 패턴 소진 보정
    pos_score = base['pos_score']
    pattern_s = pattern_raw
    if pattern_raw > 0 and pos_score <= -2:
        pattern_s = 1
        pattern_desc = (pattern_desc or "") + " (하락 상당부분 진행)"
    elif pattern_raw < 0 and pos_score >= 2:
        pattern_s = -1
        pattern_desc = (pattern_desc or "") + " (상승 상당부분 진행)"
    else:
        pattern_s = pattern_raw
    # 넥라인 돌파 후 5% 이상 추가 진행 시 절댓값 1로 축소
    if chosen is not None and chosen[4]:  # confirmed
        neck = chosen[3]
        prog = abs(close_last - neck)/neck*100
        if prog >= 5 and abs(pattern_s) > 1:
            pattern_s = 1 if pattern_s > 0 else -1
            pattern_desc = (pattern_desc or "") + " (신호 소진)"

    # 4-1) 목표가 및 경고 (하락반전형 확정 패턴)
    target = None; gap = None; risk_flag = False; risk_desc = None
    up_target = None
    if chosen is not None and chosen[4] and chosen[1] > 0:  # 하락반전형 확정
        neck = chosen[3]; head = chosen[5]
        target = neck - (head - neck)
        gap = (close_last - target)/close_last*100
        target = round(float(target),1); gap = round(float(gap),1)
        label = "H&S" if chosen[6]=="hs_top" else "더블탑"
        if gap > 3:
            risk_flag = True
            risk_desc = f"{label} 목표가 {target} (추가 {gap}% 여지)"
        else:
            risk_flag = False
            risk_desc = "패턴 목표가 도달, 하락 목표 달성"
    elif chosen is not None and chosen[4] and chosen[1] < 0:  # 상승반전형 확정
        neck = chosen[3]; bottom = chosen[5]
        up_target = round(float(neck + (neck - bottom)),1)
        target = up_target
        gap = round(float((close_last - target)/close_last*100),1)

    # 5) 캔들패턴 (최근 3거래일 시가/종가)
    candle_s = 0; candle_desc = None
    if n >= 3:
        o1,c1 = float(df['o'].iloc[-3]), float(df['close'].iloc[-3])
        o2,c2 = float(df['o'].iloc[-2]), float(df['close'].iloc[-2])
        o3,c3 = float(df['o'].iloc[-1]), float(df['close'].iloc[-1])
        body1 = abs(c1-o1); body2 = abs(c2-o2)
        mid1 = (o1+c1)/2
        # 모닝스타: 큰 음봉 -> 작은 몸통 -> 1일차 몸통 중간값 위로 마감 양봉
        if c1 < o1 and body2 < body1*0.5 and c3 > o3 and c3 > mid1:
            candle_s = -1; candle_desc = "모닝스타 출현, 상승반전 신호"
        # 이브닝스타: 큰 양봉 -> 작은 몸통 -> 1일차 몸통 중간값 아래로 마감 음봉
        elif c1 > o1 and body2 < body1*0.5 and c3 < o3 and c3 < mid1:
            candle_s = 1; candle_desc = "이브닝스타 출현, 하락반전 신호"

    signal_score = base['vol_s'] + div_s + pattern_s + candle_s
    total = pos_score + signal_score

    items = [base['rsi_s'], base['bb_s'], base['dev_s'], base['p52_s'],
             base['vol_s'], div_s, pattern_s, candle_s]
    low_cnt = sum(1 for x in items if x < 0)
    high_cnt = sum(1 for x in items if x > 0)
    mid_cnt = sum(1 for x in items if x == 0)
    strong_low = sum(1 for x in items if x <= -2)
    strong_high = sum(1 for x in items if x >= 2)

    if total <= -6 and low_cnt >= 4 and strong_high == 0:
        verdict = "저점권"; emoji = "🟢"
    elif total <= -3 and low_cnt >= 3 and low_cnt > high_cnt:
        verdict = "저점 근접"; emoji = "🔵"
    elif total >= 6 and high_cnt >= 4 and strong_low == 0:
        verdict = "고점권"; emoji = "🔴"
    elif total >= 3 and high_cnt >= 3 and high_cnt > low_cnt:
        verdict = "고점 근접"; emoji = "🟠"
    else:
        verdict = "중립 구간"; emoji = "⚪"

    # 4-2) verdict 보정
    verdict_note = None
    if risk_flag and verdict in ("저점권","저점 근접"):
        verdict_note = "반등 시도, 하락목표 미달"

    base.update(dict(div_s=div_s, div_desc=div_desc,
        pattern_raw=pattern_raw, pattern_s=pattern_s, pattern_desc=pattern_desc,
        pattern_confirmed=bool(chosen[4]) if chosen else False,
        target=target, gap=gap, risk_flag=risk_flag, risk_desc=risk_desc,
        candle_s=candle_s, candle_desc=candle_desc,
        signal_score=signal_score, total=total,
        low_cnt=low_cnt, high_cnt=high_cnt, mid_cnt=mid_cnt,
        strong_low=strong_low, strong_high=strong_high,
        verdict=verdict, emoji=emoji, verdict_note=verdict_note))
    return base

tickers = ["SPY","SOXX","MSTR"]
for t in tickers:
    try:
        raw = json.load(open(f"{t}.json"))["data"]
        print(json.dumps(build(t, raw), ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"ticker": t, "error": str(e)}, ensure_ascii=False))
