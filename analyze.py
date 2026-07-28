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


# ---------- 2단계-B ----------
def find_swings(df, window=90, k=3):
    n = len(df)
    start = max(k, n-window)
    highs, lows = [], []
    hv, lv = df['h'].values, df['l'].values
    for i in range(start, n-k):
        seg_l = range(i-k, i); seg_r = range(i+1, i+k+1)
        if all(hv[i] > hv[j] for j in seg_l) and all(hv[i] > hv[j] for j in seg_r):
            highs.append(i)
        if all(lv[i] < lv[j] for j in seg_l) and all(lv[i] < lv[j] for j in seg_r):
            lows.append(i)
    return highs, lows

def enrich(base, df):
    n = len(df)
    highs, lows = find_swings(df)
    L = df.iloc[-1]
    close = float(L.close)
    hv, lv, cv, rsiv = df['h'].values, df['l'].values, df['close'].values, df['rsi'].values

    # 2) RSI 다이버전스
    div_s, div_desc = 0, None
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if lv[b] < lv[a] and rsiv[b] > rsiv[a]:
            div_s, div_desc = -2, "RSI 강세 다이버전스, 저점 신호"
    if div_s == 0 and len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if hv[b] > hv[a] and rsiv[b] < rsiv[a]:
            div_s, div_desc = 2, "RSI 약세 다이버전스, 고점 신호"

    # 3) 차트패턴(raw)
    cands = []  # (abs, raw, name, broken, neckline, head_high/peak)
    recent = 25
    # 더블탑
    if len(highs) >= 2:
        h1, h2 = highs[-2], highs[-1]
        if abs(hv[h2]-hv[h1])/hv[h1] <= 0.03 and (n-1-h2) <= recent:
            mids = [i for i in lows if h1 < i < h2]
            if mids:
                neck = min(lv[i] for i in mids)
                broken = close < neck
                raw = 2 if broken else 1
                peak = max(hv[h1], hv[h2])
                cands.append((abs(raw), raw, "더블탑", broken, neck, peak))
    # 더블바텀
    if len(lows) >= 2:
        l1, l2 = lows[-2], lows[-1]
        if abs(lv[l2]-lv[l1])/lv[l1] <= 0.03 and (n-1-l2) <= recent:
            mids = [i for i in highs if l1 < i < l2]
            if mids:
                neck = max(hv[i] for i in mids)
                broken = close > neck
                raw = -2 if broken else -1
                trough = min(lv[l1], lv[l2])
                cands.append((abs(raw), raw, "더블바텀", broken, neck, trough))
    # 헤드앤숄더 (스윙고점 3개)
    if len(highs) >= 3:
        a, b, c = highs[-3], highs[-2], highs[-1]
        if hv[b] > hv[a] and hv[b] > hv[c] and abs(hv[a]-hv[c])/hv[a] <= 0.06 and (n-1-c) <= recent:
            mids = [lv[i] for i in lows if a < i < c]
            if mids:
                neck = sum(mids)/len(mids)
                broken = close < neck
                raw = 3 if broken else 1
                cands.append((abs(raw), raw, "헤드앤숄더", broken, neck, hv[b]))
    # 역헤드앤숄더 (스윙저점 3개)
    if len(lows) >= 3:
        a, b, c = lows[-3], lows[-2], lows[-1]
        if lv[b] < lv[a] and lv[b] < lv[c] and abs(lv[a]-lv[c])/lv[a] <= 0.06 and (n-1-c) <= recent:
            mids = [hv[i] for i in highs if a < i < c]
            if mids:
                neck = sum(mids)/len(mids)
                broken = close > neck
                raw = -3 if broken else -1
                cands.append((abs(raw), raw, "역헤드앤숄더", broken, neck, lv[b]))

    pattern_raw, pattern_s, pattern_desc = 0, 0, None
    pat_name, broken, neck, apex = None, False, None, None
    if cands:
        cands.sort(key=lambda x: x[0], reverse=True)
        _, pattern_raw, pat_name, broken, neck, apex = cands[0]
        nb = "넥라인 돌파" if broken else "넥라인 미돌파"
        pattern_desc = f"{pat_name} {nb}"

    # 4) 패턴 소진 보정
    pos_score = base['pos_score']
    if pattern_raw > 0:
        if pos_score <= -2:
            pattern_s = 1; pattern_desc += " (하락 상당부분 진행)"
        else:
            pattern_s = pattern_raw
    elif pattern_raw < 0:
        if pos_score >= 2:
            pattern_s = -1; pattern_desc += " (상승 상당부분 진행)"
        else:
            pattern_s = pattern_raw
    else:
        pattern_s = 0
    # 넥라인 돌파 후 5% 이상 추가 진행 → 신호 소진
    if broken and neck:
        if pattern_raw > 0 and (neck-close)/neck*100 >= 5:
            pattern_s = 1 if pattern_s > 0 else pattern_s
            if pattern_s != 0 and abs(pattern_s) > 1: pattern_s = pattern_s//abs(pattern_s)
            pattern_s = max(-1, min(1, pattern_s)); pattern_desc += " (신호 소진)"
        elif pattern_raw < 0 and (close-neck)/neck*100 >= 5:
            pattern_s = max(-1, min(1, pattern_s)); pattern_desc += " (신호 소진)"

    base.update(div_s=div_s, div_desc=div_desc,
                pattern_raw=pattern_raw, pattern_s=pattern_s, pattern_desc=pattern_desc)

    # 4-1) 목표가 (하락반전형 확정 패턴)
    target = None; gap = None; risk_flag = False; risk_desc = None
    confirmed_bear = broken and pattern_raw > 0
    confirmed_bull = broken and pattern_raw < 0
    if confirmed_bear:
        target = neck - (apex - neck)
        gap = (close - target)/close*100
        if gap > 3:
            risk_flag = True
            pfx = "더블탑" if pat_name == "더블탑" else "H&S"
            risk_desc = f"{pfx} 목표가 {round(target,2)} (추가 {round(gap,1)}% 여지)"
        else:
            risk_flag = False
            risk_desc = "패턴 목표가 도달, 하락 목표 달성"
    elif confirmed_bull:
        target = neck + (neck - apex)
        gap = (target - close)/close*100
    base.update(target=(round(target,2) if target is not None else None),
                gap=(round(gap,1) if gap is not None else None),
                risk_flag=risk_flag, risk_desc=risk_desc)

    # 5) 캔들패턴 (최근 3거래일)
    candle_s, candle_desc = 0, None
    o = df['o'].values
    o1,c1 = o[-3], cv[-3]; o2,c2 = o[-2], cv[-2]; o3,c3 = o[-1], cv[-1]
    body1 = abs(c1-o1); body2 = abs(c2-o2); body3 = abs(c3-o3)
    mid1 = (o1+c1)/2
    if c1 < o1 and body2 < body1*0.5 and c3 > o3 and c3 > mid1:
        candle_s, candle_desc = -1, "모닝스타 출현, 상승반전 신호"
    elif c1 > o1 and body2 < body1*0.5 and c3 < o3 and c3 < mid1:
        candle_s, candle_desc = 1, "이브닝스타 출현, 하락반전 신호"
    base.update(candle_s=candle_s, candle_desc=candle_desc)

    # 종합
    signal_score = vol = base['vol_s'] + div_s + pattern_s + candle_s
    total = pos_score + signal_score
    items = [base['rsi_s'], base['bb_s'], base['dev_s'], base['p52_s'],
             base['vol_s'], div_s, pattern_s, candle_s]
    low_cnt = sum(1 for x in items if x < 0)
    high_cnt = sum(1 for x in items if x > 0)
    mid_cnt = sum(1 for x in items if x == 0)
    strong_low = sum(1 for x in items if x <= -2)
    strong_high = sum(1 for x in items if x >= 2)
    base.update(signal_score=signal_score, total=total,
                low_cnt=low_cnt, high_cnt=high_cnt, mid_cnt=mid_cnt,
                strong_low=strong_low, strong_high=strong_high)

    # verdict
    if total <= -6 and low_cnt >= 4 and strong_high == 0:
        verdict, emoji = "저점권", "🟢"
    elif total <= -3 and low_cnt >= 3 and low_cnt > high_cnt:
        verdict, emoji = "저점 근접", "🔵"
    elif total >= 6 and high_cnt >= 4 and strong_low == 0:
        verdict, emoji = "고점권", "🔴"
    elif total >= 3 and high_cnt >= 3 and high_cnt > low_cnt:
        verdict, emoji = "고점 근접", "🟠"
    else:
        verdict, emoji = "중립 구간", "⚪"

    # 4-2) verdict 보정
    if risk_flag and verdict in ("저점권", "저점 근접"):
        verdict = verdict + " (반등 시도 구간, 구조적 하락 목표 미달)"
    base.update(verdict=verdict, emoji=emoji)
    return base


tickers = ["SPY", "SOXX", "MSTR"]
out = []
for t in tickers:
    try:
        raw = json.load(open(f"{t}.json"))["data"]
        base, df = analyze(t, raw)
        base = enrich(base, df)
        out.append(base)
        print(json.dumps(base, ensure_ascii=False))
    except Exception as e:
        err = {"ticker": t, "error": str(e)}
        out.append(err)
        print(json.dumps(err, ensure_ascii=False))
