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


def detect_swings(df, window=90, k=3):
    n = len(df)
    start = max(k, n-window)
    highs = []  # swing highs
    lows = []   # swing lows
    for i in range(start, n-k):
        hv = df['h'].iloc[i]
        if all(hv > df['h'].iloc[i-j] for j in range(1,k+1)) and all(hv > df['h'].iloc[i+j] for j in range(1,k+1)):
            highs.append(dict(i=i, price=float(hv), close=float(df['close'].iloc[i]),
                              rsi=float(df['rsi'].iloc[i]), date=str(df['t'].iloc[i].date())))
        lv = df['l'].iloc[i]
        if all(lv < df['l'].iloc[i-j] for j in range(1,k+1)) and all(lv < df['l'].iloc[i+j] for j in range(1,k+1)):
            lows.append(dict(i=i, price=float(lv), close=float(df['close'].iloc[i]),
                             rsi=float(df['rsi'].iloc[i]), date=str(df['t'].iloc[i].date())))
    return highs, lows


def analyze2(base, df):
    L = df.iloc[-1]
    n = len(df)
    close = float(L.close)
    highs, lows = detect_swings(df, 90, 3)

    # 2) RSI 다이버전스
    div_s = 0; div_desc = None
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if b['price'] < a['price'] and b['rsi'] > a['rsi']:
            div_s = -2; div_desc = "RSI 강세 다이버전스, 저점 신호"
    if div_s == 0 and len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if b['price'] > a['price'] and b['rsi'] < a['rsi']:
            div_s = 2; div_desc = "RSI 약세 다이버전스, 고점 신호"
    base['div_s'] = div_s; base['div_desc'] = div_desc

    # 3) 차트패턴(raw)
    candidates = []  # (raw, desc, name, broke, neckline, head_or_top, kind)
    pos_score = base['pos_score']

    def trough_between(i1, i2):
        seg = df['l'].iloc[i1:i2+1]
        return float(seg.min())
    def peak_between(i1, i2):
        seg = df['h'].iloc[i1:i2+1]
        return float(seg.max())

    # 더블탑 (하락반전, +)
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        recent = (n-1) - b['i']
        if a['price'] and abs(b['price']-a['price'])/a['price'] <= 0.03 and recent <= 25:
            neck = trough_between(a['i'], b['i'])
            broke = close < neck
            raw = 2 if broke else 1
            candidates.append((raw, "더블탑", "더블탑", broke, neck, max(a['price'],b['price']), 'dt'))
    # 더블바텀 (상승반전, -)
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        recent = (n-1) - b['i']
        if a['price'] and abs(b['price']-a['price'])/a['price'] <= 0.03 and recent <= 25:
            neck = peak_between(a['i'], b['i'])
            broke = close > neck
            raw = -2 if broke else -1
            candidates.append((raw, "더블바텀", "더블바텀", broke, neck, min(a['price'],b['price']), 'db'))
    # 헤드앤숄더 (하락반전, +)  세 스윙고점 중 가운데가 극대
    if len(highs) >= 3:
        s1, hd, s2 = highs[-3], highs[-2], highs[-1]
        recent = (n-1) - s2['i']
        if hd['price'] > s1['price'] and hd['price'] > s2['price'] and s1['price'] and \
           abs(s1['price']-s2['price'])/s1['price'] <= 0.06 and recent <= 25:
            neck = min(trough_between(s1['i'], hd['i']), trough_between(hd['i'], s2['i']))
            broke = close < neck
            raw = 3 if broke else 1
            candidates.append((raw, "헤드앤숄더", "헤드앤숄더", broke, neck, hd['price'], 'hs'))
    # 역헤드앤숄더 (상승반전, -) 세 스윙저점 중 가운데가 극소
    if len(lows) >= 3:
        s1, hd, s2 = lows[-3], lows[-2], lows[-1]
        recent = (n-1) - s2['i']
        if hd['price'] < s1['price'] and hd['price'] < s2['price'] and s1['price'] and \
           abs(s1['price']-s2['price'])/s1['price'] <= 0.06 and recent <= 25:
            neck = max(peak_between(s1['i'], hd['i']), peak_between(hd['i'], s2['i']))
            broke = close > neck
            raw = -3 if broke else -1
            candidates.append((raw, "역헤드앤숄더", "역헤드앤숄더", broke, neck, hd['price'], 'ihs'))

    pattern_raw = 0; pattern_desc = None; chosen = None
    if candidates:
        chosen = max(candidates, key=lambda c: abs(c[0]))
        pattern_raw = chosen[0]; pattern_desc = chosen[1]

    # 4) 패턴 소진 보정
    pattern_s = pattern_raw
    if pattern_raw > 0 and pos_score <= -2:
        pattern_s = 1
        if pattern_desc: pattern_desc += " (하락 상당부분 진행)"
    elif pattern_raw < 0 and pos_score >= 2:
        pattern_s = -1
        if pattern_desc: pattern_desc += " (상승 상당부분 진행)"

    broke_conf = chosen[3] if chosen else False
    neck = chosen[4] if chosen else None
    # 넥라인 돌파 확정 후 5% 이상 추가 진행 → 소진
    if chosen and broke_conf and neck:
        prog = abs(close - neck)/neck*100
        if prog >= 5:
            if pattern_s != 0:
                pattern_s = 1 if pattern_s > 0 else -1
            if pattern_desc: pattern_desc += " (신호 소진)"

    base['pattern_raw'] = pattern_raw
    base['pattern_s'] = pattern_s
    base['pattern_desc'] = pattern_desc
    base['pattern_broke'] = bool(broke_conf)

    # 4-1) 목표가 (하락반전형 확정 패턴에만)
    base['target'] = None; base['gap'] = None
    base['risk_flag'] = False; base['risk_desc'] = None
    if chosen and broke_conf and pattern_raw > 0:
        head = chosen[5]
        target = neck - (head - neck)
        gap = (close - target)/close*100
        base['target'] = round(target, 1); base['gap'] = round(gap, 1)
        label = "더블탑" if chosen[6]=='dt' else "H&S"
        if gap > 3:
            base['risk_flag'] = True
            base['risk_desc'] = f"{label} 목표가 {round(target,1)} (추가 {round(gap,1)}% 여지)"
        else:
            base['risk_flag'] = False
            base['risk_desc'] = "패턴 목표가 도달, 하락 목표 달성"
    elif chosen and broke_conf and pattern_raw < 0:
        # 상승반전형 목표 (참고용, risk_flag 미사용)
        bottom = chosen[5]
        target = neck + (neck - bottom)
        gap = (close - target)/close*100
        base['target'] = round(target, 1); base['gap'] = round(gap, 1)

    # 5) 캔들패턴 (최근 3일)
    candle_s = 0; candle_desc = None
    if n >= 3:
        o1,c1 = float(df['o'].iloc[-3]), float(df['close'].iloc[-3])
        o2,c2 = float(df['o'].iloc[-2]), float(df['close'].iloc[-2])
        o3,c3 = float(df['o'].iloc[-1]), float(df['close'].iloc[-1])
        b1 = abs(c1-o1); b2 = abs(c2-o2); b3 = abs(c3-o3)
        mid1 = (o1+c1)/2
        avgbody = df.apply(lambda x: abs(x['close']-x['o']), axis=1).tail(20).mean()
        # 모닝스타: 큰 음봉 → 작은 몸통 → 1일차 몸통 중간값 위 마감 양봉
        if c1 < o1 and b1 > avgbody and b2 < b1*0.5 and c3 > o3 and c3 > mid1:
            candle_s = -1; candle_desc = "모닝스타 출현, 상승반전 신호"
        elif c1 > o1 and b1 > avgbody and b2 < b1*0.5 and c3 < o3 and c3 < mid1:
            candle_s = 1; candle_desc = "이브닝스타 출현, 하락반전 신호"
    base['candle_s'] = candle_s; base['candle_desc'] = candle_desc

    # 종합
    signal_score = base['vol_s'] + div_s + pattern_s + candle_s
    total = pos_score + signal_score
    base['signal_score'] = signal_score
    base['total'] = total

    items = [base['rsi_s'], base['bb_s'], base['dev_s'], base['p52_s'],
             base['vol_s'], div_s, pattern_s, candle_s]
    low_cnt = sum(1 for x in items if x < 0)
    high_cnt = sum(1 for x in items if x > 0)
    mid_cnt = sum(1 for x in items if x == 0)
    strong_low = sum(1 for x in items if x <= -2)
    strong_high = sum(1 for x in items if x >= 2)
    base.update(low_cnt=low_cnt, high_cnt=high_cnt, mid_cnt=mid_cnt,
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
    if base['risk_flag'] and verdict in ("저점권", "저점 근접"):
        verdict += " (반등 시도 구간, 구조적 하락 목표 미달)"
    base['verdict'] = verdict; base['emoji'] = emoji
    return base


tickers = ["QQQ","SOXX","MSTR"]
results = []
for t in tickers:
    try:
        raw = json.load(open(f"{t}.json"))["data"]
        b, df = analyze(t, raw)
        b = analyze2(b, df)
        results.append(b)
    except Exception as e:
        results.append({"ticker": t, "error": str(e)})

for b in results:
    print(json.dumps(b, ensure_ascii=False))
