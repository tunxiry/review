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
    """좌우 k거래일보다 고가가 모두 높으면 스윙고점, 저가가 모두 낮으면 스윙저점."""
    n = len(df)
    start = max(k, n - lookback)
    highs, lows = [], []
    for i in range(start, n - k):
        hi = df['h'].iloc[i]; lo = df['l'].iloc[i]
        if all(hi > df['h'].iloc[j] for j in range(i-k, i)) and \
           all(hi > df['h'].iloc[j] for j in range(i+1, i+k+1)):
            highs.append(i)
        if all(lo < df['l'].iloc[j] for j in range(i-k, i)) and \
           all(lo < df['l'].iloc[j] for j in range(i+1, i+k+1)):
            lows.append(i)
    return highs, lows


def divergence(df, highs, lows):
    """RSI 다이버전스"""
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if df['l'].iloc[b] < df['l'].iloc[a] and df['rsi'].iloc[b] > df['rsi'].iloc[a]:
            return -2, "RSI 강세 다이버전스, 저점 신호"
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if df['h'].iloc[b] > df['h'].iloc[a] and df['rsi'].iloc[b] < df['rsi'].iloc[a]:
            return 2, "RSI 약세 다이버전스, 고점 신호"
    return 0, None


def detect_patterns(df, highs, lows):
    """차트패턴 후보 탐지. 상승반전형=음수, 하락반전형=양수. 절댓값 큰 것 하나 채택."""
    n = len(df)
    last = df['close'].iloc[-1]
    cands = []  # (raw, name, broke, neckline, extreme)

    # --- 더블탑 (하락반전, 양수)
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        ha, hb = df['h'].iloc[a], df['h'].iloc[b]
        if abs(hb-ha)/ha <= 0.03 and (n-1-b) <= 25:
            mid = df['l'].iloc[a:b+1].min()  # 두 고점 사이 저점 = 넥라인
            broke = last < mid
            cands.append((2 if broke else 1, "더블탑", broke, float(mid), float(max(ha, hb))))

    # --- 더블바텀 (상승반전, 음수)
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        la, lb = df['l'].iloc[a], df['l'].iloc[b]
        if abs(lb-la)/la <= 0.03 and (n-1-b) <= 25:
            mid = df['h'].iloc[a:b+1].max()
            broke = last > mid
            cands.append((-2 if broke else -1, "더블바텀", broke, float(mid), float(min(la, lb))))

    # --- 헤드앤숄더 (하락반전, 양수)
    if len(highs) >= 3:
        a, b, c = highs[-3], highs[-2], highs[-1]
        ha, hb, hc = df['h'].iloc[a], df['h'].iloc[b], df['h'].iloc[c]
        if hb > ha and hb > hc and abs(hc-ha)/ha <= 0.06 and (n-1-c) <= 25:
            t1 = df['l'].iloc[a:b+1].min(); t2 = df['l'].iloc[b:c+1].min()
            neck = float((t1+t2)/2)
            broke = last < neck
            cands.append((3 if broke else 1, "헤드앤숄더", broke, neck, float(hb)))

    # --- 역헤드앤숄더 (상승반전, 음수)
    if len(lows) >= 3:
        a, b, c = lows[-3], lows[-2], lows[-1]
        la, lb, lc = df['l'].iloc[a], df['l'].iloc[b], df['l'].iloc[c]
        if lb < la and lb < lc and abs(lc-la)/la <= 0.06 and (n-1-c) <= 25:
            p1 = df['h'].iloc[a:b+1].max(); p2 = df['h'].iloc[b:c+1].max()
            neck = float((p1+p2)/2)
            broke = last > neck
            cands.append((-3 if broke else -1, "역헤드앤숄더", broke, neck, float(lb)))

    if not cands:
        return 0, None, False, None, None
    cands.sort(key=lambda x: abs(x[0]), reverse=True)
    raw, name, broke, neck, extreme = cands[0]
    return int(raw), name, bool(broke), float(neck), float(extreme)


def candle_pattern(df):
    """최근 3거래일 시가/종가 기준 모닝스타/이브닝스타"""
    o1, c1 = float(df['o'].iloc[-3]), float(df['close'].iloc[-3])
    o2, c2 = float(df['o'].iloc[-2]), float(df['close'].iloc[-2])
    o3, c3 = float(df['o'].iloc[-1]), float(df['close'].iloc[-1])
    b1, b2, b3 = abs(c1-o1), abs(c2-o2), abs(c3-o3)
    avg = float((df['close']-df['o']).abs().tail(20).mean())
    mid1 = (o1+c1)/2
    # 모닝스타: 큰 음봉 -> 작은 몸통 -> 1일차 몸통 중간값 위로 마감하는 양봉
    if c1 < o1 and b1 >= avg and b2 <= b1*0.5 and c3 > o3 and c3 > mid1:
        return -1, "모닝스타 출현, 상승반전 신호"
    # 이브닝스타: 큰 양봉 -> 작은 몸통 -> 1일차 몸통 중간값 아래로 마감하는 음봉
    if c1 > o1 and b1 >= avg and b2 <= b1*0.5 and c3 < o3 and c3 < mid1:
        return 1, "이브닝스타 출현, 하락반전 신호"
    return 0, None


def extend(base, df):
    highs, lows = find_swings(df)
    last = float(df['close'].iloc[-1])

    # 2) 다이버전스
    div_s, div_desc = divergence(df, highs, lows)

    # 3) 차트패턴 raw
    pattern_raw, pname, broke, neck, extreme = detect_patterns(df, highs, lows)
    pattern_desc = None
    if pattern_raw != 0:
        pattern_desc = f"{pname} {'넥라인 돌파' if broke else '넥라인 미돌파'}"

    # 4) 패턴 소진 보정
    pos_score = base['pos_score']
    if pattern_raw > 0 and pos_score <= -2:
        pattern_s = 1
        pattern_desc += " (하락 상당부분 진행)"
    elif pattern_raw < 0 and pos_score >= 2:
        pattern_s = -1
        pattern_desc += " (상승 상당부분 진행)"
    else:
        pattern_s = pattern_raw

    # 넥라인 돌파 후 5% 이상 추가 진행 → 신호 소진
    if pattern_raw != 0 and broke and neck:
        prog = (neck-last)/neck*100 if pattern_raw > 0 else (last-neck)/neck*100
        if prog >= 5 and abs(pattern_s) > 1:
            pattern_s = 1 if pattern_s > 0 else -1
            pattern_desc += " (신호 소진)"

    # 4-1) 목표가
    target = gap = None; risk_flag = False; risk_desc = None
    if pattern_raw != 0 and broke and neck is not None:
        if pattern_raw > 0:  # 하락반전형
            target = round(neck - (extreme - neck), 2)
            gap = round((last - target)/last*100, 1)
            if gap > 3:
                risk_flag = True
                label = "H&S" if pname == "헤드앤숄더" else "더블탑"
                risk_desc = f"{label} 목표가 {target} (추가 {gap}% 여지)"
            else:
                risk_flag = False
                risk_desc = "패턴 목표가 도달, 하락 목표 달성"
        else:  # 상승반전형
            target = round(neck + (neck - extreme), 2)
            gap = round((target - last)/last*100, 1)

    # 5) 캔들패턴
    candle_s, candle_desc = candle_pattern(df)

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
        verdict = "저점권 🟢"; emoji = "🟢"
    elif total <= -3 and low_cnt >= 3 and low_cnt > high_cnt:
        verdict = "저점 근접 🔵"; emoji = "🔵"
    elif total >= 6 and high_cnt >= 4 and strong_low == 0:
        verdict = "고점권 🔴"; emoji = "🔴"
    elif total >= 3 and high_cnt >= 3 and high_cnt > low_cnt:
        verdict = "고점 근접 🟠"; emoji = "🟠"
    else:
        verdict = "중립 구간 ⚪"; emoji = "⚪"

    # 4-2) verdict 보정
    verdict_note = None
    if risk_flag and verdict in ("저점권 🟢", "저점 근접 🔵"):
        verdict_note = "(반등 시도 구간, 구조적 하락 목표 미달)"

    base.update(div_s=div_s, div_desc=div_desc,
                pattern_raw=pattern_raw, pattern_s=pattern_s, pattern_desc=pattern_desc,
                pattern_name=pname, pattern_broke=broke, neckline=(round(neck,2) if neck else None),
                target=target, gap=gap, risk_flag=risk_flag, risk_desc=risk_desc,
                candle_s=candle_s, candle_desc=candle_desc,
                signal_score=signal_score, total=total,
                low_cnt=low_cnt, high_cnt=high_cnt, mid_cnt=mid_cnt,
                strong_low=strong_low, strong_high=strong_high,
                verdict=verdict, emoji=emoji, verdict_note=verdict_note)
    return base


tickers = ["SPY", "SOXX", "MSTR", "TSLA", "INTC", "NVO"]
results = []
for t in tickers:
    try:
        raw = json.load(open(f"{t}.json"))["data"]
        base, df = analyze(t, raw)
        base = extend(base, df)
        results.append(base)
        print(json.dumps(base, ensure_ascii=False))
    except Exception as e:
        results.append({"ticker": t, "error": str(e)})
        print(json.dumps({"ticker": t, "error": str(e)}, ensure_ascii=False))

json.dump(results, open("results.json", "w"), ensure_ascii=False)
