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


# ==================== [2단계-B] ====================
def find_swings(df, lookback=90):
    """좌우 3거래일보다 고가가 모두 높으면 스윙고점, 저가가 모두 낮으면 스윙저점."""
    n = len(df)
    start = max(3, n - lookback)
    highs, lows = [], []
    for i in range(start, n-3):
        hv = df['h'].iloc[i]
        if all(hv > df['h'].iloc[i-k] for k in range(1,4)) and all(hv > df['h'].iloc[i+k] for k in range(1,4)):
            highs.append(i)
        lv = df['l'].iloc[i]
        if all(lv < df['l'].iloc[i-k] for k in range(1,4)) and all(lv < df['l'].iloc[i+k] for k in range(1,4)):
            lows.append(i)
    return highs, lows


def build(t, base, df):
    n = len(df)
    highs, lows = find_swings(df, 90)
    last_close = float(df['close'].iloc[-1])

    # ---------- 2) RSI 다이버전스 ----------
    div_s, div_desc = 0, None
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if df['l'].iloc[b] < df['l'].iloc[a] and df['rsi'].iloc[b] > df['rsi'].iloc[a]:
            div_s, div_desc = -2, "RSI 강세 다이버전스, 저점 신호"
    if div_s == 0 and len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if df['h'].iloc[b] > df['h'].iloc[a] and df['rsi'].iloc[b] < df['rsi'].iloc[a]:
            div_s, div_desc = 2, "RSI 약세 다이버전스, 고점 신호"

    # ---------- 3) 차트패턴(raw) ----------
    pattern_raw, pattern_desc = 0, None
    neckline = None            # 넥라인 값
    head_extreme = None        # 머리(또는 고점/저점) 극단값
    pat_broken = False         # 넥라인 돌파 확정 여부
    pat_type = None            # 'DT','DB','HS','IHS'
    candidates = []            # (abs_score, raw, desc, type, neckline, head, broken)

    def recent_enough(idx):
        return (n-1 - idx) <= 25

    # 더블탑: 최근 두 스윙고점 차이 3% 이내 + 최근 스윙 25거래일 이내
    if len(highs) >= 2:
        h1, h2 = highs[-2], highs[-1]
        p1, p2 = df['h'].iloc[h1], df['h'].iloc[h2]
        if abs(p1-p2)/max(p1,p2) <= 0.03 and recent_enough(h2):
            neck = float(df['l'].iloc[h1:h2+1].min())
            broken = last_close < neck
            sc = 2 if broken else 1
            candidates.append((sc, sc, "더블탑" + ("(넥라인 돌파)" if broken else "(형성중)"),
                               'DT', neck, float(max(p1,p2)), broken))
    # 더블바텀: 최근 두 스윙저점
    if len(lows) >= 2:
        l1, l2 = lows[-2], lows[-1]
        p1, p2 = df['l'].iloc[l1], df['l'].iloc[l2]
        if abs(p1-p2)/max(p1,p2) <= 0.03 and recent_enough(l2):
            neck = float(df['h'].iloc[l1:l2+1].max())
            broken = last_close > neck
            sc = 2 if broken else 1
            candidates.append((sc, -sc, "더블바텀" + ("(넥라인 돌파)" if broken else "(형성중)"),
                               'DB', neck, float(min(p1,p2)), broken))
    # 헤드앤숄더: 스윙고점 3개 중 가운데가 극단(가장 높음), 양 어깨 차이 6% 이내 + 25일 이내
    if len(highs) >= 3:
        s1, hd, s2 = highs[-3], highs[-2], highs[-1]
        ph1, phd, ph2 = df['h'].iloc[s1], df['h'].iloc[hd], df['h'].iloc[s2]
        if phd > ph1 and phd > ph2 and abs(ph1-ph2)/max(ph1,ph2) <= 0.06 and recent_enough(s2):
            neck = float(df['l'].iloc[s1:s2+1].min())
            broken = last_close < neck
            sc = 3 if broken else 1
            candidates.append((sc, sc, "헤드앤숄더" + ("(넥라인 돌파)" if broken else "(형성중)"),
                               'HS', neck, float(phd), broken))
    # 역헤드앤숄더: 스윙저점 3개 중 가운데가 극단(가장 낮음)
    if len(lows) >= 3:
        s1, hd, s2 = lows[-3], lows[-2], lows[-1]
        pl1, pld, pl2 = df['l'].iloc[s1], df['l'].iloc[hd], df['l'].iloc[s2]
        if pld < pl1 and pld < pl2 and abs(pl1-pl2)/max(pl1,pl2) <= 0.06 and recent_enough(s2):
            neck = float(df['h'].iloc[s1:s2+1].max())
            broken = last_close > neck
            sc = 3 if broken else 1
            candidates.append((sc, -sc, "역헤드앤숄더" + ("(넥라인 돌파)" if broken else "(형성중)"),
                               'IHS', neck, float(pld), broken))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, pattern_raw, pattern_desc, pat_type, neckline, head_extreme, pat_broken = candidates[0]

    # ---------- 4) 패턴 소진 보정 ----------
    pos_score = base['pos_score']
    pattern_s = pattern_raw
    if pattern_raw > 0 and pos_score <= -2:
        pattern_s = 1
        pattern_desc = (pattern_desc or "") + " (하락 상당부분 진행)"
    elif pattern_raw < 0 and pos_score >= 2:
        pattern_s = -1
        pattern_desc = (pattern_desc or "") + " (상승 상당부분 진행)"

    # 넥라인 돌파 확정 후 5% 이상 추가 진행 시 신호 소진
    if pat_broken and neckline:
        progressed = abs(last_close - neckline) / neckline * 100
        if progressed >= 5:
            if pattern_s > 0: pattern_s = 1
            elif pattern_s < 0: pattern_s = -1
            pattern_desc = (pattern_desc or "") + " (신호 소진)"

    # ---------- 4-1) 목표가 계산 및 경고 (하락반전형 확정 패턴) ----------
    target = None; gap = None; risk_flag = False; risk_desc = None
    if pat_broken and pat_type in ('HS','DT') and pattern_raw > 0 and neckline and head_extreme:
        target = neckline - (head_extreme - neckline)
        gap = (last_close - target) / last_close * 100
        target = round(target, 2); gap = round(gap, 1)
        label = "H&S" if pat_type == 'HS' else "더블탑"
        if gap > 3:
            risk_flag = True
            risk_desc = f"{label} 목표가 {target} (추가 {gap}% 여지)"
        else:
            risk_desc = "패턴 목표가 도달, 하락 목표 달성"
    elif pat_broken and pat_type in ('IHS','DB') and pattern_raw < 0 and neckline and head_extreme:
        # 상승반전형 확정 패턴 목표가 (risk_flag 미사용)
        target = neckline + (neckline - head_extreme)
        gap = (last_close - target) / last_close * 100
        target = round(target, 2); gap = round(gap, 1)

    # ---------- 5) 캔들패턴 ----------
    candle_s, candle_desc = 0, None
    if n >= 3:
        o1,c1 = float(df['o'].iloc[-3]), float(df['close'].iloc[-3])
        o2,c2 = float(df['o'].iloc[-2]), float(df['close'].iloc[-2])
        o3,c3 = float(df['o'].iloc[-1]), float(df['close'].iloc[-1])
        b1 = abs(c1-o1); b2 = abs(c2-o2); b3 = abs(c3-o3)
        mid1 = (o1+c1)/2
        # 모닝스타: 큰 음봉 → 작은 몸통 → 1일차 몸통 중간값 위로 마감하는 양봉
        if c1 < o1 and b2 < b1*0.5 and c3 > o3 and c3 > mid1:
            candle_s, candle_desc = -1, "모닝스타 출현, 상승반전 신호"
        # 이브닝스타: 큰 양봉 → 작은 몸통 → 1일차 몸통 중간값 아래로 마감하는 음봉
        elif c1 > o1 and b2 < b1*0.5 and c3 < o3 and c3 < mid1:
            candle_s, candle_desc = 1, "이브닝스타 출현, 하락반전 신호"

    # ---------- 종합 ----------
    signal_score = base['vol_s'] + div_s + pattern_s + candle_s
    total = pos_score + signal_score

    items = [base['rsi_s'], base['bb_s'], base['dev_s'], base['p52_s'],
             base['vol_s'], div_s, pattern_s, candle_s]
    low_cnt = sum(1 for x in items if x < 0)
    high_cnt = sum(1 for x in items if x > 0)
    mid_cnt = sum(1 for x in items if x == 0)
    strong_low = sum(1 for x in items if x <= -2)
    strong_high = sum(1 for x in items if x >= 2)

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
        verdict = verdict + "(반등 시도 구간, 구조적 하락 목표 미달)"

    base.update(div_s=div_s, div_desc=div_desc,
                pattern_raw=pattern_raw, pattern_s=pattern_s, pattern_desc=pattern_desc,
                candle_s=candle_s, candle_desc=candle_desc,
                target=target, gap=gap, risk_flag=risk_flag, risk_desc=risk_desc,
                signal_score=signal_score, total=total,
                low_cnt=low_cnt, high_cnt=high_cnt, mid_cnt=mid_cnt,
                strong_low=strong_low, strong_high=strong_high,
                verdict=verdict, emoji=emoji)
    return base


tickers = ["MSTR","GOOGL","SOXX","QQQ"]
results = []
for t in tickers:
    try:
        raw = json.load(open(f"{t}.json"))["data"]
        base, df = analyze(t, raw)
        base = build(t, base, df)
        results.append(base)
        print(json.dumps(base, ensure_ascii=False))
    except Exception as e:
        import traceback
        results.append({"ticker": t, "error": str(e)})
        print(json.dumps({"ticker": t, "error": str(e), "tb": traceback.format_exc()}, ensure_ascii=False))
