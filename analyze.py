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

# ===================== 2단계-B =====================
def swings(df, lookback=90, k=3):
    """좌우 k거래일보다 고가가 모두 높으면 스윙고점, 저가가 모두 낮으면 스윙저점."""
    n = len(df)
    hi, lo = [], []
    start = max(k, n-lookback)
    for i in range(start, n-k):
        wh = df['h'].iloc[i-k:i+k+1]
        wl = df['l'].iloc[i-k:i+k+1]
        if df['h'].iloc[i] == wh.max() and (wh.values[k] > wh.drop(wh.index[k]).values).all():
            hi.append(i)
        if df['l'].iloc[i] == wl.min() and (wl.values[k] < wl.drop(wl.index[k]).values).all():
            lo.append(i)
    return hi, lo

def extend(base, df):
    n = len(df)
    hi_idx, lo_idx = swings(df, 90, 3)

    # 2) RSI 다이버전스
    div_s, div_desc = 0, None
    if len(lo_idx) >= 2:
        a, b = lo_idx[-2], lo_idx[-1]
        if df['close'].iloc[b] < df['close'].iloc[a] and df['rsi'].iloc[b] > df['rsi'].iloc[a]:
            div_s, div_desc = -2, "RSI 강세 다이버전스, 저점 신호"
    if div_s == 0 and len(hi_idx) >= 2:
        a, b = hi_idx[-2], hi_idx[-1]
        if df['close'].iloc[b] > df['close'].iloc[a] and df['rsi'].iloc[b] < df['rsi'].iloc[a]:
            div_s, div_desc = 2, "RSI 약세 다이버전스, 고점 신호"

    # 3) 차트패턴 (raw)
    close_last = df['close'].iloc[-1]
    cands = []  # (abs, pattern_raw, pattern_desc, broke(bool), meta)

    # 더블탑: 최근 두 스윙고점
    if len(hi_idx) >= 2:
        p1, p2 = hi_idx[-2], hi_idx[-1]
        h1, h2 = df['h'].iloc[p1], df['h'].iloc[p2]
        if abs(h1-h2)/max(h1,h2) <= 0.03 and (n-1-p2) <= 25:
            neck = df['l'].iloc[p1:p2+1].min()
            broke = close_last < neck
            raw = 2 if broke else 1
            cands.append((raw, raw, "더블탑", broke, dict(head=max(h1,h2), neck=neck, kind='dt')))
    # 더블바텀: 최근 두 스윙저점
    if len(lo_idx) >= 2:
        p1, p2 = lo_idx[-2], lo_idx[-1]
        l1, l2 = df['l'].iloc[p1], df['l'].iloc[p2]
        if abs(l1-l2)/max(l1,l2) <= 0.03 and (n-1-p2) <= 25:
            neck = df['h'].iloc[p1:p2+1].max()
            broke = close_last > neck
            raw = -2 if broke else -1
            cands.append((2 if broke else 1, raw, "더블바텀", broke, dict(head=min(l1,l2), neck=neck, kind='db')))
    # 헤드앤숄더: 스윙고점 3개, 가운데가 최대, 양 어깨 6% 이내
    if len(hi_idx) >= 3:
        s1, hd, s2 = hi_idx[-3], hi_idx[-2], hi_idx[-1]
        H1, HD, H2 = df['h'].iloc[s1], df['h'].iloc[hd], df['h'].iloc[s2]
        if HD > H1 and HD > H2 and abs(H1-H2)/max(H1,H2) <= 0.06 and (n-1-s2) <= 25:
            neck = df['l'].iloc[s1:s2+1].min()
            broke = close_last < neck
            raw = 3 if broke else 1
            cands.append((raw, raw, "헤드앤숄더", broke, dict(head=HD, neck=neck, kind='hs')))
    # 역헤드앤숄더: 스윙저점 3개, 가운데가 최소
    if len(lo_idx) >= 3:
        s1, hd, s2 = lo_idx[-3], lo_idx[-2], lo_idx[-1]
        L1, LD, L2 = df['l'].iloc[s1], df['l'].iloc[hd], df['l'].iloc[s2]
        if LD < L1 and LD < L2 and abs(L1-L2)/max(L1,L2) <= 0.06 and (n-1-s2) <= 25:
            neck = df['h'].iloc[s1:s2+1].max()
            broke = close_last > neck
            raw = -3 if broke else -1
            cands.append((3 if broke else 1, raw, "역헤드앤숄더", broke, dict(head=LD, neck=neck, kind='ihs')))

    pattern_raw, pattern_desc, pat_broke, pat_meta = 0, None, False, None
    if cands:
        cands.sort(key=lambda x: x[0], reverse=True)
        _, pattern_raw, pattern_desc, pat_broke, pat_meta = cands[0]

    # 4) 패턴 소진 보정
    pos_score = base['pos_score']
    pattern_s = pattern_raw
    if pattern_raw != 0:
        if pattern_raw > 0 and pos_score <= -2:
            pattern_s = 1; pattern_desc = pattern_desc + " (하락 상당부분 진행)"
        elif pattern_raw < 0 and pos_score >= 2:
            pattern_s = -1; pattern_desc = pattern_desc + " (상승 상당부분 진행)"
        else:
            pattern_s = pattern_raw
        # 넥라인 돌파 후 5% 이상 추가 진행 시 신호 소진
        if pat_broke and pat_meta is not None:
            neck = pat_meta['neck']
            prog = abs(close_last - neck)/neck*100
            if prog >= 5 and abs(pattern_s) > 1:
                pattern_s = 1 if pattern_s > 0 else -1
                pattern_desc = pattern_desc + " (신호 소진)"

    # 4-1) 목표가 계산 및 경고 (하락반전형 확정 패턴에만 risk_flag 적용)
    target, gap = None, None
    risk_flag, risk_desc = False, None
    if pat_meta is not None and pat_broke:
        neck = pat_meta['neck']; head = pat_meta['head']; kind = pat_meta['kind']
        if pattern_raw > 0:  # 하락반전형 (더블탑/헤드앤숄더)
            target = neck - (head - neck)
            gap = (close_last - target)/close_last*100
            target_r = round(float(target), 1); gap_r = round(float(gap), 1)
            nm = "더블탑" if kind == 'dt' else "H&S"
            if gap > 3:
                risk_flag = True
                risk_desc = f"{nm} 목표가 {target_r} (추가 {gap_r}% 여지)"
            else:
                risk_flag = False
                risk_desc = "패턴 목표가 도달, 하락 목표 달성"
            target, gap = target_r, gap_r
        elif pattern_raw < 0:  # 상승반전형 (반대로 계산, risk_flag 미사용)
            target = neck + (neck - head)
            gap = (target - close_last)/close_last*100
            target, gap = round(float(target), 1), round(float(gap), 1)

    # 5) 캔들패턴 (최근 3거래일 시가/종가 기준)
    candle_s, candle_desc = 0, None
    if n >= 3:
        o1, c1 = df['o'].iloc[-3], df['close'].iloc[-3]
        o2, c2 = df['o'].iloc[-2], df['close'].iloc[-2]
        o3, c3 = df['o'].iloc[-1], df['close'].iloc[-1]
        b1 = abs(c1-o1); b2 = abs(c2-o2); mid1 = (o1+c1)/2
        # 모닝스타: 큰 음봉 -> 작은 몸통 -> 1일차 몸통 중간값 위로 마감하는 양봉
        if c1 < o1 and b2 < b1*0.5 and c3 > o3 and c3 > mid1:
            candle_s, candle_desc = -1, "모닝스타 출현, 상승반전 신호"
        # 이브닝스타: 큰 양봉 -> 작은 몸통 -> 1일차 몸통 중간값 아래로 마감하는 음봉
        elif c1 > o1 and b2 < b1*0.5 and c3 < o3 and c3 < mid1:
            candle_s, candle_desc = 1, "이브닝스타 출현, 하락반전 신호"

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

    base.update(dict(
        div_s=div_s, div_desc=div_desc,
        pattern_raw=pattern_raw, pattern_s=pattern_s, pattern_desc=pattern_desc,
        pat_broke=bool(pat_broke),
        target=target, gap=gap, risk_flag=bool(risk_flag), risk_desc=risk_desc,
        candle_s=candle_s, candle_desc=candle_desc,
        signal_score=signal_score, total=total,
        low_cnt=low_cnt, high_cnt=high_cnt, mid_cnt=mid_cnt,
        strong_low=strong_low, strong_high=strong_high,
        verdict=verdict, emoji=emoji))
    return base

tickers = ["QQQ","SOXX","MSTR"]
results = []
for t in tickers:
    try:
        raw = json.load(open(f"{t}.json"))["data"]
        base, df = analyze(t, raw)
        base = extend(base, df)
        results.append(base)
        print(json.dumps(base, ensure_ascii=False))
    except Exception as e:
        import traceback; traceback.print_exc()
        print(json.dumps({"ticker": t, "error": str(e)}, ensure_ascii=False))
