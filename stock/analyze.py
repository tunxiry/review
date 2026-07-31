import json, pandas as pd

def analyze(t, data):
    df = pd.DataFrame(data)
    df['t'] = pd.to_datetime(df['t'])
    df = df.sort_values('t').reset_index(drop=True)
    df['close'] = df.apply(lambda r: r['a'] if ('a' in r and pd.notna(r['a'])) else r['c'], axis=1)
    df['h'] = df['h'].astype(float); df['l'] = df['l'].astype(float); df['o'] = df['o'].astype(float)
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    sd = df['close'].rolling(20).std()
    df['bb_up'] = df['ma20'] + 2*sd
    df['bb_lo'] = df['ma20'] - 2*sd
    df['pb'] = (df['close']-df['bb_lo'])/(df['bb_up']-df['bb_lo'])
    d = df['close'].diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al = l.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['rsi'] = 100 - 100/(1+ag/al)
    v20 = df['v'].rolling(20).mean()
    L, P = df.iloc[-1], df.iloc[-2]
    chg = (L.close-P.close)/P.close*100
    chg3 = (L.close-df['close'].iloc[-4])/df['close'].iloc[-4]*100
    vr = L.v/v20.iloc[-1] if v20.iloc[-1] else 1

    # 추세 판정: 종가-MA50 위치 + MA50 기울기(5거래일)
    ma50_now = df['ma50'].iloc[-1]; ma50_prev = df['ma50'].iloc[-6]
    if pd.notna(ma50_now) and pd.notna(ma50_prev):
        downtrend = bool(L.close < ma50_now and ma50_now < ma50_prev)
        uptrend = bool(L.close > ma50_now and ma50_now > ma50_prev)
    else:
        downtrend = uptrend = False
    trend = "하락추세" if downtrend else "상승추세" if uptrend else "추세중립"

    # 음수=저점방향, 0=중립, 양수=고점방향
    r = float(L.rsi)
    rsi_s = -3 if r<=25 else -2 if r<=32 else -1 if r<=42 else 3 if r>=75 else 2 if r>=68 else 1 if r>=58 else 0

    pb = float(L.pb) if L.bb_up!=L.bb_lo else 0.5
    # 밴드워킹: 추세 방향으로 3일 연속 밴드 끝단에 붙어 있으면 평균회귀(반전) 신호로 보지 않음
    bw_lo = downtrend and bool((df['pb'].tail(3)<=0.15).all())
    bw_hi = uptrend and bool((df['pb'].tail(3)>=0.85).all())
    if bw_lo or bw_hi:
        bb_s = 0
        bandwalk = "하단 밴드워킹, 하락지속" if bw_lo else "상단 밴드워킹, 상승지속"
    else:
        bandwalk = None
        bb_s = -2 if pb<=0.05 else -1 if pb<=0.25 else 2 if pb>=0.95 else 1 if pb>=0.75 else 0
        # 추세 역방향(역추세) 신호는 강도 1로 제한
        if downtrend and bb_s<0: bb_s = -1
        if uptrend and bb_s>0: bb_s = 1

    dev = (L.close-L.ma20)/L.ma20*100
    dev_s = -2 if dev<=-15 else -1 if dev<=-7 else 2 if dev>=15 else 1 if dev>=7 else 0

    hi52 = df['close'].tail(252).max(); lo52 = df['close'].tail(252).min()
    pos52 = (L.close-lo52)/(hi52-lo52)*100 if hi52!=lo52 else 50
    p52_s = -2 if pos52<=10 else -1 if pos52<=25 else 2 if pos52>=90 else 1 if pos52>=75 else 0

    # 거래량: 낙폭(상승폭)이 충분히 누적된 뒤의 대량 거래만 투매/과열 반전 신호로 인정
    hi20 = df['close'].tail(20).max(); lo20 = df['close'].tail(20).min()
    dd20 = (L.close-hi20)/hi20*100; ru20 = (L.close-lo20)/lo20*100
    vol_ctx = None
    if vr>=1.3 and (chg<=-2 or chg3<=-5):
        if r<=32 or dd20<=-15: vol_s=-2
        else: vol_s=0; vol_ctx="분산성 매도, 바닥신호 아님"
    elif vr>=1.3 and (chg>=2 or chg3>=5):
        if r>=68 or ru20>=15: vol_s=2
        else: vol_s=0; vol_ctx="흡수성 매수, 천장신호 아님"
    else: vol_s=0

    pos_score = rsi_s+bb_s+dev_s+p52_s
    base = dict(ticker=t, date=str(L.t.date()), close=round(float(L.close),2),
        trend=trend, rsi=round(r), rsi_s=rsi_s, pb=round(pb,2), bb_s=bb_s, bandwalk=bandwalk,
        dev=round(float(dev),1), dev_s=dev_s, pos52=round(float(pos52)), p52_s=p52_s,
        vratio=round(float(vr)*100), chg=round(float(chg),1), chg3=round(float(chg3),1),
        vol_s=vol_s, vol_ctx=vol_ctx, pos_score=pos_score)
    return base, df

def resample(df, rule):
    w = df.set_index('t').resample(rule).agg(
        o=('o','first'), h=('h','max'), l=('l','min'), close=('close','last'), v=('v','sum')).dropna()
    return w.reset_index()

# ---------------- [2단계-B] ----------------

def find_swings(df, k, lookback):
    """좌우 k봉보다 높은/낮은 지점. 최근 lookback 봉 내에서 탐지."""
    n = len(df)
    hs, ls = [], []
    start = max(k, n - lookback)
    for i in range(start, n - k):
        hi = df['h'].iloc[i]
        if hi > df['h'].iloc[i-k:i].max() and hi > df['h'].iloc[i+1:i+k+1].max():
            hs.append(i)
        lo = df['l'].iloc[i]
        if lo < df['l'].iloc[i-k:i].min() and lo < df['l'].iloc[i+1:i+k+1].min():
            ls.append(i)
    return hs, ls


def detect_patterns(df, hs, ls, recent):
    """차트패턴 후보 탐지. 양수=하락반전형, 음수=상승반전형."""
    n = len(df); c = float(df['close'].iloc[-1]); out = []

    # 더블탑
    if len(hs) >= 2:
        i1, i2 = hs[-2], hs[-1]
        h1, h2 = float(df['h'].iloc[i1]), float(df['h'].iloc[i2])
        if abs(h1-h2)/max(h1, h2) <= 0.03 and (n-1-i2) <= recent:
            neck = float(df['l'].iloc[i1:i2+1].min())
            conf = c < neck
            out.append(dict(name='더블탑', raw=2 if conf else 1, conf=conf,
                            neck=neck, ext=max(h1, h2), bear=True))
    # 더블바텀
    if len(ls) >= 2:
        i1, i2 = ls[-2], ls[-1]
        l1, l2 = float(df['l'].iloc[i1]), float(df['l'].iloc[i2])
        if abs(l1-l2)/max(l1, l2) <= 0.03 and (n-1-i2) <= recent:
            neck = float(df['h'].iloc[i1:i2+1].max())
            conf = c > neck
            out.append(dict(name='더블바텀', raw=-2 if conf else -1, conf=conf,
                            neck=neck, ext=min(l1, l2), bear=False))
    # 헤드앤숄더
    if len(hs) >= 3:
        i1, i2, i3 = hs[-3], hs[-2], hs[-1]
        a, b, cc = [float(df['h'].iloc[i]) for i in (i1, i2, i3)]
        if b > a and b > cc and abs(a-cc)/max(a, cc) <= 0.06 and (n-1-i3) <= recent:
            neck = float(df['l'].iloc[i1:i3+1].min())
            conf = c < neck
            out.append(dict(name='헤드앤숄더', raw=3 if conf else 1, conf=conf,
                            neck=neck, ext=b, bear=True))
    # 역헤드앤숄더
    if len(ls) >= 3:
        i1, i2, i3 = ls[-3], ls[-2], ls[-1]
        a, b, cc = [float(df['l'].iloc[i]) for i in (i1, i2, i3)]
        if b < a and b < cc and abs(a-cc)/max(a, cc) <= 0.06 and (n-1-i3) <= recent:
            neck = float(df['h'].iloc[i1:i3+1].max())
            conf = c > neck
            out.append(dict(name='역헤드앤숄더', raw=-3 if conf else -1, conf=conf,
                            neck=neck, ext=b, bear=False))
    if not out:
        return None
    out.sort(key=lambda p: abs(p['raw']), reverse=True)
    return out[0]


def enrich(base, df, wk, mo, fallback_1y=False):
    n = len(df); close = float(df['close'].iloc[-1]); pos_score = base['pos_score']
    hs, ls = find_swings(df, 3, 90)

    # 2) RSI 다이버전스
    div_s, div_desc = 0, None
    if len(ls) >= 2:
        i1, i2 = ls[-2], ls[-1]
        if df['l'].iloc[i2] < df['l'].iloc[i1] and df['rsi'].iloc[i2] > df['rsi'].iloc[i1]:
            div_s, div_desc = -2, "RSI 강세 다이버전스, 저점 신호"
    if div_s == 0 and len(hs) >= 2:
        i1, i2 = hs[-2], hs[-1]
        if df['h'].iloc[i2] > df['h'].iloc[i1] and df['rsi'].iloc[i2] < df['rsi'].iloc[i1]:
            div_s, div_desc = 2, "RSI 약세 다이버전스, 고점 신호"

    # 3) 차트패턴(raw)
    p = detect_patterns(df, hs, ls, 25)
    pattern_raw, pattern_desc = 0, None
    target = gap = None; risk_flag = False; risk_desc = None
    if p:
        pattern_raw = p['raw']
        pattern_desc = f"{p['name']} " + ("넥라인 돌파" if p['conf'] else "넥라인 미돌파")

    # 4) 패턴 소진 보정
    if pattern_raw > 0 and pos_score <= -2:
        pattern_s = 1; pattern_desc += " (하락 상당부분 진행)"
    elif pattern_raw < 0 and pos_score >= 2:
        pattern_s = -1; pattern_desc += " (상승 상당부분 진행)"
    else:
        pattern_s = pattern_raw
    if p and p['conf']:
        neck = p['neck']
        prog = (neck-close)/neck*100 if p['bear'] else (close-neck)/neck*100
        if prog >= 5 and pattern_s != 0:
            pattern_s = 1 if pattern_s > 0 else -1
            pattern_desc += " (신호 소진)"

    # 4-1) 목표가 계산 및 경고
    if p and p['conf']:
        neck = p['neck']
        if p['bear']:
            target = neck - (p['ext'] - neck)
            gap = (close - target)/close*100
            if gap > 3:
                risk_flag = True
                nm = "H&S" if p['name'] == '헤드앤숄더' else "더블탑"
                risk_desc = f"{nm} 목표가 {round(target,2)} (추가 {round(gap,1)}% 여지)"
            else:
                risk_flag = False
                risk_desc = "패턴 목표가 도달, 하락 목표 달성"
        else:
            target = neck + (neck - p['ext'])
            gap = (target - close)/close*100

    # 5) 캔들패턴 (최근 3거래일)
    candle_s, candle_desc = 0, None
    o1, c1 = float(df['o'].iloc[-3]), float(df['close'].iloc[-3])
    o2, c2 = float(df['o'].iloc[-2]), float(df['close'].iloc[-2])
    o3, c3 = float(df['o'].iloc[-1]), float(df['close'].iloc[-1])
    b1, b2 = abs(c1-o1), abs(c2-o2)
    avgb = (df['close'] - df['o']).abs().tail(20).mean()
    mid1 = (o1+c1)/2
    if c1 < o1 and b1 >= avgb and b2 < b1*0.5 and c3 > o3 and c3 > mid1:
        candle_s, candle_desc = -1, "모닝스타 출현, 상승반전 신호"
    elif c1 > o1 and b1 >= avgb and b2 < b1*0.5 and c3 < o3 and c3 < mid1:
        candle_s, candle_desc = 1, "이브닝스타 출현, 하락반전 신호"

    # 6) 주봉/월봉 패턴
    wk_s, wk_desc, mo_desc = 0, None, None
    if not fallback_1y:
        whs, wls = find_swings(wk, 2, 60)
        wp = detect_patterns(wk, whs, wls, 13)
        if wp:
            if wp['conf']:
                wk_s = 1 if wp['raw'] > 0 else -1
                wk_desc = f"주봉 {wp['name']} 넥라인 돌파"
            else:
                wk_s = 0
                wk_desc = f"주봉 {wp['name']} 형성 중"
            if wk_s > 0 and pos_score <= -2:
                wk_s = 0; wk_desc += " (진행 상당)"
            elif wk_s < 0 and pos_score >= 2:
                wk_s = 0; wk_desc += " (진행 상당)"
        mhs, mls = find_swings(mo, 2, 60)
        mp = detect_patterns(mo, mhs, mls, 8)
        if mp:
            mo_desc = f"월봉 {mp['name']} " + ("넥라인 돌파" if mp['conf'] else "형성 중")

    signal_score = base['vol_s'] + div_s + pattern_s + candle_s + wk_s
    total = pos_score + signal_score
    items = [base['rsi_s'], base['bb_s'], base['dev_s'], base['p52_s'],
             base['vol_s'], div_s, pattern_s, candle_s, wk_s]
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
                target=(round(target, 2) if target is not None else None),
                gap=(round(gap, 1) if gap is not None else None),
                risk_flag=risk_flag, risk_desc=risk_desc,
                candle_s=candle_s, candle_desc=candle_desc,
                wk_s=wk_s, wk_desc=wk_desc, mo_desc=mo_desc,
                signal_score=signal_score, total=total,
                low_cnt=low_cnt, high_cnt=high_cnt, mid_cnt=mid_cnt,
                strong_low=strong_low, strong_high=strong_high,
                verdict=verdict, emoji=emoji, verdict_note=verdict_note)
    return base


tickers = ["SPY","SOXX","MSTR"]
results = []
for t in tickers:
    try:
        raw = json.load(open(f"{t}.json"))["data"]
        base, df = analyze(t, raw)
        try:
            wk = resample(df, 'W-FRI'); mo = resample(df, 'ME')
        except ValueError:
            wk = resample(df, 'W-FRI'); mo = resample(df, 'M')
        results.append((base, df, wk, mo))
    except Exception as e:
        print(json.dumps({"ticker": t, "error": str(e)}, ensure_ascii=False))

for base, df, wk, mo in results:
    print(json.dumps(enrich(base, df, wk, mo), ensure_ascii=False))
