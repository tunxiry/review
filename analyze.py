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

# ---------------- 2단계-B ----------------

def swings(df, k, lookback):
    """좌우 k봉보다 고가가 모두 높으면 스윙고점, 저가가 모두 낮으면 스윙저점."""
    n = len(df)
    start = max(k, n - lookback)
    hi, lo = [], []
    H = df['h'].values; LW = df['l'].values
    for i in range(start, n-k):
        if all(H[i] > H[j] for j in range(i-k, i)) and all(H[i] > H[j] for j in range(i+1, i+k+1)):
            hi.append(i)
        if all(LW[i] < LW[j] for j in range(i-k, i)) and all(LW[i] < LW[j] for j in range(i+1, i+k+1)):
            lo.append(i)
    return hi, lo

def divergence(df, hi, lo):
    if len(lo) >= 2:
        a, b = lo[-2], lo[-1]
        if df['l'].iloc[b] < df['l'].iloc[a] and df['rsi'].iloc[b] > df['rsi'].iloc[a]:
            return -2, "RSI 강세 다이버전스, 저점 신호"
    if len(hi) >= 2:
        a, b = hi[-2], hi[-1]
        if df['h'].iloc[b] > df['h'].iloc[a] and df['rsi'].iloc[b] < df['rsi'].iloc[a]:
            return 2, "RSI 약세 다이버전스, 고점 신호"
    return 0, None

def find_pattern(df, hi, lo, recency, conf_dt=2, conf_hs=3):
    """반환: dict(raw, name, broken, neckline, extreme, desc) 또는 None"""
    n = len(df); last = n-1
    close = float(df['close'].iloc[-1])
    H = df['h']; LW = df['l']
    cands = []

    # 더블탑
    if len(hi) >= 2:
        a, b = hi[-2], hi[-1]
        if last-b <= recency and abs(H.iloc[b]-H.iloc[a])/H.iloc[a] <= 0.03:
            neck = float(LW.iloc[a:b+1].min())
            broken = close < neck
            cands.append(dict(raw=conf_dt if broken else 1, name="더블탑", broken=broken,
                              neckline=neck, extreme=float(max(H.iloc[a], H.iloc[b])), bear=True))
    # 더블바텀
    if len(lo) >= 2:
        a, b = lo[-2], lo[-1]
        if last-b <= recency and abs(LW.iloc[b]-LW.iloc[a])/LW.iloc[a] <= 0.03:
            neck = float(H.iloc[a:b+1].max())
            broken = close > neck
            cands.append(dict(raw=-(conf_dt if broken else 1), name="더블바텀", broken=broken,
                              neckline=neck, extreme=float(min(LW.iloc[a], LW.iloc[b])), bear=False))
    # 헤드앤숄더
    if len(hi) >= 3:
        a, m, b = hi[-3], hi[-2], hi[-1]
        if last-b <= recency and H.iloc[m] > H.iloc[a] and H.iloc[m] > H.iloc[b] \
           and abs(H.iloc[b]-H.iloc[a])/H.iloc[a] <= 0.06:
            neck = float(min(LW.iloc[a:m+1].min(), LW.iloc[m:b+1].min()))
            broken = close < neck
            cands.append(dict(raw=conf_hs if broken else 1, name="헤드앤숄더", broken=broken,
                              neckline=neck, extreme=float(H.iloc[m]), bear=True))
    # 역헤드앤숄더
    if len(lo) >= 3:
        a, m, b = lo[-3], lo[-2], lo[-1]
        if last-b <= recency and LW.iloc[m] < LW.iloc[a] and LW.iloc[m] < LW.iloc[b] \
           and abs(LW.iloc[b]-LW.iloc[a])/LW.iloc[a] <= 0.06:
            neck = float(max(H.iloc[a:m+1].max(), H.iloc[m:b+1].max()))
            broken = close > neck
            cands.append(dict(raw=-(conf_hs if broken else 1), name="역헤드앤숄더", broken=broken,
                              neckline=neck, extreme=float(LW.iloc[m]), bear=False))
    if not cands:
        return None
    best = max(cands, key=lambda c: abs(c['raw']))
    best['desc'] = "%s %s" % (best['name'], "넥라인 돌파" if best['broken'] else "넥라인 미돌파")
    return best

def candle(df):
    o = df['o'].values[-3:]; c = df['close'].values[-3:]
    b1 = c[0]-o[0]; b2 = c[1]-o[1]; b3 = c[2]-o[2]
    r1 = abs(b1)/o[0] if o[0] else 0
    r2 = abs(b2)/o[1] if o[1] else 0
    mid = (o[0]+c[0])/2
    if b1 < 0 and r1 >= 0.01 and r2 <= 0.5*r1 and b3 > 0 and c[2] > mid:
        return -1, "모닝스타 출현, 상승반전 신호"
    if b1 > 0 and r1 >= 0.01 and r2 <= 0.5*r1 and b3 < 0 and c[2] < mid:
        return 1, "이브닝스타 출현, 하락반전 신호"
    return 0, None

def enrich(base, df, wk, mo, has_long):
    hi, lo = swings(df, 3, 90)
    div_s, div_desc = divergence(df, hi, lo)
    base['div_s'] = div_s; base['div_desc'] = div_desc

    close = float(df['close'].iloc[-1])
    pat = find_pattern(df, hi, lo, 25)
    pattern_raw = 0; pattern_desc = None; pattern_s = 0
    base['target'] = None; base['gap'] = None
    base['risk_flag'] = False; base['risk_desc'] = None
    if pat:
        pattern_raw = pat['raw']; pattern_desc = pat['desc']
        # 4) 패턴 소진 보정
        if pattern_raw > 0 and base['pos_score'] <= -2:
            pattern_s = 1; pattern_desc += " (하락 상당부분 진행)"
        elif pattern_raw < 0 and base['pos_score'] >= 2:
            pattern_s = -1; pattern_desc += " (상승 상당부분 진행)"
        else:
            pattern_s = pattern_raw
        if pat['broken']:
            prog = abs(close - pat['neckline'])/pat['neckline']*100
            if prog >= 5:
                pattern_s = (1 if pattern_s > 0 else -1) if pattern_s != 0 else 0
                pattern_desc += " (신호 소진)"
            # 4-1) 목표가
            if pat['bear']:
                target = pat['neckline'] - (pat['extreme'] - pat['neckline'])
                gap = (close - target)/close*100
                base['target'] = round(target, 2); base['gap'] = round(gap, 1)
                if gap > 3:
                    base['risk_flag'] = True
                    base['risk_desc'] = "%s 목표가 %s (추가 %s%% 여지)" % (pat['name'], round(target,2), round(gap,1))
                else:
                    base['risk_flag'] = False
                    base['risk_desc'] = "패턴 목표가 도달, 하락 목표 달성"
            else:
                target = pat['neckline'] + (pat['neckline'] - pat['extreme'])
                gap = (target - close)/close*100
                base['target'] = round(target, 2); base['gap'] = round(gap, 1)
    base['pattern_raw'] = pattern_raw; base['pattern_s'] = pattern_s; base['pattern_desc'] = pattern_desc

    candle_s, candle_desc = candle(df)
    base['candle_s'] = candle_s; base['candle_desc'] = candle_desc

    # 6) 주봉 / 월봉
    wk_s = 0; wk_desc = None; mo_desc = None
    if has_long:
        whi, wlo = swings(wk, 2, 60)
        wpat = find_pattern(wk, whi, wlo, 13, conf_dt=1, conf_hs=1)
        if wpat:
            if wpat['broken']:
                wk_s = 1 if wpat['raw'] > 0 else -1
                wk_desc = "주봉 %s 넥라인 돌파" % wpat['name']
            else:
                wk_s = 0
                wk_desc = "주봉 %s 형성 중" % wpat['name']
            if wk_s > 0 and base['pos_score'] <= -2:
                wk_s = 0; wk_desc += " (진행 상당)"
            elif wk_s < 0 and base['pos_score'] >= 2:
                wk_s = 0; wk_desc += " (진행 상당)"
        mhi, mlo = swings(mo, 2, 60)
        mpat = find_pattern(mo, mhi, mlo, 8, conf_dt=1, conf_hs=1)
        if mpat:
            mo_desc = "월봉 %s %s" % (mpat['name'], "넥라인 돌파" if mpat['broken'] else "형성 중")
    base['wk_s'] = wk_s; base['wk_desc'] = wk_desc; base['mo_desc'] = mo_desc

    signal_score = base['vol_s']+div_s+pattern_s+candle_s+wk_s
    base['signal_score'] = signal_score
    total = base['pos_score'] + signal_score
    base['total'] = total

    items = [base['rsi_s'], base['bb_s'], base['dev_s'], base['p52_s'],
             base['vol_s'], div_s, pattern_s, candle_s, wk_s]
    low_cnt = sum(1 for x in items if x < 0)
    high_cnt = sum(1 for x in items if x > 0)
    mid_cnt = sum(1 for x in items if x == 0)
    strong_low = sum(1 for x in items if x <= -2)
    strong_high = sum(1 for x in items if x >= 2)
    base.update(low_cnt=low_cnt, high_cnt=high_cnt, mid_cnt=mid_cnt,
                strong_low=strong_low, strong_high=strong_high)

    if total <= -6 and low_cnt >= 4 and strong_high == 0:
        verdict = "저점권 🟢"
    elif total <= -3 and low_cnt >= 3 and low_cnt > high_cnt:
        verdict = "저점 근접 🔵"
    elif total >= 6 and high_cnt >= 4 and strong_low == 0:
        verdict = "고점권 🔴"
    elif total >= 3 and high_cnt >= 3 and high_cnt > low_cnt:
        verdict = "고점 근접 🟠"
    else:
        verdict = "중립 구간 ⚪"
    # 4-2) verdict 보정
    verdict_note = None
    if base['risk_flag'] and verdict.startswith(("저점권", "저점 근접")):
        verdict_note = "반등 시도 구간, 구조적 하락 목표 미달"
    base['verdict'] = verdict; base['verdict_note'] = verdict_note
    return base

tickers = ["SPY", "SOXX", "SKHY", "MSTR"]
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
    has_long = len(df) >= 250
    base = enrich(base, df, wk, mo, has_long)
    print(json.dumps(base, ensure_ascii=False))
