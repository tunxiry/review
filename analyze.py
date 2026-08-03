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

def find_swings(df, k, lookback):
    """좌우 k봉보다 고가가 모두 높으면 스윙고점, 저가가 모두 낮으면 스윙저점."""
    n = len(df)
    hi = df['h'].values; lo = df['l'].values
    start = max(k, n - lookback)
    sh, sl = [], []
    for i in range(start, n-k):
        if all(hi[i] > hi[j] for j in range(i-k, i)) and all(hi[i] > hi[j] for j in range(i+1, i+k+1)):
            sh.append(i)
        if all(lo[i] < lo[j] for j in range(i-k, i)) and all(lo[i] < lo[j] for j in range(i+1, i+k+1)):
            sl.append(i)
    return sh, sl


def divergence(df, sh, sl):
    if len(sl) >= 2:
        a, b = sl[-2], sl[-1]
        if df['l'].iloc[b] < df['l'].iloc[a] and df['rsi'].iloc[b] > df['rsi'].iloc[a]:
            return -2, "RSI 강세 다이버전스, 저점 신호"
    if len(sh) >= 2:
        a, b = sh[-2], sh[-1]
        if df['h'].iloc[b] > df['h'].iloc[a] and df['rsi'].iloc[b] < df['rsi'].iloc[a]:
            return 2, "RSI 약세 다이버전스, 고점 신호"
    return 0, None


def detect_patterns(df, sh, sl, recency, strong=True):
    """반환: list of dict(score, name, broke, neckline, extreme)
    strong=False면 H&S 계열도 ±1/±2 스케일(주봉용)이 아니라 호출측에서 처리."""
    n = len(df)
    close = float(df['close'].iloc[-1])
    out = []
    # 더블탑
    if len(sh) >= 2 and (n-1 - sh[-1]) <= recency:
        a, b = sh[-2], sh[-1]
        ha, hb = df['h'].iloc[a], df['h'].iloc[b]
        if abs(hb-ha)/ha <= 0.03:
            neck = float(df['l'].iloc[a:b+1].min())
            broke = close < neck
            out.append(dict(score=2 if broke else 1, name="더블탑", broke=broke,
                            neck=neck, extreme=float(max(ha, hb)), bear=True))
    # 더블바텀
    if len(sl) >= 2 and (n-1 - sl[-1]) <= recency:
        a, b = sl[-2], sl[-1]
        la, lb = df['l'].iloc[a], df['l'].iloc[b]
        if abs(lb-la)/la <= 0.03:
            neck = float(df['h'].iloc[a:b+1].max())
            broke = close > neck
            out.append(dict(score=-2 if broke else -1, name="더블바텀", broke=broke,
                            neck=neck, extreme=float(min(la, lb)), bear=False))
    # 헤드앤숄더
    if len(sh) >= 3 and (n-1 - sh[-1]) <= recency:
        a, b, c = sh[-3], sh[-2], sh[-1]
        ha, hb, hc = df['h'].iloc[a], df['h'].iloc[b], df['h'].iloc[c]
        if hb > ha and hb > hc and abs(hc-ha)/ha <= 0.06:
            neck = float(min(df['l'].iloc[a:b+1].min(), df['l'].iloc[b:c+1].min()))
            broke = close < neck
            out.append(dict(score=3 if broke else 1, name="헤드앤숄더", broke=broke,
                            neck=neck, extreme=float(hb), bear=True))
    # 역헤드앤숄더
    if len(sl) >= 3 and (n-1 - sl[-1]) <= recency:
        a, b, c = sl[-3], sl[-2], sl[-1]
        la, lb, lc = df['l'].iloc[a], df['l'].iloc[b], df['l'].iloc[c]
        if lb < la and lb < lc and abs(lc-la)/la <= 0.06:
            neck = float(max(df['h'].iloc[a:b+1].max(), df['h'].iloc[b:c+1].max()))
            broke = close > neck
            out.append(dict(score=-3 if broke else -1, name="역헤드앤숄더", broke=broke,
                            neck=neck, extreme=float(lb), bear=False))
    if not out:
        return None
    out.sort(key=lambda p: abs(p['score']), reverse=True)
    return out[0]


def candle(df):
    o = df['o'].values; c = df['close'].values
    b1o, b1c = o[-3], c[-3]; b2o, b2c = o[-2], c[-2]; b3o, b3c = o[-1], c[-1]
    body = abs(c - o)
    avg = body[-22:-2].mean()
    b1 = abs(b1c-b1o); b2 = abs(b2c-b2o)
    mid1 = (b1o+b1c)/2
    if b1c < b1o and b1 >= avg and b2 <= 0.5*b1 and b3c > b3o and b3c > mid1:
        return -1, "모닝스타 출현, 상승반전 신호"
    if b1c > b1o and b1 >= avg and b2 <= 0.5*b1 and b3c < b3o and b3c < mid1:
        return 1, "이브닝스타 출현, 하락반전 신호"
    return 0, None


def enrich(base, df, wk, mo, fallback_1y=False):
    sh, sl = find_swings(df, 3, 90)
    base['div_s'], base['div_desc'] = divergence(df, sh, sl)

    close = float(df['close'].iloc[-1])
    pos_score = base['pos_score']
    p = detect_patterns(df, sh, sl, 25)
    base['target'] = None; base['gap'] = None
    base['risk_flag'] = False; base['risk_desc'] = None
    if p is None:
        base['pattern_raw'] = 0; base['pattern_s'] = 0; base['pattern_desc'] = None
    else:
        raw = p['score']
        desc = f"{p['name']} " + ("넥라인 돌파" if p['broke'] else "넥라인 미돌파")
        s = raw
        if raw > 0 and pos_score <= -2:
            s = 1; desc += " (하락 상당부분 진행)"
        elif raw < 0 and pos_score >= 2:
            s = -1; desc += " (상승 상당부분 진행)"
        # 돌파 이후 5% 이상 추가 진행 시 신호 소진
        if p['broke']:
            prog = abs(close - p['neck'])/p['neck']*100
            if prog >= 5 and abs(s) > 1:
                s = 1 if s > 0 else -1
                desc += " (신호 소진)"
        base['pattern_raw'] = raw; base['pattern_s'] = s; base['pattern_desc'] = desc
        # 목표가
        if p['broke']:
            target = p['neck'] - (p['extreme'] - p['neck'])
            gap = (close - target)/close*100
            base['target'] = round(target, 2); base['gap'] = round(gap, 1)
            if p['bear']:
                if gap > 3:
                    base['risk_flag'] = True
                    base['risk_desc'] = f"{p['name']} 목표가 {round(target,2)} (추가 {round(gap,1)}% 여지)"
                else:
                    base['risk_flag'] = False
                    base['risk_desc'] = "패턴 목표가 도달, 하락 목표 달성"

    base['candle_s'], base['candle_desc'] = candle(df)

    # 주봉 / 월봉
    base['wk_s'] = 0; base['wk_desc'] = None; base['mo_desc'] = None
    if not fallback_1y:
        wsh, wsl = find_swings(wk, 2, 60)
        wp = detect_patterns(wk, wsh, wsl, 13)
        if wp:
            wk_s = 0
            if wp['broke']:
                wk_s = 1 if wp['score'] > 0 else -1
            desc = f"주봉 {wp['name']} " + ("넥라인 돌파" if wp['broke'] else "형성 중")
            if wk_s > 0 and pos_score <= -2:
                wk_s = 0; desc += " (진행 상당)"
            elif wk_s < 0 and pos_score >= 2:
                wk_s = 0; desc += " (진행 상당)"
            base['wk_s'] = wk_s; base['wk_desc'] = desc
        msh, msl = find_swings(mo, 2, 60)
        mp = detect_patterns(mo, msh, msl, 8)
        if mp:
            base['mo_desc'] = f"월봉 {mp['name']} " + ("넥라인 돌파" if mp['broke'] else "형성 중")

    signal_score = base['vol_s']+base['div_s']+base['pattern_s']+base['candle_s']+base['wk_s']
    total = pos_score + signal_score
    base['signal_score'] = signal_score; base['total'] = total

    items = [base['rsi_s'], base['bb_s'], base['dev_s'], base['p52_s'], base['vol_s'],
             base['div_s'], base['pattern_s'], base['candle_s'], base['wk_s']]
    low_cnt = sum(1 for x in items if x < 0)
    high_cnt = sum(1 for x in items if x > 0)
    mid_cnt = sum(1 for x in items if x == 0)
    strong_low = sum(1 for x in items if x <= -2)
    strong_high = sum(1 for x in items if x >= 2)
    base.update(low_cnt=low_cnt, high_cnt=high_cnt, mid_cnt=mid_cnt,
                strong_low=strong_low, strong_high=strong_high)

    if total <= -6 and low_cnt >= 4 and strong_high == 0:
        v = "저점권 🟢"
    elif total <= -3 and low_cnt >= 3 and low_cnt > high_cnt:
        v = "저점 근접 🔵"
    elif total >= 6 and high_cnt >= 4 and strong_low == 0:
        v = "고점권 🔴"
    elif total >= 3 and high_cnt >= 3 and high_cnt > low_cnt:
        v = "고점 근접 🟠"
    else:
        v = "중립 구간 ⚪"
    if base['risk_flag'] and v.startswith(("저점권", "저점 근접")):
        v += " (반등 시도 구간, 구조적 하락 목표 미달)"
    base['verdict'] = v
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
