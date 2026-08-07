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

def find_swings(d, k, lookback):
    """좌우 k봉보다 고가가 모두 높으면 스윙고점, 저가가 모두 낮으면 스윙저점."""
    n = len(d)
    hi = d['h'].values; lo = d['l'].values
    start = max(k, n - lookback)
    sh, sl = [], []
    for i in range(start, n - k):
        nb = list(range(i-k, i)) + list(range(i+1, i+k+1))
        if all(hi[i] > hi[j] for j in nb): sh.append(i)
        if all(lo[i] < lo[j] for j in nb): sl.append(i)
    return sh, sl


def divergence(d, sh, sl):
    if len(sl) >= 2:
        a, b = sl[-2], sl[-1]
        if d['l'].iloc[b] < d['l'].iloc[a] and d['rsi'].iloc[b] > d['rsi'].iloc[a]:
            return -2, "RSI 강세 다이버전스, 저점 신호"
    if len(sh) >= 2:
        a, b = sh[-2], sh[-1]
        if d['h'].iloc[b] > d['h'].iloc[a] and d['rsi'].iloc[b] < d['rsi'].iloc[a]:
            return 2, "RSI 약세 다이버전스, 고점 신호"
    return 0, None


def detect_patterns(d, sh, sl, recency, hns_break=3, dbl_break=2, forming=1):
    """반환: (score, name, broken, neckline, extreme_price)  없으면 None"""
    n = len(d); close = float(d['close'].iloc[-1])
    cands = []

    # 더블탑
    if len(sh) >= 2:
        a, b = sh[-2], sh[-1]
        pa, pb_ = d['h'].iloc[a], d['h'].iloc[b]
        if abs(pa-pb_)/max(pa, pb_) <= 0.03 and (n-1-b) <= recency and b > a:
            neck = float(d['l'].iloc[a:b+1].min())
            broken = close < neck
            cands.append(((dbl_break if broken else forming), "더블탑", broken, neck, float(max(pa, pb_))))
    # 더블바텀
    if len(sl) >= 2:
        a, b = sl[-2], sl[-1]
        pa, pb_ = d['l'].iloc[a], d['l'].iloc[b]
        if abs(pa-pb_)/max(pa, pb_) <= 0.03 and (n-1-b) <= recency and b > a:
            neck = float(d['h'].iloc[a:b+1].max())
            broken = close > neck
            cands.append(((-dbl_break if broken else -forming), "더블바텀", broken, neck, float(min(pa, pb_))))
    # 헤드앤숄더
    if len(sh) >= 3:
        a, b, c = sh[-3], sh[-2], sh[-1]
        ha, hb, hc = d['h'].iloc[a], d['h'].iloc[b], d['h'].iloc[c]
        if hb > ha and hb > hc and abs(ha-hc)/max(ha, hc) <= 0.06 and (n-1-c) <= recency:
            neck = float(min(d['l'].iloc[a:b+1].min(), d['l'].iloc[b:c+1].min()))
            broken = close < neck
            cands.append(((hns_break if broken else forming), "헤드앤숄더", broken, neck, float(hb)))
    # 역헤드앤숄더
    if len(sl) >= 3:
        a, b, c = sl[-3], sl[-2], sl[-1]
        la, lb, lc = d['l'].iloc[a], d['l'].iloc[b], d['l'].iloc[c]
        if lb < la and lb < lc and abs(la-lc)/max(la, lc) <= 0.06 and (n-1-c) <= recency:
            neck = float(max(d['h'].iloc[a:b+1].max(), d['h'].iloc[b:c+1].max()))
            broken = close > neck
            cands.append(((-hns_break if broken else -forming), "역헤드앤숄더", broken, neck, float(lb)))

    if not cands: return None
    cands.sort(key=lambda x: abs(x[0]), reverse=True)
    return cands[0]


def candle(d):
    o = d['o'].values; c = d['close'].values
    if len(d) < 3: return 0, None
    o1, c1, o2, c2, o3, c3 = o[-3], c[-3], o[-2], c[-2], o[-1], c[-1]
    b1 = abs(c1-o1); b2 = abs(c2-o2); b3 = abs(c3-o3); mid1 = (o1+c1)/2
    if c1 < o1 and b2 < b1*0.5 and c3 > o3 and c3 > mid1 and b3 > b2:
        return -1, "모닝스타 출현, 상승반전 신호"
    if c1 > o1 and b2 < b1*0.5 and c3 < o3 and c3 < mid1 and b3 > b2:
        return 1, "이브닝스타 출현, 하락반전 신호"
    return 0, None


def enrich(base, df, wk, mo, short_hist=False):
    close = base['close']; pos_score = base['pos_score']

    sh, sl = find_swings(df, 3, 90)
    base['div_s'], base['div_desc'] = divergence(df, sh, sl)

    p = detect_patterns(df, sh, sl, 25)
    if p:
        raw, name, broken, neck, extreme = p
        desc = f"{name} {'넥라인 돌파' if broken else '넥라인 미돌파'}"
        s = raw
        if raw > 0 and pos_score <= -2:
            s = 1; desc += " (하락 상당부분 진행)"
        elif raw < 0 and pos_score >= 2:
            s = -1; desc += " (상승 상당부분 진행)"
        if broken:
            prog = abs(close-neck)/neck*100
            if prog >= 5 and abs(s) > 1:
                s = 1 if s > 0 else -1
                desc += " (신호 소진)"
        base['pattern_raw'] = raw; base['pattern_s'] = s
        base['pattern_desc'] = desc; base['pattern_name'] = name; base['pattern_broken'] = broken
        # 4-1) 목표가
        if broken:
            target = neck - (extreme - neck) if raw > 0 else neck + (neck - extreme)
            gap = (close - target)/close*100
            base['target'] = round(target, 2); base['gap'] = round(gap, 1)
            if raw > 0:
                if gap > 3:
                    base['risk_flag'] = True
                    base['risk_desc'] = f"{name} 목표가 {round(target,2)} (추가 {round(gap,1)}% 여지)"
                else:
                    base['risk_flag'] = False
                    base['risk_desc'] = "패턴 목표가 도달, 하락 목표 달성"
    else:
        base['pattern_raw'] = 0; base['pattern_s'] = 0; base['pattern_desc'] = None
    base.setdefault('risk_flag', False); base.setdefault('risk_desc', None)
    base.setdefault('target', None); base.setdefault('gap', None)

    base['candle_s'], base['candle_desc'] = candle(df)

    # 주봉 / 월봉
    base['wk_s'] = 0; base['wk_desc'] = None; base['mo_desc'] = None
    if not short_hist:
        wk = wk.copy(); wk['t'] = pd.to_datetime(wk['t'])
        wsh, wsl = find_swings(wk, 2, 60)
        wp = detect_patterns(wk, wsh, wsl, 13, hns_break=1, dbl_break=1, forming=0)
        if wp:
            raw, name, broken, neck, extreme = wp
            if broken:
                ws = 1 if raw > 0 else -1
                wdesc = f"주봉 {name} 넥라인 돌파"
            else:
                ws = 0
                wdesc = f"주봉 {name} 형성 중"
            if ws > 0 and pos_score <= -2:
                ws = 0; wdesc += " (진행 상당)"
            elif ws < 0 and pos_score >= 2:
                ws = 0; wdesc += " (진행 상당)"
            base['wk_s'] = ws; base['wk_desc'] = wdesc
        msh, msl = find_swings(mo, 2, 60)
        mp = detect_patterns(mo, msh, msl, 8, hns_break=1, dbl_break=1, forming=0)
        if mp:
            _, name, broken, _, _ = mp
            base['mo_desc'] = f"월봉 {name} {'넥라인 돌파' if broken else '형성 중'}"

    base['signal_score'] = base['vol_s']+base['div_s']+base['pattern_s']+base['candle_s']+base['wk_s']
    base['total'] = pos_score + base['signal_score']

    items = [base[k] for k in ('rsi_s','bb_s','dev_s','p52_s','vol_s','div_s','pattern_s','candle_s','wk_s')]
    base['low_cnt'] = sum(1 for x in items if x < 0)
    base['high_cnt'] = sum(1 for x in items if x > 0)
    base['mid_cnt'] = sum(1 for x in items if x == 0)
    base['strong_low'] = sum(1 for x in items if x <= -2)
    base['strong_high'] = sum(1 for x in items if x >= 2)

    tt = base['total']; lc = base['low_cnt']; hc = base['high_cnt']
    if tt <= -6 and lc >= 4 and base['strong_high'] == 0: v = "저점권 🟢"
    elif tt <= -3 and lc >= 3 and lc > hc: v = "저점 근접 🔵"
    elif tt >= 6 and hc >= 4 and base['strong_low'] == 0: v = "고점권 🔴"
    elif tt >= 3 and hc >= 3 and hc > lc: v = "고점 근접 🟠"
    else: v = "중립 구간 ⚪"
    base['verdict_adj'] = None
    if base['risk_flag'] and v.startswith(("저점권", "저점 근접")):
        base['verdict_adj'] = "반등 시도 구간, 구조적 하락 목표 미달"
    base['verdict'] = v
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
    short = len(df) < 300
    base = enrich(base, df, wk, mo, short_hist=short)
    base['bars'] = len(df)
    print(json.dumps(base, ensure_ascii=False))
