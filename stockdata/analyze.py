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
def find_swings(df, lookback=90, k=3):
    n = len(df)
    start = max(k, n-lookback)
    highs=[]; lows=[]
    h=df['h'].values; lo=df['l'].values
    for i in range(start, n-k):
        wl_h = h[i-k:i]; wr_h = h[i+1:i+1+k]
        if h[i] > wl_h.max() and h[i] > wr_h.max():
            highs.append(i)
        wl_l = lo[i-k:i]; wr_l = lo[i+1:i+1+k]
        if lo[i] < wl_l.min() and lo[i] < wr_l.min():
            lows.append(i)
    return highs, lows

def extend(t, data):
    base, df = analyze(t, data)
    n=len(df)
    highs, lows = find_swings(df)
    close=df['close'].values; h=df['h'].values; lo=df['l'].values; rsi=df['rsi'].values
    last_i = n-1
    cur_close = close[-1]

    # 2) RSI 다이버전스
    div_s=0; div_desc=None
    if len(lows)>=2:
        a,b=lows[-2],lows[-1]
        if lo[b] < lo[a] and rsi[b] > rsi[a]:
            div_s=-2; div_desc="RSI 강세 다이버전스, 저점 신호"
    if len(highs)>=2:
        a,b=highs[-2],highs[-1]
        if h[b] > h[a] and rsi[b] < rsi[a]:
            div_s=2; div_desc="RSI 약세 다이버전스, 고점 신호"
    base['div_s']=div_s; base['div_desc']=div_desc

    # 3) 차트패턴 후보 (raw)
    cands=[]  # (abs, raw, desc, meta)
    # 더블탑 (하락반전형, +)
    if len(highs)>=2:
        a,b=highs[-2],highs[-1]
        diff=abs(h[b]-h[a])/h[a]*100
        recent=(last_i-b)<=25
        if diff<=3 and recent:
            trough=[j for j in lows if a<j<b]
            neck=min([lo[j] for j in trough]) if trough else lo[a:b+1].min()
            broke = cur_close < neck
            raw = 2 if broke else 1
            cands.append((abs(raw), raw, "더블탑", dict(neck=neck, top=max(h[a],h[b]), broke=broke, kind='dt')))
    # 더블바텀 (상승반전형, -)
    if len(lows)>=2:
        a,b=lows[-2],lows[-1]
        diff=abs(lo[b]-lo[a])/lo[a]*100
        recent=(last_i-b)<=25
        if diff<=3 and recent:
            peak=[j for j in highs if a<j<b]
            neck=max([h[j] for j in peak]) if peak else h[a:b+1].max()
            broke = cur_close > neck
            raw = -2 if broke else -1
            cands.append((abs(raw), raw, "더블바텀", dict(neck=neck, bot=min(lo[a],lo[b]), broke=broke, kind='db')))
    # 헤드앤숄더 (하락반전형, +): 최근 스윙고점 3개
    if len(highs)>=3:
        a,b,c=highs[-3],highs[-2],highs[-1]
        recent=(last_i-c)<=25
        if h[b]>h[a] and h[b]>h[c] and abs(h[a]-h[c])/h[a]*100<=6 and recent:
            trough=[lo[j] for j in lows if a<j<c]
            neck=sum(trough)/len(trough) if trough else lo[a:c+1].min()
            broke = cur_close < neck
            raw = 3 if broke else 1
            cands.append((abs(raw), raw, "헤드앤숄더", dict(neck=neck, head=h[b], broke=broke, kind='hs')))
    # 역헤드앤숄더 (상승반전형, -)
    if len(lows)>=3:
        a,b,c=lows[-3],lows[-2],lows[-1]
        recent=(last_i-c)<=25
        if lo[b]<lo[a] and lo[b]<lo[c] and abs(lo[a]-lo[c])/lo[a]*100<=6 and recent:
            peak=[h[j] for j in highs if a<j<c]
            neck=sum(peak)/len(peak) if peak else h[a:c+1].max()
            broke = cur_close > neck
            raw = -3 if broke else -1
            cands.append((abs(raw), raw, "역헤드앤숄더", dict(neck=neck, head=lo[b], broke=broke, kind='ihs')))

    pattern_raw=0; pattern_desc=None; meta=None
    if cands:
        cands.sort(key=lambda x:(-x[0]))
        _, pattern_raw, pattern_desc, meta = cands[0]

    # 4) 패턴 소진 보정
    pos_score=base['pos_score']
    if pattern_raw>0 and pos_score<=-2:
        pattern_s=1; pattern_desc=(pattern_desc or "")+" (하락 상당부분 진행)"
    elif pattern_raw<0 and pos_score>=2:
        pattern_s=-1; pattern_desc=(pattern_desc or "")+" (상승 상당부분 진행)"
    else:
        pattern_s=pattern_raw
    # 넥라인 돌파 확정 후 5% 이상 추가 진행 시 소진
    if meta and meta.get('broke'):
        neck=meta['neck']
        prog=abs(cur_close-neck)/neck*100
        if prog>=5 and pattern_s!=0:
            pattern_s = (1 if pattern_s>0 else -1)
            pattern_desc=(pattern_desc or "")+" (신호 소진)"
    base['pattern_raw']=pattern_raw; base['pattern_s']=pattern_s; base['pattern_desc']=pattern_desc
    base['pattern_broke']= bool(meta.get('broke')) if meta else False

    # 4-1) 목표가 (하락반전형 확정 패턴에만)
    target=None; gap=None; risk_flag=False; risk_desc=None
    if meta and meta.get('broke') and pattern_raw>0:
        neck=meta['neck']
        if meta['kind']=='hs':
            target = neck - (meta['head'] - neck)
            label="H&S"
        else:  # dt
            target = neck - (meta['top'] - neck)
            label="더블탑"
        gap = (cur_close - target)/cur_close*100
        if gap>3:
            risk_flag=True
            risk_desc=f"{label} 목표가 {round(target,2)} (추가 {round(gap,1)}% 여지)"
        else:
            risk_flag=False
            risk_desc="패턴 목표가 도달, 하락 목표 달성"
    elif meta and meta.get('broke') and pattern_raw<0:
        neck=meta['neck']
        if meta['kind']=='ihs':
            target = neck + (neck - meta['head'])
        else:  # db
            target = neck + (neck - meta['bot'])
        gap = (cur_close - target)/cur_close*100
    base['target']=round(target,2) if target is not None else None
    base['gap']=round(gap,1) if gap is not None else None
    base['risk_flag']=risk_flag; base['risk_desc']=risk_desc

    # 5) 캔들패턴 (최근 3거래일)
    candle_s=0; candle_desc=None
    o=df['o'].values
    bodies=abs(close-o)
    avg_body=pd.Series(bodies).tail(10).mean()
    o1,c1=o[-3],close[-3]; o2,c2=o[-2],close[-2]; o3,c3=o[-1],close[-1]
    b1=abs(c1-o1); b2=abs(c2-o2); b3=abs(c3-o3)
    mid1=(o1+c1)/2
    # 모닝스타: 큰 음봉 → 작은 몸통 → 1일차 몸통 중간값 위로 마감 양봉
    if c1<o1 and b1>=avg_body and b2<=0.5*b1 and c3>o3 and c3>mid1:
        candle_s=-1; candle_desc="모닝스타 출현, 상승반전 신호"
    # 이브닝스타: 큰 양봉 → 작은 몸통 → 1일차 몸통 중간값 아래로 마감 음봉
    elif c1>o1 and b1>=avg_body and b2<=0.5*b1 and c3<o3 and c3<mid1:
        candle_s=1; candle_desc="이브닝스타 출현, 하락반전 신호"
    base['candle_s']=candle_s; base['candle_desc']=candle_desc

    # scores
    signal_score=base['vol_s']+div_s+pattern_s+candle_s
    total=pos_score+signal_score
    base['signal_score']=signal_score; base['total']=total

    items=[base['rsi_s'],base['bb_s'],base['dev_s'],base['p52_s'],base['vol_s'],div_s,pattern_s,candle_s]
    low_cnt=sum(1 for x in items if x<0)
    high_cnt=sum(1 for x in items if x>0)
    mid_cnt=sum(1 for x in items if x==0)
    strong_low=sum(1 for x in items if x<=-2)
    strong_high=sum(1 for x in items if x>=2)
    base.update(low_cnt=low_cnt,high_cnt=high_cnt,mid_cnt=mid_cnt,strong_low=strong_low,strong_high=strong_high)

    # verdict
    if total<=-6 and low_cnt>=4 and strong_high==0:
        verdict="저점권"; emoji="🟢"
    elif total<=-3 and low_cnt>=3 and low_cnt>high_cnt:
        verdict="저점 근접"; emoji="🔵"
    elif total>=6 and high_cnt>=4 and strong_low==0:
        verdict="고점권"; emoji="🔴"
    elif total>=3 and high_cnt>=3 and high_cnt>low_cnt:
        verdict="고점 근접"; emoji="🟠"
    else:
        verdict="중립 구간"; emoji="⚪"
    # 4-2) verdict 보정
    verdict_note=None
    if risk_flag and verdict in ("저점권","저점 근접"):
        verdict_note="반등 시도 구간, 구조적 하락 목표 미달"
    base['verdict']=verdict; base['emoji']=emoji; base['verdict_note']=verdict_note
    return base

tickers = ["QQQ","SOXX","AAPL","MSTR","BTC"]
for t in tickers:
    try:
        raw = json.load(open(f"{t}.json"))["data"]
        base = extend(t, raw)
        print(json.dumps(base, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"ticker": t, "error": str(e)}, ensure_ascii=False))
