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

# ===== 2단계-B =====
def extend(base, df):
    n = len(df)
    L = df.iloc[-1]
    start = max(3, n-90)
    swing_highs=[]; swing_lows=[]
    for i in range(start, n-3):
        hi=df['h'].iloc[i]; lo=df['l'].iloc[i]
        if all(hi>df['h'].iloc[i-k] for k in range(1,4)) and all(hi>df['h'].iloc[i+k] for k in range(1,4)):
            swing_highs.append((i, float(hi), float(df['rsi'].iloc[i])))
        if all(lo<df['l'].iloc[i-k] for k in range(1,4)) and all(lo<df['l'].iloc[i+k] for k in range(1,4)):
            swing_lows.append((i, float(lo), float(df['rsi'].iloc[i])))

    # 2) RSI 다이버전스
    div_s=0; div_desc=None
    if len(swing_lows)>=2:
        (i1,p1,r1),(i2,p2,r2)=swing_lows[-2],swing_lows[-1]
        if p2<p1 and r2>r1:
            div_s=-2; div_desc="RSI 강세 다이버전스, 저점 신호"
    if div_s==0 and len(swing_highs)>=2:
        (i1,p1,r1),(i2,p2,r2)=swing_highs[-2],swing_highs[-1]
        if p2>p1 and r2<r1:
            div_s=2; div_desc="RSI 약세 다이버전스, 고점 신호"

    # 3) 차트패턴
    cands=[]
    close=float(L.close)
    if len(swing_highs)>=2:
        (ia,pa,_),(ib,pb,_)=swing_highs[-2],swing_highs[-1]
        if abs(pa-pb)/max(pa,pb)<=0.03 and (n-1-ib)<=25:
            neck=float(df['l'].iloc[ia:ib+1].min()); broke=close<neck
            cands.append(dict(raw=2 if broke else 1, desc="더블탑", broke=broke, neck=neck, head=max(pa,pb), kind='dt'))
    if len(swing_lows)>=2:
        (ia,pa,_),(ib,pb,_)=swing_lows[-2],swing_lows[-1]
        if abs(pa-pb)/max(pa,pb)<=0.03 and (n-1-ib)<=25:
            neck=float(df['h'].iloc[ia:ib+1].max()); broke=close>neck
            cands.append(dict(raw=-2 if broke else -1, desc="더블바텀", broke=broke, neck=neck, head=min(pa,pb), kind='db'))
    if len(swing_highs)>=3:
        (i1,p1,_),(i2,p2,_),(i3,p3,_)=swing_highs[-3],swing_highs[-2],swing_highs[-1]
        if p2>p1 and p2>p3 and abs(p1-p3)/max(p1,p3)<=0.06 and (n-1-i3)<=25:
            neck=float(df['l'].iloc[i1:i3+1].min()); broke=close<neck
            cands.append(dict(raw=3 if broke else 1, desc="헤드앤숄더", broke=broke, neck=neck, head=p2, kind='hs'))
    if len(swing_lows)>=3:
        (i1,p1,_),(i2,p2,_),(i3,p3,_)=swing_lows[-3],swing_lows[-2],swing_lows[-1]
        if p2<p1 and p2<p3 and abs(p1-p3)/max(p1,p3)<=0.06 and (n-1-i3)<=25:
            neck=float(df['h'].iloc[i1:i3+1].max()); broke=close>neck
            cands.append(dict(raw=-3 if broke else -1, desc="역헤드앤숄더", broke=broke, neck=neck, head=p2, kind='ihs'))

    pattern_raw=0; pattern_desc=None; chosen=None
    if cands:
        chosen=max(cands, key=lambda c: abs(c['raw']))
        pattern_raw=chosen['raw']; pattern_desc=chosen['desc']

    # 4) 소진 보정
    pattern_s=pattern_raw
    if pattern_raw>0 and base['pos_score']<=-2:
        pattern_s=1; pattern_desc=(pattern_desc or "")+" (하락 상당부분 진행)"
    elif pattern_raw<0 and base['pos_score']>=2:
        pattern_s=-1; pattern_desc=(pattern_desc or "")+" (상승 상당부분 진행)"
    if chosen and chosen['broke']:
        mv=abs(close-chosen['neck'])/chosen['neck']*100
        if mv>=5 and abs(pattern_s)>1:
            pattern_s=1 if pattern_s>0 else -1
            pattern_desc=(pattern_desc or "")+" (신호 소진)"

    # 4-1) 목표가
    target=None; gap=None; risk_flag=False; risk_desc=None
    if chosen and chosen['broke'] and pattern_raw>0:
        target=chosen['neck']-(chosen['head']-chosen['neck'])
        gap=(close-target)/close*100
        nm="더블탑" if chosen['kind']=='dt' else "H&S"
        if gap>3:
            risk_flag=True; risk_desc=f"{nm} 목표가 {round(target,2)} (추가 {round(gap,1)}% 여지)"
        else:
            risk_flag=False; risk_desc="패턴 목표가 도달, 하락 목표 달성"
        target=round(target,2); gap=round(gap,1)
    elif chosen and chosen['broke'] and pattern_raw<0:
        target=chosen['neck']+(chosen['neck']-chosen['head'])
        gap=(close-target)/close*100
        target=round(target,2); gap=round(gap,1)

    # 5) 캔들패턴
    o1,c1=float(df['o'].iloc[-3]),float(df['c'].iloc[-3])
    o2,c2=float(df['o'].iloc[-2]),float(df['c'].iloc[-2])
    o3,c3=float(df['o'].iloc[-1]),float(df['c'].iloc[-1])
    b1=abs(c1-o1); b2=abs(c2-o2); mid1=(o1+c1)/2
    candle_s=0; candle_desc=None
    if c1<o1 and b1>0 and b2<0.5*b1 and c3>o3 and c3>mid1:
        candle_s=-1; candle_desc="모닝스타 출현, 상승반전 신호"
    elif c1>o1 and b1>0 and b2<0.5*b1 and c3<o3 and c3<mid1:
        candle_s=1; candle_desc="이브닝스타 출현, 하락반전 신호"

    signal_score=base['vol_s']+div_s+pattern_s+candle_s
    total=base['pos_score']+signal_score
    items=[base['rsi_s'],base['bb_s'],base['dev_s'],base['p52_s'],base['vol_s'],div_s,pattern_s,candle_s]
    low_cnt=sum(1 for x in items if x<0); high_cnt=sum(1 for x in items if x>0)
    mid_cnt=sum(1 for x in items if x==0)
    strong_low=sum(1 for x in items if x<=-2); strong_high=sum(1 for x in items if x>=2)

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

    verdict_adj=None
    if risk_flag and verdict in ("저점권","저점 근접"):
        verdict_adj="반등 시도 구간, 구조적 하락 목표 미달"

    base.update(div_s=div_s, div_desc=div_desc, pattern_raw=pattern_raw, pattern_s=pattern_s,
        pattern_desc=pattern_desc, pattern_broke=(chosen['broke'] if chosen else None),
        candle_s=candle_s, candle_desc=candle_desc, signal_score=signal_score, total=total,
        low_cnt=low_cnt, high_cnt=high_cnt, mid_cnt=mid_cnt, strong_low=strong_low, strong_high=strong_high,
        verdict=verdict, emoji=emoji, target=target, gap=gap, risk_flag=risk_flag, risk_desc=risk_desc,
        verdict_adj=verdict_adj)
    return base

tickers = ["SPY","SOXX","MSTR"]
results=[]
for t in tickers:
    try:
        raw=json.load(open(f"{t}.json"))["data"]
        b,df=analyze(t,raw)
        b=extend(b,df)
        results.append(b)
        print(json.dumps(b, ensure_ascii=False))
    except Exception as e:
        results.append({"ticker":t,"error":str(e)})
        print(json.dumps({"ticker":t,"error":str(e)}, ensure_ascii=False))

json.dump(results, open("results.json","w"), ensure_ascii=False)
