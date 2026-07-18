import json, subprocess

# analyze.py 결과를 재실행하여 수집
import importlib.util
spec = importlib.util.spec_from_file_location("az", "analyze.py")
# 대신 stdout 파싱이 간단 -> 직접 재실행
out = subprocess.run(["python3","analyze.py"], capture_output=True, text=True).stdout
rows = [json.loads(l) for l in out.strip().splitlines() if l.strip().startswith("{")]

def sgn(x): return f"{x:+d}"

def rsi_txt(s, r):
    if s <= -2: return "과매도"
    if s == -1: return "약과매도"
    if s >= 2: return "과매수"
    if s == 1: return "약과매수"
    return "중립"

def bb_txt(s):
    return {-2:"하단 이탈", -1:"하단 근접", 2:"상단 이탈", 1:"상단 근접", 0:"중단 중립"}[s]

def dev_txt(s):
    if s <= -1: return "과매도"
    if s >= 1: return "과열"
    return "정상"

def p52_txt(s):
    if s <= -1: return "저점권"
    if s >= 1: return "고점권"
    return "중간"

def vol_txt(vr, s):
    if vr >= 130:
        if s <= -2: return "급증(투매성)"
        if s >= 2: return "급증(과열성)"
        return "급증(방향중립)"
    return "평이"

# 기준 거래일 (가장 늦은 날짜)
dates = [r["date"] for r in rows if "date" in r]
base_date = max(dates)
y,m,d = base_date.split("-")
md = f"{int(m)}/{int(d)}"

lines = [f"📊 기술적 분석 ({md})", "※ 음수=저점권 / 양수=고점권"]

for r in rows:
    lines.append("")  # 종목 사이 빈 줄
    if "error" in r:
        lines.append(f"⚠️ {r['ticker']} ㅣ 데이터 조회 실패")
        continue
    lines.append(f"{r['emoji']} {r['ticker']} ㅣ {r['verdict']} ({sgn(r['total'])}) ㅣ 종가 {r['close']}")
    lines.append(f"ㆍ저점 {r['low_cnt']} / 중립 {r['mid_cnt']} / 고점 {r['high_cnt']}")
    lines.append(f"- RSI {r['rsi']} ({sgn(r['rsi_s'])}) {rsi_txt(r['rsi_s'], r['rsi'])}")
    lines.append(f"- 볼린저밴드 ({sgn(r['bb_s'])}) {bb_txt(r['bb_s'])}")
    lines.append(f"- 이격도 {r['dev']}% ({sgn(r['dev_s'])}) {dev_txt(r['dev_s'])}")
    lines.append(f"- 52주 위치 {r['pos52']}% ({sgn(r['p52_s'])}) {p52_txt(r['p52_s'])}")
    lines.append(f"- 거래량 {r['vratio']}% ({sgn(r['vol_s'])}) {vol_txt(r['vratio'], r['vol_s'])}")
    if r['div_s'] != 0:
        lines.append(f"- 다이버전스 ({sgn(r['div_s'])}) {r['div_desc']}")
    if r['pattern_s'] != 0:
        lines.append(f"- 차트패턴 ({sgn(r['pattern_s'])}) {r['pattern_desc']}")
    if r['candle_s'] != 0:
        lines.append(f"- 캔들패턴 ({sgn(r['candle_s'])}) {r['candle_desc']}")
    if r.get('risk_desc'):
        lines.append(f"- ⚠️ 목표가 {r['risk_desc']}")

msg = "\n".join(lines)
open("message.txt","w").write(msg)
print(msg)
print("\n----- length:", len(msg))
