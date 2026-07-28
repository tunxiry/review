#!/bin/bash
cd /home/user/review
T_TODAY="2026-07-28"
for attempt in 1 2; do
  sleep 120
  for TK in SPY SOXX MSTR; do
    curl -sS "https://stockanalysis.com/api/symbol/s/$TK/history?range=1Y&period=Daily" -o "$TK.json"
  done
  LATEST=$(python3 -c "
import json
ds=[]
for tk in ['SPY','SOXX','MSTR']:
    try:
        ds.append(json.load(open(tk+'.json'))['data'][-1]['t'])
    except: pass
print(max(ds) if ds else 'none')
")
  echo "attempt $attempt latest=$LATEST"
  if [ "$LATEST" = "$T_TODAY" ]; then echo "UPDATED"; break; fi
done
echo "DONE"
