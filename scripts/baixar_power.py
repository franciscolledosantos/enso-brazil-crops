import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # run relative to the repository root
_os.makedirs('resultados', exist_ok=True); _os.makedirs('dados/power', exist_ok=True)
import pandas as pd, subprocess, os, time, json
c=pd.read_csv('dados/celulas_clima.csv')
P='PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M,GWETROOT,ALLSKY_SFC_SW_DWN'
for _,r in c.iterrows():
    f=f'dados/power/c_{r.clat:+.1f}_{r.clon:+.1f}.json'
    if os.path.exists(f) and os.path.getsize(f)>5000: continue
    u=f'https://power.larc.nasa.gov/api/temporal/monthly/point?parameters={P}&community=AG&longitude={r.clon}&latitude={r.clat}&start=1981&end=2025&format=JSON'
    for k in range(3):
        subprocess.run(['curl','-s','-m','120','-o',f,u])
        if os.path.exists(f) and os.path.getsize(f)>5000: break
        time.sleep(5)
    print(f, os.path.getsize(f) if os.path.exists(f) else 0, flush=True)
print('FIM')
