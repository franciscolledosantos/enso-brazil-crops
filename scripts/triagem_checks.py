import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
src=open('codigo/fase_fenologica.py').read().split('# ---------------------------------------------------------------- F1')[0]
exec(compile(src,'prep','exec'))
grandes = p[p.ano == 2025].query('prod_t > 2e5')[['cod_uf', 'cultura']]
pg = p.merge(grandes, on=['cod_uf', 'cultura'])
c=pg.groupby(['cultura','sigla']).size(); print(c[c<36])
print(pg.groupby('cultura').agg(states=('sigla','nunique'), n=('ano','size')))
SIG2COD={v:k for k,v in UF_SIGLA.items()}
c2=pd.read_csv('dados_conab/conab_milho2_uf_ano.csv'); c2=c2[(c2.ano>=1990)&(c2.ano<=2025)&c2.rend.notna()&(c2.rend>0)].copy(); c2['cod_uf']=c2.uf.map(SIG2COD); c2['ly']=np.log(c2.rend); c2['sigla']=c2.uf
clima_m=p[p.cultura=='milho'][['cod_uf','ano']+[c+'_z' for c in PH]]
d2=c2[c2.sigla.isin(['MT','PR','MS','GO','SP','MG'])].merge(clima_m,on=['cod_uf','ano']); d2['cultura']='milho2'
y=anomalia(d2); cols=[c+'_z' for c in PH]; m=y.notna()
X=np.column_stack([np.ones(m.sum()),d2.loc[m,cols].values]); b,*_=np.linalg.lstsq(X,y[m].values,rcond=None); r=y[m].values-X@b
print('R2 CONAB nine vars', round(1-r.var()/y[m].var(),3), 'n', m.sum())
d6=pg[(pg.cultura=='milho')&(pg.sigla.isin(['MT','PR','MS','GO','SP','MG']))]
rows=[]
for c in PH:
    dd=d6.dropna(subset=[c+'_z','oni_ndj'])
    b,lo,hi,pv,n=block_boot(dd,['oni_ndj'],lambda x:x[c+'_z'],B=1000)
    v,f=c.split('_'); rows.append(dict(group='maize second crop, six states',phase=f,variable={'P':'rainfall','G':'soil wetness','T':'Tmax'}[v],sd_per_degC=b[0],lo=lo[0],hi=hi[0],p=pv[0],n=n))
F2b=pd.DataFrame(rows); F2b.to_csv('resultados/F2b_enso_fase_seis_estados.csv',index=False); print(F2b.round(2).to_string(index=False))
