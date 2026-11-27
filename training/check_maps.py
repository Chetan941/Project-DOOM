"""Scripted smoke test, training maps only. Never imported by submission."""
from pathlib import Path
import numpy as np
from doom_env import make_env
ROOT=Path(__file__).resolve().parents[1]
for name in ('blue','red','yellow'):
    path=ROOT/'maps'/'procedural'/f'{name}_0.wad'
    env=make_env(str(path),'MAP01',episode_ticks=2100,render_buffers=False)
    try:
        obs,_=env.reset(seed=42)
        print(name,'initial key entities:',int(((obs['visible_entities']['entity_type']==9)&(obs['visible_entities']['mask']!=0)).sum()),flush=True)
        key=False; door=False; exited=False
        for i in range(400):
            a=np.zeros(9,dtype=np.int8); a[0]=1
            # Pulse USE only near the first door, with time for it to open.
            state=env.privileged_state()
            if 310<state['x']<385 and i%40==0: a[7]=1
            obs,_,term,trunc,_=env.step(a)
            state=env.privileged_state()
            key |= bool(state['keys'].any()); door |= bool(state['x']>448); exited |= bool(state['exit_reached'])
            if term or trunc: break
        print(f'{name}: key={key} crossed_door={door} exit={exited} x={state["x"]:.1f}',flush=True)
        assert key and door and exited, f'{name}: smoke test failed'
    finally: env.close()
