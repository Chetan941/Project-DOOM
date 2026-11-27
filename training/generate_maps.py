"""Training-only UDMF maps, using the official WAD writer."""
from pathlib import Path
from doom_env.wad import Lump, write_wad
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'maps' / 'procedural'

def generate(path, key_type=5, lock=2, key_y=128, angle=0, locked=True):
    blocks = ['namespace = "zdoom";']
    def block(kind, **props):
        def fmt(v):
            if isinstance(v, bool): return str(v).lower()
            if isinstance(v, str): return '"'+v+'"'
            return str(v)
        blocks.append(kind+' { '+' '.join(f'{k} = {fmt(v)};' for k,v in props.items())+' }')
    xs = [0,384,448,768,832]
    for x in xs:
        for y in (0,256): block('vertex', x=float(x), y=float(y))
    for i in range(4):
        block('sector', texturefloor='FLOOR0_1', textureceiling='CEIL1_1', heightfloor=0, heightceiling=0 if i==1 else 128, lightlevel=192)
    sides=0
    def side(sector, middle='STARTAN2', top='-'):
        nonlocal sides
        n=sides; sides+=1
        block('sidedef',sector=sector,texturemiddle=middle,texturetop=top,texturebottom='-')
        return n
    def wall(a,b,s): block('linedef',v1=a,v2=b,sidefront=side(s),blocking=True)
    for i in range(4):
        wall(2*i+1,2*i+3,i); wall(2*i+2,2*i,i)
    wall(0,1,0); wall(9,8,3)
    for i in (1,2,3):
        # Downward line: front is west, back is east.
        p=dict(v1=2*i+1,v2=2*i,sidefront=side(i-1,'-','BIGDOOR2' if i==1 else '-'),sideback=side(i,'-','BIGDOOR2' if i==2 else '-'),twosided=True)
        if i==1: p.update(special=13 if locked else 12,arg0=0,arg1=16,arg2=150,playeruse=True,repeatspecial=True,**({'arg3':lock} if locked else {}))
        if i==3: p.update(special=243,playercross=True)
        block('linedef',**p)
    def thing(x,y,t,a=0): block('thing',x=float(x),y=float(y),type=t,angle=a,skill1=True,skill2=True,skill3=True,skill4=True,skill5=True,single=True)
    thing(96,128,1,angle); thing(224,key_y,key_type)
    write_wad(path,[Lump('MAP01',b''),Lump('TEXTMAP','\n'.join(blocks).encode()),Lump('ENDMAP',b'')])

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    for name,typ,lock in [('blue',5,2),('red',13,1),('yellow',6,3)]:
        for j,(y,a) in enumerate([(128,0),(80,0),(176,0),(128,45)]):
            p=OUT/f'{name}_{j}.wad'; generate(p,typ,lock,y,a); print(p.name)
