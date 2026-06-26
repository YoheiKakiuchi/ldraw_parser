import pickle
exec(open('irsl_code.py').read())

with open('parts/data.pkl', 'rb') as f:
  data=pickle.load(f)
newdata = convData(data)

def getPos(dat):
  return 0.01*dat[0][:3, 3]

def getRot(dat):
  return dat[1]

def makeCoords(dat):
  res=coordinates(getPos(dat))
  res.rot = getRot(dat)
  return res

exec(open('/home/irsl/temp/irsl_cnoid_plugin/samples/pick_obj.py').read())
di = DrawInterface()
po = PickedObject()

def draw_parts(name, color=[0.8, 0.15, 0.15], drawLegend=True):
    obj = mkshapes.loadMesh(f'parts/{name}.stl', color=color)
    di.addObject(obj)
    for idx, k in enumerate(newdata[name].keys()):
        for n, p in enumerate(newdata[name][k]):
            pos = getPos(p)
            bx = make_box(idx, length=0.02)
            bx.object.name = f'{k}%{n}'
            c = coordinates(pos)
            c.transform(bx)
            bx.newcoords(c)
            po.addObject(bx)
    po.genShapeMap()
    ### draw text
    if drawLegend:
      mat = ib.getCameraMatrix()
      cam, fov = ib.getCameraCoords()
      ivmat = np.linalg.inv(mat)
      sv = ib.currentSceneView()
      width = sv.width()
      height = sv.height()
      base = 6 * ivmat @ fv(width*0.01, height*0.05, 1)
      cam.transformVector(base)
      for idx, k in enumerate(newdata[name].keys()):
          bx = make_box(idx, length=0.02)
          txt = mkshapes.makeText(k, textHeight=0.14) ##
          pos = base + (cam.y_axis * idx * 0.21)
          bx.locate(pos, coordinates.wrt.world)
          cds=ib.makeCameraFacingCoords(pos)
          cds.translate(fv(0.03, -0.07, 0)) ##
          txt.newcoords(cds)
          di.addObject(bx)
          di.addObject(txt)



#### Data
n_pin_ = '3673.dat'
n_br08 = '3702.dat'
n_br12 = '3895.dat'
n_br16 = '3703.dat'
n_bm__ = '32009.dat'
n_axl8 = '3707.dat'
n_bsh0 = '32039.dat'

nlst = [
  n_pin_,
  n_br08,
  n_br12,
  n_br16,
  n_bm__,
  n_axl8,
  n_bsh0,
  ]

def common_drawings(nameA, nameB):
  sA = set(newdata[nameA].keys())
  sB = set(newdata[nameB].keys())
  return list(sA.intersection(sB))

#>sA = set(newdata[n_bm__].keys())
#>sB = set(newdata[n_pin_].keys())
#>res = list(sA.intersection(sB))
#>
#>p_pin_ = newdata[n_pin_][res[0]] # pin
#>p_br08 = newdata[n_br08][res[0]] # brick 8
#>p_br12 = newdata[n_br12][res[0]] # brick 12
#>p_br16 = newdata[n_br16][res[0]] # brick 16
#>p_bm__ = newdata[n_bm__][res[0]] # brick 16
#>p_axl8 = newdata[n_axl8][res[0]] # axle 8
#>p_bsh0 = newdata[n_bsh0][res[0]] # bush
#>
#>q_pin_ = mkshapes.loadMesh(f'parts/{n_pin_}.stl', color=[0.7, 0.7, 0.7])
#>q_br08 = mkshapes.loadMesh(f'parts/{n_br08}.stl', color=[0.9, 0.1, 0.1])
#>q_br12 = mkshapes.loadMesh(f'parts/{n_br12}.stl', color=[0.1, 0.1, 0.9])
#>q_br16 = mkshapes.loadMesh(f'parts/{n_br16}.stl', color=[0.9, 0.1, 0.1])
#>q_bm__ = mkshapes.loadMesh(f'parts/{n_bm__}.stl', color=[0.1, 0.1, 0.9])
#>q_axl8 = mkshapes.loadMesh(f'parts/{n_axl8}.stl', color=[0.3, 0.3, 0.3])
#>q_bsh0 = mkshapes.loadMesh(f'parts/{n_bsh0}.stl', color=[0.7, 0.7, 0.7])

#>di.addObject(q_br08)
#>for idx, k in enumerate(newdata[n_br08].keys()):
#>  for n, p in enumerate(newdata[n_br08][k]):
#>    pos = getPos(p)
#>    bx = make_box(idx, length=0.02)
#>    bx.object.name = f'{k}%{n}'
#>    bx.locate(pos, coordinates.wrt.world)
#>    po.addObject(bx)
#>
#>for idx, k in enumerate(newdata[n_br08].keys()):
#>    bx = make_box(idx, length=0.02)
#>    txt = mkshapes.makeText(k, textHeight=0.14) ##
#>    pos = fv(-4, -4, idx*0.21)
#>    bx.locate(pos, coordinates.wrt.world)
#>    cds=ib.makeCameraFacingCoords(pos)
#>    cds.translate(fv(0.03, -0.07, 0)) ##
#>    txt.newcoords(cds)
#>    di.addObject(bx)
#>    di.addObject(txt)
#>
#>##
#>di.addObject(q_pin_)
#>for idx, k in enumerate(newdata[n_pin_].keys()):
#>  for n, p in enumerate(newdata[n_pin_][k]):
#>    pos = getPos(p)
#>    bx = make_box(idx, length=0.02)
#>    bx.object.name = f'{k}%{n}'
#>    bx.locate(pos, coordinates.wrt.world)
#>    di.addObject(bx)
#>
#>lst=[]
#>di.addObject(q_br08)
#>for idx, k in enumerate(newdata[n_br08].keys()):
#>  for n, p in enumerate(newdata[n_br08][k]):
#>    pos = getPos(p)
#>    bx = make_box(idx, length=0.02)
#>    lst.append(bx)
#>    bx.object.name = f'{k}%{n}'
#>    bx.locate(pos, coordinates.wrt.world)
#>    di.addObject(bx)
#>
#>po = PickedObject(di=di)
#>po.genShapeMap()

