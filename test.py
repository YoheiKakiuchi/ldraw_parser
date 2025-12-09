
beam(blue)
'32009.dat.stl'

pin-axle(blue)
'3749.dat.stl'

pin (gray)
'3673.dat.stl'

brik
8 (red)
'3702.dat.stl'
12 (blue)
'3895.dat.stl'
16 (red)
'3703.dat.stl'

Bush gray
'3713.dat.stl'

axle 8
'3707.dat.stl'


scl, rot = IC.computeScalingRotation(data['3673.dat'][2][0][:3, :3])

np.linalg.det( rot )

for a,b,c in data['3673.dat']: ## pin
  rot, scl = IC.computeRotationScaling(a[:3, :3])
  print(scl, rot, np.linalg.det( rot ), b, c)

for a,b,c in data['3702.dat']: ## brick
  rot, scl = IC.computeRotationScaling(a[:3, :3])
  print(scl, rot, np.linalg.det( rot ), b, c)  

n_pin_ = '3673.dat'
n_br08 = '3702.dat'
n_br12 = '3895.dat'
n_br16 = '3703.dat'
n_bm__ = '32009.dat'
n_axl8 = '3707.dat'
n_bsh0 = '32039.dat'

p_pin_ = newdata[n_pin_][res[0]] # pin 
p_br8_ = newdata[n_br08][res[0]] # brick 8
p_br12 = newdata[n_br12][res[0]] # brick 12
p_br16 = newdata[n_br16][res[0]] # brick 16
p_bm__ = newdata[n_bm__][res[0]] # brick 16
p_axl8 = newdata[n_axl8][res[0]] # axle 8
p_bsh0 = newdata[n_bsh0][res[0]] # bush

q_pin_ = mkshapes.loadMesh(f'parts/{n_pin_}.stl', color=[0.7, 0.7, 0.7])
q_br8_ = mkshapes.loadMesh(f'parts/{n_br08}.stl', color=[0.9, 0.1, 0.1])
q_br12 = mkshapes.loadMesh(f'parts/{n_br12}.stl', color=[0.1, 0.1, 0.9])
q_br16 = mkshapes.loadMesh(f'parts/{n_br16}.stl', color=[0.9, 0.1, 0.1])
q_bm__ = mkshapes.loadMesh(f'parts/{n_bm__}.stl', color=[0.1, 0.1, 0.9])
q_axl8 = mkshapes.loadMesh(f'parts/{n_axl8}.stl', color=[0.3, 0.3, 0.3])
q_bsh0 = mkshapes.loadMesh(f'parts/{n_bsh0}.stl', color=[0.7, 0.7, 0.7])

di.addObjects((q_pin_, q_br8_, q_br12, q_br16, q_bm__))

sA = set(newdata[n_bm__].keys())
sB = set(newdata[n_pin_].keys())
res = list(sA.intersection(sB))


# placeObj(q_bm__, q_pin_, makeCoords(p_bm__[6]), makeCoords(p_pin_[0])) # pin
# placeObj(q_bm__, q_pin_, makeCoords(p_bm__[5]), makeCoords(p_pin_[0])) # bar
#
# placeObj(q_br12, q_pin_, makeCoords(p_br12[4]), makeCoords(p_pin_[0])) ## front
# placeObj(q_br12, q_pin_, makeCoords(p_br12[10]), makeCoords(p_pin_[0])) ## back
#
# placeObj(q_br8_, q_pin_, makeCoords(p_br8_[0]), makeCoords(p_pin_[0])) ## cross
# placeObj(q_br8_, q_pin_, makeCoords(p_br8_[2]), makeCoords(p_pin_[0])) ## top
#
# placeObj(q_br16, q_pin_, makeCoords(p_br16[5]), makeCoords(p_pin_[0])) # front-1
# placeObj(q_br16, q_pin_, makeCoords(p_br16[6]), makeCoords(p_pin_[0])) # front
# placeObj(q_br16, q_pin_, makeCoords(p_br16[10]), makeCoords(p_pin_[0])) # cross

placeObj(q_br16, q_pin_, makeCoords(p_br16[5]), makeCoords(p_pin_[0]))
q_pin_.translate(fv(-0.02, 0, 0))
placeObj(q_pin_, q_bm__, makeCoords(p_pin_[1]), makeCoords(p_bm__[6]))
q_bm__.translate(-0.02*q_pin_.x_axis, coordinates.wrt.world)

pin_cds = mkshapes.makeCoords(coords=coordinates(q_pin_.pos, q_pin_.rot))
di.addObject(pin_cds)
pin_cds.assoc(q_bm__)
pin_cds.rotate(PI/2, coordinates.Z, q_pin_)

# move q_bm__

mat, rot, scl, name, inv, key = newdata['3702.dat']['8.0_2.0_8.0_4-4cyli.dat'][0]
mat, rot, scl, name, inv, key = newdata['3702.dat']['8.0_-2.0_8.0_4-4cyli.dat'][0]
mat, rot, scl, name, inv, key = newdata['3702.dat']['1.0_0.0_0.0_peghole.dat'][0]
 
lst = []
for mat, rot, scl, name, inv, key in newdata['3702.dat']['8.0_2.0_8.0_4-4cyli.dat']:
  pos = 0.01*mat[:3, 3]
  lst.append(mkshapes.makeCoords(coords=coordinates(pos, rot)))

for mat, rot, scl, name, inv, key in newdata['3702.dat']['8.0_2.0_8.0_4-4cyli.dat']:
  pos = 0.01*mat[:3, 3]
  print(inv, np.linalg.det(scl))
  lst.append(mkshapes.makeCoords(coords=coordinates(pos, rot)))
