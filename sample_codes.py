exec(open('/choreonoid_ws/install/share/irsl_choreonoid/sample/irsl_import.py').read())

%autoindent
import ldraw_parser.ldraw_parser as lp
di=DrawInterface()
def makeMeshes(pt, scale=0.0004):
    objs = []
    indices_4=[0, 1, 2, 0, 2, 3]
    indices_3=[0, 1, 2]
    for p4 in pt.getQuads():
        mat = scale * p4.copy(order='C')
        objs.append( mkshapes.makeTriangles( mat, indices_4) )
    for p3 in pt.getTriangles():
        mat = scale * p3.copy(order='C')
        objs.append( mkshapes.makeTriangles( mat, indices_3) )
    return objs

lib=lp.Library.create('ldraw')

ptype = lib.loadParts('parts_name')
ptinst = lp.instantiate(ptype, lib) ## -> lib.instantiate(ptype) / lib.loadParts. -> (ptype, inst)

objs=makeMeshes(ptinst)
di.addObjects(objs)

## export stl
mkshapes.exportMesh('/tmp/hoge.stl', di.SgPosTransform, outputType='stlb')


### testing
###

# parts to be checked
parts_list = [
    # Axle 4, 6, 8
    '3705.dat',
    '3706.dat',
    '3707.dat',
    # Axle stop 3, 4, 5
    '24316.dat',
    '87083.dat',
    '15462.dat',
    # Axle Pin
    '3749.dat',
    # Axle Pin long
    '65249.dat',
    # Technic Beam 3x7
    '32009.dat',
    # Brick 1x2, 1x8, 1x12, 1x16
    '3700.dat',
    '3701.dat',
    '3894.dat',
    '3702.dat',
    '3895.dat',
    '3703.dat',
    # Connector Axle/Bush
    '32039.dat',
    # Pin
    '3673.dat',
    # Pin long
    '32556.dat',
    # Pin Joiner
    '62462.dat',
    # Beam 2x0.5
    '41677.dat',
    # Bush with two
    '3713.dat',
    # Beam 3 x 5 90
    '32526.dat',
    # Beam 5
    '32316.dat',
    # Beam 7
    '32524.dat',
    # Beam 9
    '40490.dat',
    # Beam 2 x 4 Liftarm
    '32140.dat',
    # Beam Liftarm triangle
    '99773.dat',
    # Beam 3,4,5 x 0.5 liftarm
    '6632.dat',
    '32449.dat',
    '11478.dat',
    # Beam 3 x 7 liftarm bend
    '32271.dat',
    # Beam 4 x 4 liftarm bend
    '32348.dat',
    # Beam 4 x 6 liftarm bend
    '6629.dat',
    # Beam 4 x 3 liftarm triangle
    '43464.dat',
    # Beam 2 liftarm
    '60483.dat',
    # Beam 3 with Center Axle
    '7229.dat',
    # Bush 1/2
    '32123.dat',
    # Cross block 2 x 2 split (Axle/Twin Pin)
    '41678.dat',
    # Cross block 2 x 2 (Axle/Twin Pin)
    '32291.dat',
    # Cross block 2 x 3 (Pin/Pin/Twin Pin)
    '32557.dat',
    # Cross block 3 x 2 (Axle/Tripple Pin)
    '63869.dat',
    # Cross block 1 x 3 (Axle/Pin/Pin)
    '42003.dat',
    # Cross block 1 x 3 (Axle/Pin/Pin)
    '42003.dat',
    # Cross block 1 x 3 (Axle/Pin/Axle)
    '32184.dat',
    # Cross block 1 x 2 (Axle/Pin)
    '6536.dat',
    # Plate 2x4 hole
    '3709b.dat',
    # <gears>
    # Gear 40
    '3649.dat',
    # Gear 24
    '3648b.dat',
    # Gear 8
    '3647.dat',
    ]

# locations to be checked
_cc_list_=('peghole.dat', 'axle.dat', 'axlehole.dat', 'axlehol5.dat', 'axlehol4.dat', 'axl2hole.dat', 'connect.dat', 'connect8.dat', '4-4cyli.dat')

### locations have to be checked INVERT 
def checkCC(cc):
    if cc[1] in _cc_list_:
        return True
        if np.linalg.det( cc[0][:3, :3] > 0 ):
            return True
    return False

for pname in parts_list:
    pt=lp.instantiate(lib.loadParts(pname), lib)
    di.clear()
    objs=makeMeshes(pt, scale=0.01)
    di.addObjects(objs)
    mkshapes.exportMesh('/tmp/{}.stl'.format(pname), di.SgPosTransform, outputType='stlb')
    aa=[ cc for cc in pt.getLocations() if checkCC(cc) ]
    if len(aa) > 0:
        cdslst=[ mkshapes.makeCoords(coords=coordinates(0.01*a[0][:3, 3])) for a in aa ]
        di.addObjects(cdslst)
    IU.processEvent()
    print(pname)
    print(aa)
    print('\n\n')
    aa = input()
