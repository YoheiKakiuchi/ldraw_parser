exec(open('/choreonoid_ws/install/share/irsl_choreonoid/sample/irsl_import.py').read())

exec(open('irsl_code.py').read())

import pickle
import ldraw_parser.ldraw_parser as lp
lib = lp.Library.create('ldraw')

di = DrawInterface()

dumpData()

#>  %autoindent
#>  import ldraw_parser.ldraw_parser as lp
#>  di=DrawInterface()
#>  def makeMeshes(pt, scale=0.0004):
#>      objs = []
#>      indices_4=[0, 1, 2, 0, 2, 3]
#>      indices_3=[0, 1, 2]
#>      for p4 in pt.getQuads():
#>          mat = scale * p4.copy(order='C')
#>          objs.append( mkshapes.makeTriangles( mat, indices_4) )
#>      for p3 in pt.getTriangles():
#>          mat = scale * p3.copy(order='C')
#>          objs.append( mkshapes.makeTriangles( mat, indices_3) )
#>      return objs
#>  
#>  lib=lp.Library.create('ldraw')
#>  
#>  ptype = lib.loadParts('parts_name')
#>  ptinst = lp.instantiate(ptype, lib) ## -> lib.instantiate(ptype) / lib.loadParts. -> (ptype, inst)
#>  
#>  objs=makeMeshes(ptinst)
#>  di.addObjects(objs)
#>  
#>  ## export stl
#>  mkshapes.exportMesh('/tmp/hoge.stl', di.SgPosTransform, outputType='stlb')

