# create a square mesh
import numpy as np
import sys

def generate_mesh(Nx,Ny,d,output):
    mesh_template="""Mesh.Algorithm = 8;
N=1e22;
MY_OWN_CODE_GOES_HERE
    """
    d=d/2
    dx=1/Nx
    dy=1/Ny
    lines=[]
    pointStr="Point({:d})={{{:f},{:f},0,N}};"
    lineStr="Line({})={{{:d},{:d}}};"
    pointCtr = 0
    lineCtr = 0
    def new_node_id():
        new_node_id.node += 1
        return new_node_id.node
    new_node_id.node = 0
    def new_vertex_id():
        new_vertex_id.v += 1
        return new_vertex_id.v
    new_vertex_id.v = 0
    pts_x=[0]
    for i in range(1,Nx):
        x=i*dx
        lhs=x-d
        rhs=x+d
        pts_x.append(lhs)
        pts_x.append(rhs)
    pts_x.append(1)
    pts_y=[0]
    for i in range(1,Nx):
        y=i*dy
        lhs=y-d
        rhs=y+d
        pts_y.append(lhs)
        pts_y.append(rhs)
    pts_y.append(1)
    pts=np.zeros((len(pts_x),len(pts_y),2))
    for i in range(len(pts_x)):
        for j in range(len(pts_y)):
            pts[i][j][0]=pts_x[i]
            pts[i][j][1]=pts_y[j]
    lineLoopStr="Line Loop ({:d})={{{:d},{:d},{:d},{:d}}};"
    elements=1
    physicalLineStr="Physical Line({:d})={{{:d}}};"
    planeSurfStr="Plane Surface({:d})={{{:d}}};"
    physicalSurfaceStr="Physical Surface({:d})={{{:d}}};"
    recombStr="Recombine Surface {{{:d}}};"
    idx_x=[[0,2]]
    v=1
    for ix in range(1,Nx-1):
        idx_x.append([v,v+3])
        v+=2
    idx_x.append([v,v+2])
    idx_y=[[0,2]]
    v=1
    for iy in range(1,Ny-1):
        idx_y.append([v,v+3])
        v+=2
    idx_y.append([v,v+2])
    for iy in range(Ny):
        for ix in range(Nx):
            n1=new_node_id()
            n2=new_node_id()
            n3=new_node_id()
            n4=new_node_id()
            x0=idx_x[ix][0]
            x1=idx_x[ix][1]
            y0=idx_y[iy][0]
            y1=idx_y[iy][1]
            e1=[n1,[pts[x0][y0][0],pts[x0][y0][1]]] 
            e2=[n2,[pts[x1][y0][0],pts[x0][y0][1]]] 
            e3=[n3,[pts[x0][y0][0],pts[x0][y1][1]]]
            e4=[n4,[pts[x1][y1][0],pts[x1][y1][1]]] 
            nodes=[e1,e2,e3,e4]
            v1=new_vertex_id()
            v2=new_vertex_id()
            v3=new_vertex_id()
            v4=new_vertex_id()
            vertices=[ [v1,e1,e2],[v2,e1,e3],[v3,e3,e4],[v4,e2,e4]  ]
            for node in nodes:
                lines.append(pointStr.format(node[0],node[1][0],node[1][1]))
            for vertex in vertices:
                lines.append(lineStr.format(vertex[0],vertex[1][0],vertex[2][0]))
                lines.append(physicalLineStr.format(vertex[0],vertex[0]))
            lines.append(lineLoopStr.format(elements,-vertices[0][0],vertices[1][0],vertices[2][0],-vertices[3][0]))
            lines.append(planeSurfStr.format(elements,elements))
            lines.append(physicalSurfaceStr.format(elements,elements))
            elements += 1
    
    for element in range(elements-1):
        e=element+1
        lines.append(recombStr.format(e,e))
    mesh_code=mesh_template.replace("MY_OWN_CODE_GOES_HERE","\n".join(lines))
    text_file=open(output,"w")
    text_file.write(mesh_code)
    text_file.close()
def generate_structured_mesh(Nx,Ny,output,nelems):
    mesh_template="""Mesh.Algorithm = 8;
N=1e22;
MY_OWN_CODE_GOES_HERE
    """
    num_in_x=np.sqrt(nelems)
    num_elems_in_domain=num_in_x/Nx+1
    d=1/num_in_x
    dx=1/Nx
    dy=1/Ny
    lines=[]
    pointStr="Point({:d})={{{:f},{:f},0,N}};"
    lineStr="Line({})={{{:d},{:d}}};"
    pointCtr = 0
    lineCtr = 0
    def new_node_id():
        new_node_id.node += 1
        return new_node_id.node
    new_node_id.node = 0
    def new_vertex_id():
        new_vertex_id.v += 1
        return new_vertex_id.v
    new_vertex_id.v = 0
    pts_x=[0]
    for i in range(1,Nx):
        x=i*dx
        lhs=x-d
        rhs=x+d
        pts_x.append(lhs)
        pts_x.append(rhs)
    pts_x.append(1)
    pts_y=[0]
    for i in range(1,Nx):
        y=i*dy
        lhs=y-d
        rhs=y+d
        pts_y.append(lhs)
        pts_y.append(rhs)
    pts_y.append(1)
    pts=np.zeros((len(pts_x),len(pts_y),2))
    for i in range(len(pts_x)):
        for j in range(len(pts_y)):
            pts[i][j][0]=pts_x[i]
            pts[i][j][1]=pts_y[j]
    lineLoopStr="Line Loop ({:d})={{{:d},{:d},{:d},{:d}}};"
    elements=1
    physicalLineStr="Physical Line({:d})={{{:d}}};"
    planeSurfStr="Plane Surface({:d})={{{:d}}};"
    physicalSurfaceStr="Physical Surface({:d})={{{:d}}};"
    recombStr="Recombine Surface {{{:d}}};"
    transfiniteLineStr="Transfinite Line{{{},{}}}={}"
    idx_x=[[0,2]]
    v=1
    for ix in range(1,Nx-1):
        idx_x.append([v,v+3])
        v+=2
    idx_x.append([v,v+2])
    idx_y=[[0,2]]
    v=1
    for iy in range(1,Ny-1):
        idx_y.append([v,v+3])
        v+=2
    idx_y.append([v,v+2])
    for iy in range(Ny):
        for ix in range(Nx):
            n1=new_node_id()
            n2=new_node_id()
            n3=new_node_id()
            n4=new_node_id()
            x0=idx_x[ix][0]
            x1=idx_x[ix][1]
            y0=idx_y[iy][0]
            y1=idx_y[iy][1]
            e1=[n1,[pts[x0][y0][0],pts[x0][y0][1]]] 
            e2=[n2,[pts[x1][y0][0],pts[x0][y0][1]]] 
            e3=[n3,[pts[x0][y0][0],pts[x0][y1][1]]]
            e4=[n4,[pts[x1][y1][0],pts[x1][y1][1]]] 
            nodes=[e1,e2,e3,e4]
            v1=new_vertex_id()
            v2=new_vertex_id()
            v3=new_vertex_id()
            v4=new_vertex_id()
            vertices=[ [v1,e1,e2],[v2,e1,e3],[v3,e3,e4],[v4,e2,e4]  ]
            for node in nodes:
                lines.append(pointStr.format(node[0],node[1][0],node[1][1]))
            for vertex in vertices:
                lines.append(lineStr.format(vertex[0],vertex[1][0],vertex[2][0]))
                lines.append(physicalLineStr.format(vertex[0],vertex[0]))
            lines.append(lineLoopStr.format(elements,-vertices[0][0],vertices[1][0],vertices[2][0],-vertices[3][0]))
            lines.append(planeSurfStr.format(elements,elements))
            # generate transfinite line markers
            lines.append(transfiniteLineStr.format(vertices[0][0],vertices[1][0],num_elems_in_domain))
            lines.append(physicalSurfaceStr.format(elements,elements))
            elements += 1
    
    for element in range(elements-1):
        e=element+1
        lines.append(recombStr.format(e,e))
    mesh_code=mesh_template.replace("MY_OWN_CODE_GOES_HERE","\n".join(lines))
    text_file=open(output,"w")
    text_file.write(mesh_code)
    text_file.close()
