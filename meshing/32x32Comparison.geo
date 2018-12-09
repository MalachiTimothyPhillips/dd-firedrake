// This is a good "fake" way to do a partition, for instance
Mesh.Algorithm = 8;
N=1e22;
Point(1)={0,0,0,N};
Point(2)={0,1,0,N};
Point(3)={1,1,0,N};
Point(4)={1,0,0,N};


Line(1)={1,2};
Line(2)={2,3};
Line(3)={3,4};
Line(4)={4,1};

Line Loop(1)={1,2,3,4};
Transfinite Line{1,2}=72;
Transfinite Line{3,4}=72;

Plane Surface(1) = {1};


Physical Line(1)={1};
Physical Line(2)={2};
Physical Line(3)={3};
Physical Line(4)={4};

Physical Surface(1) = {1};
Recombine Surface {1};
