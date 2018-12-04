// This is a good "fake" way to do a partition, for instance
Mesh.Algorithm = 8;
N=1e22;
Point(1)={0,0,0,N};
Point(2)={0.33,0,0,N};
Point(3)={0.66,0,0,N};
Point(4)={1.0,0,0,N};

// x points along y=1
Point(5)={0,1,0,N};
Point(6)={0.33,1,0,N};
Point(7)={0.66,1,0,N};
Point(8)={1.0,1,0,N};

// vertical lines first
Line(1)={1,5};
Line(2)={2,6};
Line(3)={3,7};
Line(4)={4,8};

// horizontal line segments, along y=0
Line(5)={1,3};
Line(6)={2,4};

// horizontal line segments, along y=1
Line(7)={5,7};
Line(8)={6,8};

// define line loops
Line Loop(1)={1,7,-3,-5};
Line Loop(2)={2,8,-4,-6}; // not sure about directionality here for outward normal

Plane Surface(1) = {1};
Plane Surface(2) = {2}; // based on line loops from above


Physical Line(1)={1};
Physical Line(2)={2};
Physical Line(3)={3};
Physical Line(4)={4};
Physical Line(5)={5};
Physical Line(6)={6};
Physical Line(7)={7};
Physical Line(8)={8};

Physical Surface(1) = {1};
Physical Surface(2) = {2};
Recombine Surface {1};
Recombine Surface {2};
