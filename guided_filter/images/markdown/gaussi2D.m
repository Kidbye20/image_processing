clc;
sig = 4.5;
w_x = (-5:0.1:5);
w_y = (-5:0.1:5);
[X,Y] = meshgrid(w_x,w_y);
W = exp(-(X.^2+Y.^2)./sig.^2);
mesh(w_x,w_y,W);
title('sigma = ' + num2str(sig));