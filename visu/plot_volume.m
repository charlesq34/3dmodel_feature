function plot_volume(volume, point_size)
% PLOT_VOLUME
%   Usage: plot_volume(volume) % volume is DxDxD 3d array
%   Usage: plot_volume(volume, point_size) % point_size specify how large
%   each point is
%   point colors represent depth (distance from origin on X-Y plane)
%
% Author: Charles R. Qi
% Date: May 1, 2016
%

if nargin < 2
    point_size = 50;
end

[X,Y,Z] = size(volume);

cnt = 1;
n = sum(volume(:) > 0);
points = zeros(n,3);
values = zeros(n,1);

for i = 1:X
for j = 1:Y
for k = 1:Z
    if volume(i,j,k) > 0
        points(cnt,:) = [i,j,k];
        values(cnt) = volume(i,j,k);
        cnt = cnt + 1;
    end
end
end
end
c = sqrt((points(:,1)).^2 + (points(:,2)).^2);
scatter3(points(:,1), points(:,2), points(:,3),point_size * values, c, 'filled'); 
axis([1 X 1 Y 1 Z]); grid on; view([-45 25]);
