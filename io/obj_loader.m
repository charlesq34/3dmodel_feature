function FV = obj_loader(filename, theta, axis, stretch)
% Load .obj 3d model, return FV with vertices and faces
% filename is .obj file name
% theta is viewpoint (azimuth)
FV = struct();
[FV.vertices, FV.faces] = read_obj(filename);

% do some translation and rotation
center = (max(FV.vertices) + min(FV.vertices)) / 2;
FV.vertices = bsxfun(@minus, FV.vertices, center);
if exist('axis', 'var')
    switch axis
        case 'x',
            FV.vertices(:,1) = FV.vertices(:,1) * stretch;
        case 'y',
            FV.vertices(:,2) = FV.vertices(:,2) * stretch;
        case 'z',
            FV.vertices(:,3) = FV.vertices(:,3) * stretch;
        otherwise,
            error('obj_loader axis set wrong');
    end
end

% make the object upright
R = [1 0 0; 0 cos(-pi/2) -sin(-pi/2); 0 sin(-pi/2) cos(-pi/2)];
FV.vertices = FV.vertices * R;