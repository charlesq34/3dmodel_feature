function [output] = conv3d(input, w, b, stride, pad)
% input: Ci x D x H x W
% w: Co x Ci x D x H x W
% b: Co
% stride: int
% pad: int

assert(length(size(input)) == 4);
assert(length(size(w)) == 5 || length(size(w)) == 2);
assert(size(input,2) == size(input,3));
assert(size(input,3) == size(input,4));
assert(size(w,2) == size(input,1));


% padding
input = padarray(input, [0, pad, pad, pad]);

S = stride;
if length(size(w)) == 2 % for cccp kernel
    A = 1;
else
    A = size(w,3);
end
C_out = size(w,1);

% create tensor for output
n = floor((size(input,2)-A) / stride + 1);
output = zeros(size(w,1),n,n,n);

% 3d convolution 
for i = 1:n
    for j = 1:n
        for k = 1:n
            patch = input(:,(i-1)*S+1:(i-1)*S+A,...
                    (j-1)*S+1:(j-1)*S+A,...
                    (k-1)*S+1:(k-1)*S+A);
              output(:,i,j,k) = sum(repmat(reshape(patch,1,[]),C_out,1).* ...
                  reshape(w,C_out,[]),2) + reshape(b,[],1);
        end
    end
end

end
