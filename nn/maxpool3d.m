function [output] = maxpool3d(input, kernel_size, stride)
% input: C x D x H x W

assert(length(size(input)) == 4);
assert(size(input,2) == size(input,3));
assert(size(input,3) == size(input,4));

% allocate memory for output
n = floor((size(input,2)-kernel_size) / stride + 1);
output = zeros(size(input,1),n,n,n);

% max poolin
A = kernel_size;
S = stride;
C_in = size(input,1);

for i = 1:n
    for j = 1:n
        for k = 1:n
            tmp =  input(:,(i-1)*S+1:(i-1)*S+A,...
                    (j-1)*S+1:(j-1)*S+A,...
                    (k-1)*S+1:(k-1)*S+A);
            tmp = reshape(tmp,C_in,A^3);
            output(:,i,j,k) = max(tmp,[],2);
        end
    end
end

