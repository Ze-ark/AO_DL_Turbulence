function g = ift2_ao(G, delta_f)
%IFT2_AO Centered inverse Fourier transform used by phase-screen synthesis.
N = size(G, 1);
g = ifftshift(ifft2(ifftshift(G))) * (N * delta_f)^2;
end
