import sys
import numpy as np
import cv2
import ctypes
import os
import matplotlib.pyplot as plt
import scipy.ndimage
import torch
import torch.nn.functional as F
import time


def farneback_polyexp_numpy(src: np.ndarray, n: int, sigma: float):
    """
    Numpy copy of the OpenCV FarnebackPolyExp(src, dst, n, sigma)
    src: 2D float32 image
    n: kernel half-width (PolyN)
    sigma: Gaussian sigma
    Returns:
      A: (H, W, 2,2) float32
      B: (H, W, 2)    float32
      C: (H, W)       float32
    """
    H, W = src.shape
    # build 1-D kernels
    k = 2*n + 1
    x = np.arange(-n, n+1).astype(np.float32)
    g   = np.exp(-(x**2)/(2*sigma*sigma))
    g  /= g.sum()
    xg  = x * g
    xxg = (x*x) * g

    # precompute the Farneback constants for G inverse
    # these integrals come from OpenCV’s FarnebackPrepareGaussian 
    ig11 =     np.dot(x*x*g, g)  
    ig03 =     np.dot(g, g)
    ig33 =     np.dot(xxg, g)
    ig55 =     np.dot(xxg, xxg)

    # vertical pass → a (H, W, 3) buffer
    # buf[y,x,:] = [sum_k g[k]*src[y±k,x], sum_k xg[k]*(src[y+k,x]-src[y-k,x]), sum_k xxg[k]*src[y±k,x]]
    buf = np.zeros((H, W, 3), dtype=np.float32)
    buf[:,:,0] = src * g[n]
    for k_ in range(1, n+1):
        gp = g[n+k_]
        xp = xg[n+k_]
        xxp= xxg[n+k_]
        top    = src[np.clip(np.arange(H)-k_, 0, H-1), :]
        bottom = src[np.clip(np.arange(H)+k_, 0, H-1), :]
        sum_   = top + bottom
        diff_  = bottom - top
        buf[:,:,0] += gp * sum_
        buf[:,:,1] += xp * diff_
        buf[:,:,2] += xxp* sum_

    # mirror the horizontal edges in buf so we can do horizontal pass
    #    we only need to fill n cols on each side
    row = np.zeros((H, W+2*n, 3), dtype=np.float32)
    row[:, n:n+W, :] = buf
    # mirror-out at x<0  and x>=W
    row[:, :n, :]   = buf[:, :n, :][:, ::-1, :]
    row[:, n+W:, :] = buf[:, -n:, :][:, ::-1, :]

    # horizontal pass → build A,B,C
    A = np.zeros((H, W, 2,2), dtype=np.float32)
    B = np.zeros((H, W, 2   ), dtype=np.float32)
    C = np.zeros((H, W      ), dtype=np.float32)

    for x in range(W):
        # compute the six b-values for all rows at once using vectorized sums
        # b1,b2,b3,b4,b5,b6 each shape (H*W,)
        g0 = g[n]                  
        b1 = row[:, x+n, 0]*g0     
        b2 = np.zeros(H, np.float32)  
        b3 = row[:, x+n, 1]*g0      
        b4 = np.zeros(H, np.float32)
        b6 = np.zeros(H, np.float32) 
        b5 = row[:, x+n, 2]*g0     

        for k_ in range(1, n+1):
            gp  = g[n+k_]
            xgp = xg[n+k_]
            xxp = xxg[n+k_]
            right = row[:, x+n+k_, :]
            left  = row[:, x+n-k_, :]
            sum0  = right[:, 0] + left[:, 0]   # constant term taps
            sum2  = right[:, 2] + left[:, 2]   # x^2 term taps
            diff0 = right[:, 0] - left[:, 0]   # for Bx
            diff1 = right[:, 1] - left[:, 1]   # for A_xy
            sum1b = right[:, 1] + left[:, 1]   # for By


            b1 += gp  * sum0
            b4 += xxp * sum2
            b2 += xgp * diff0
            b3 += gp  * sum1b
            b6 += xgp * diff1
            b5 += gp  * sum2
        # reverse engineering OpenCV this is how they compute A, B,C
        #   drow[*+0] = (b3*ig11)        → B.y
        #   drow[*+1] = (b2*ig11)        → B.x
        #   drow[*+2] = (b1*ig03 + b5*ig33) → A.yy
        #   drow[*+3] = (b1*ig03 + b4*ig33) → A.xx
        #   drow[*+4] = (b6*ig55)         → 2·A.xy
        B[:, x, 1] = b3 * ig11
        B[:, x, 0] = b2 * ig11

        A[:, x, 1,1] = b1*ig03 + b5*ig33
        A[:, x, 0,0] = b1*ig03 + b4*ig33
        A[:, x,0,1] = (b6*ig55)*0.5
        A[:, x,1,0] = A[:, x,0,1]

        C[:, x] = b1 * ig03 
    
    return A, B, C


def get_raw_moments_torch(frame: np.ndarray, polyN=5, sigma=1.0, device='cpu'):
    # make float32 tensor [1,1,H,W]
    src = torch.from_numpy(frame.astype(np.float32))[None,None].to(device)
    # build 1D kernels
    x = torch.arange(-polyN, polyN+1, dtype=torch.float32, device=device)
    g   = torch.exp(-(x**2)/(2*sigma*sigma))
    g  /= g.sum()
    xg  = x * g
    xxg = (x*x) * g

    # wrap them for conv2d: vertical kernels shape [1,1,k,1], horizontal [1,1,1,k]
    kv_g   = g.view(1,1,-1,1)
    kh_g   = g.view(1,1,1,-1)
    kv_xg  = xg.view(1,1,-1,1)
    kh_xg  = xg.view(1,1,1,-1)
    kv_xxg = xxg.view(1,1,-1,1)
    kh_xxg = xxg.view(1,1,1,-1)

    # zero‐pad exactly polyN on all sides similar to CUDA verison
    pad = (polyN, polyN, polyN, polyN)
    src_p = F.pad(src, pad, mode='constant', value=0)

    # seperate convolutions
    def sep(src, kv, kh):
        tmp = F.conv2d(src, kv, padding=(polyN,0))
        return F.conv2d(tmp, kh, padding=(0,polyN))

    # compute the six moments of intensities of frame 
    C   = sep(src_p, kv_g,   kh_g  )
    Ix  = sep(src_p, kv_xg,  kh_g  )
    Iy  = sep(src_p, kv_g,   kh_xg )
    Ixx = sep(src_p, kv_xxg, kh_g  )
    Iyy = sep(src_p, kv_g,   kh_xxg)
    Ixy = sep(src_p, kv_xg,  kh_xg )

    # remove the padding so shape is [1,1,H,W]
    C,  Ix,  Iy,  Ixx,  Iyy,  Ixy = [
        M[..., polyN:-polyN, polyN:-polyN] for M in (C, Ix, Iy, Ixx, Iyy, Ixy)
    ]

    # squeeze to 2D arrays same shape as input frame now
    return [M.squeeze().cpu().numpy() for M in (C, Ix, Iy, Ixx, Iyy, Ixy)]


# Load the shared library for PolynomialExpansion created by Makefile
lib = ctypes.cdll.LoadLibrary("./PolyExp.so")

# Setup function prototype
lib.polynomialExpansion.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # input frame
    ctypes.c_float,                   # sigma
    ctypes.c_int,                     # width
    ctypes.c_int,                     # height
    ctypes.c_int,                     # polyN
    ctypes.POINTER(ctypes.c_double),  # outC
    ctypes.POINTER(ctypes.c_double),  # outB
    ctypes.POINTER(ctypes.c_double),  # outA
    ctypes.POINTER(ctypes.c_float),   # outdC
    ctypes.POINTER(ctypes.c_float),   # outIx
    ctypes.POINTER(ctypes.c_float),   # outIy
    ctypes.POINTER(ctypes.c_float),   # outIxx
    ctypes.POINTER(ctypes.c_float),   # outIyy
    ctypes.POINTER(ctypes.c_float),   # outIxy
]
def process_frame_with_cuda(frame, polyN=5, frame_idx= 0):
    h, w = frame.shape
    total = h * w
    pad = polyN
    # pad from each side by half filter size = polyN with 0
    padded = np.pad(frame, ((pad,pad),(pad,pad)), mode='constant', constant_values= 0)

    h_pad, w_pad = padded.shape
    total_pad = h_pad * w_pad
    frame_in = np.ascontiguousarray(padded, dtype=np.float32)
    dC   = np.empty((h, w), dtype=np.float32)
    Ix  = np.empty_like(dC)
    Iy  = np.empty_like(dC)
    Ixx = np.empty_like(dC)
    Iyy = np.empty_like(dC)
    Ixy = np.empty_like(dC)
    sigma = 1.0

    # Allocate output arrays
    outC = np.zeros(total, dtype=np.float64)       # c scalar per pixel
    outB = np.zeros(total * 2, dtype=np.float64)   # B vector (2 floats) per pixel
    outA = np.zeros(total * 4, dtype=np.float64)   # A matrix (4 floats) per pixel
    print(f"\n[Frame {frame_idx}] CUDA version:")
    start = time.time()
    # call into CUDA library
    lib.polynomialExpansion(
        frame_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(sigma), 
        ctypes.c_int(w_pad),
        ctypes.c_int(h_pad),
        ctypes.c_int(polyN),
        outC.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        outB.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        outA.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dC.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Ix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Iy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Ixx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Iyy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Ixy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    print(f"CUDA took {time.time() - start:.4f} seconds")
    outC = outC.reshape((h, w))
    outB = outB.reshape((h, w, 2))
    outA = outA.reshape((h, w, 2, 2))

    return outA, outB, outC, dC, Ix, Iy, Ixx, Iyy, Ixy

def main(video):
    cap = cv2.VideoCapture(video)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #frame = cv2.resize(frame,(320,180))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-9)
        pad =5 
        padded = np.pad(frame, ((pad,pad),(pad,pad)), mode='constant', constant_values= 0)
        A_cuda, B_cuda, C_cuda, dC_cuda, Ix_cuda, Iy_cuda, Ixx_cuda, Iyy_cuda, Ixy_cuda = process_frame_with_cuda(frame, 5, frame_idx)
        print("A matrix shape:", A_cuda.shape)
        print("B vector shape:", B_cuda.shape)
        print("C scalar shape:", C_cuda.shape)
        print(f"\n[Frame {frame_idx}] Python version:")
        c = np.ones_like(frame)
        start2 = time.time()
        A_py, B_py, C_py = farneback_polyexp_numpy(frame, n=5, sigma=1.0)
        print(f"Python version took {time.time() - start2:.4f} seconds")
        dC_py,  Ix_py,  Iy_py,  Ixx_py,  Iyy_py,  Ixy_py  = get_raw_moments_torch(frame, polyN=5, sigma=1.0)
        def stats(diff):
            return {
                'mean_abs': np.mean(np.abs(diff)),
                'max_abs':  np.max(np.abs(diff)),
                'rmse':     np.sqrt(np.mean(diff**2))
            }
        # Get the error stats
        err_A = stats(A_cuda - A_py)
        err_B = stats(B_cuda - B_py)
        err_C = stats(C_cuda - C_py)
        names    = ['dC', 'Ix', 'Iy', 'Ixx', 'Iyy', 'Ixy']
        cuda_arr = [dC_cuda, Ix_cuda, Iy_cuda, Ixx_cuda, Iyy_cuda, Ixy_cuda]
        py_arr   = [dC_py,   Ix_py,   Iy_py,   Ixx_py,   Iyy_py,   Ixy_py]

        for name, c_arr, p_arr in zip(names, cuda_arr, py_arr):
            diff = c_arr - p_arr
            e = stats(diff)
            print(f"Error stats for {name}: mean={e['mean_abs']:.6f}, "
                f"max={e['max_abs']:.6f}, rmse={e['rmse']:.6f}")

        print("Error stats for A (2×2 matrix):", err_A)
        print("Error stats for B (2-vector):     ", err_B)
        print("Error stats for C (scalar):       ", err_C)
       
        breakpoint()
        frame_idx +=1

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main("walking.mp4")