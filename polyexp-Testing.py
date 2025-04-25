import sys
import numpy as np
import cv2
import ctypes
import os
import matplotlib.pyplot as plt
import scipy.ndimage
import torch
import torch.nn.functional as F

def poly_exp(f, c, sigma):
    """
    Calculates the local polynomial expansion of a 2D signal, as described by Farneback

    Uses separable normalized correlation

    $f ~ x^T A x + B^T x + C$

    If f[i, j] and c[i, j] are the signal value and certainty of pixel (i, j) then
    A[i, j] is a 2x2 array representing the quadratic term of the polynomial, B[i, j]
    is a 2-element array representing the linear term, and C[i, j] is a scalar
    representing the constant term.

    Parameters
    ----------
    f
        Input signal
    c
        Certainty of signal
    sigma
        Standard deviation of applicability Gaussian kernel

    Returns
    -------
    A
        Quadratic term of polynomial expansion
    B
        Linear term of polynomial expansion
    C
        Constant term of polynomial expansion
    """
    # Calculate applicability kernel (1D because it is separable)
    n = int(4 * sigma + 1)
    x = np.arange(-n, n + 1, dtype=int)
    a = np.exp(-(x**2) / (2 * sigma**2))  # a: applicability kernel [n]

    # b: calculate b from the paper. Calculate separately for X and Y dimensions
    # [n, 6]
    bx = np.stack(
        [np.ones(a.shape), x, np.ones(a.shape), x**2, np.ones(a.shape), x], axis=-1
    )
    by = np.stack(
        [
            np.ones(a.shape),
            np.ones(a.shape),
            x,
            np.ones(a.shape),
            x**2,
            x,
        ],
        axis=-1,
    )

    # Pre-calculate product of certainty and signal
    cf = c * f

    # G and v are used to calculate "r" from the paper: v = G*r
    # r is the parametrization of the 2nd order polynomial for f
    G = np.empty(list(f.shape) + [bx.shape[-1]] * 2)
    v = np.empty(list(f.shape) + [bx.shape[-1]])

    # Apply separable cross-correlations

    # Pre-calculate quantities recommended in paper
    ab = np.einsum("i,ij->ij", a, bx)
    abb = np.einsum("ij,ik->ijk", ab, bx)

    # Calculate G and v for each pixel with cross-correlation
    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[..., i, j] = scipy.ndimage.correlate1d(
                c, abb[..., i, j], axis=0, mode="constant", cval=0
            )

        v[..., i] = scipy.ndimage.correlate1d(
            cf, ab[..., i], axis=0, mode="constant", cval=0
        )

    # Pre-calculate quantities recommended in paper
    ab = np.einsum("i,ij->ij", a, by)
    abb = np.einsum("ij,ik->ijk", ab, by)

    # Calculate G and v for each pixel with cross-correlation
    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[..., i, j] = scipy.ndimage.correlate1d(
                G[..., i, j], abb[..., i, j], axis=1, mode="constant", cval=0
            )

        v[..., i] = scipy.ndimage.correlate1d(
            v[..., i], ab[..., i], axis=1, mode="constant", cval=0
        )

    # Solve r for each pixel
    #r = np.linalg.solve(G, v)
    v = v[..., :, np.newaxis]       # now shape = (H, W, 6, 1)
    r = np.linalg.solve(G, v)       # result shape = (H, W, 6, 1)
    r = r[..., 0]                   # squeeze K=1 dim → (H, W, 6)

    # Quadratic term
    A = np.empty(list(f.shape) + [2, 2])
    A[..., 0, 0] = r[..., 3]
    A[..., 0, 1] = r[..., 5] / 2
    A[..., 1, 0] = A[..., 0, 1]
    A[..., 1, 1] = r[..., 4]

    # Linear term
    B = np.empty(list(f.shape) + [2])
    B[..., 0] = r[..., 1]
    B[..., 1] = r[..., 2]

    # constant term
    C = r[..., 0]

    # b: [n, n, 6]
    # r: [f, f, 6]
    # f: [f, f]
    # e = b*r - f

    return A, B, C

def farneback_polyexp_numpy(src: np.ndarray, n: int, sigma: float):
    """
    Pure-NumPy version of OpenCV’s FarnebackPolyExp(src, dst, n, sigma).
    src: 2D float32 image
    n: kernel half-width
    sigma: Gaussian stddev
    Returns:
      A: (H, W, 2,2) float32
      B: (H, W, 2)    float32
      C: (H, W)       float32
    """
    H, W = src.shape
    # 1) build 1-D kernels
    k = 2*n + 1
    x = np.arange(-n, n+1).astype(np.float32)
    g   = np.exp(-(x**2)/(2*sigma*sigma))
    g  /= g.sum()
    xg  = x * g
    xxg = (x*x) * g

    # 2) precompute the Farneback constants
    #    these integrals come from OpenCV’s FarnebackPrepareGaussian
    #    (you must use the exact same values your CUDA used)
    #    Here’s how OpenCV computes them in C++:
    ig11 =     np.dot(x*x*g, g)  
    ig03 =     np.dot(g, g)
    ig33 =     np.dot(xxg, g)
    ig55 =     np.dot(xxg, xxg)

    # 3) vertical pass → a (H, W, 3) buffer
    #    buf[y,x,:] = [sum_k g[k]*src[y±k,x],
    #                  sum_k xg[k]*(src[y+k,x]-src[y-k,x]),
    #                  sum_k xxg[k]*src[y±k,x]]
    buf = np.zeros((H, W, 3), dtype=np.float32)
    # start with k=0 term
    buf[:,:,0] = src * g[n]
    # buf[:,:,1] and buf[:,:,2] are zero at k=0
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

    # 4) mirror the horizontal edges in buf so we can do horizontal pass
    #    we only need to fill n cols on each side
    row = np.zeros((H, W+2*n, 3), dtype=np.float32)
    row[:, n:n+W, :] = buf
    # mirror-out at x<0  and x>=W
    row[:, :n, :]   = buf[:, :n, :][:, ::-1, :]
    row[:, n+W:, :] = buf[:, -n:, :][:, ::-1, :]

    # 5) horizontal pass → build A,B,C
    A = np.zeros((H, W, 2,2), dtype=np.float32)
    B = np.zeros((H, W, 2   ), dtype=np.float32)
    C = np.zeros((H, W      ), dtype=np.float32)

    for x in range(W):
        # compute the six b-values for all rows at once using vectorized sums
        # b1,b2,b3,b4,b5,b6 each shape (H,)
        # start with k=0 terms
        g0 = g[n]
        b1 = row[:, x+n, 0]*g0      # constant · g0
        b2 = np.zeros(H, np.float32)  
        b3 = row[:, x+n, 1]*g0      # y·g0 term
        b4 = np.zeros(H, np.float32)
        b6 = np.zeros(H, np.float32)
        b5 = row[:, x+n, 2]*g0      # x^2·g0

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
        # now apply the Farneback closed-form
        # note OpenCV’s drow[x*5+*] layout:
        #   drow[*+0] = (b3*ig11)        → B.y
        #   drow[*+1] = (b2*ig11)        → B.x
        #   drow[*+2] = (b1*ig03 + b5*ig33) → A.yy
        #   drow[*+3] = (b1*ig03 + b4*ig33) → A.xx
        #   drow[*+4] = (b6*ig55)        → 2·A.xy
        B[:, x, 1] = b3 * ig11
        B[:, x, 0] = b2 * ig11

        A[:, x, 1,1] = b1*ig03 + b5*ig33
        A[:, x, 0,0] = b1*ig03 + b4*ig33
        A[:, x,0,1] = (b6*ig55)*0.5
        A[:, x,1,0] = A[:, x,0,1]

        C[:, x] = b1 * g[n]  # equivalently buf[:,x+n,0]

    return A, B, C


def get_raw_moments_torch(frame: np.ndarray, polyN=5, sigma=1.0, device='cpu'):
    # 1) make float32 tensor [1,1,H,W]
    src = torch.from_numpy(frame.astype(np.float32))[None,None].to(device)
    # 2) build 1D kernels
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

    # 3) zero‐pad exactly polyN on all sides
    pad = (polyN, polyN, polyN, polyN)  # (left,right,top,bottom)
    src_p = F.pad(src, pad, mode='constant', value=0)

    # helper to do separable conv
    def sep(src, kv, kh):
        tmp = F.conv2d(src, kv, padding=(polyN,0))
        return F.conv2d(tmp, kh, padding=(0,polyN))

    # 4) compute the six raw moments
    C   = sep(src_p, kv_g,   kh_g  )
    Ix  = sep(src_p, kv_xg,  kh_g  )
    Iy  = sep(src_p, kv_g,   kh_xg )
    Ixx = sep(src_p, kv_xxg, kh_g  )
    Iyy = sep(src_p, kv_g,   kh_xxg)
    Ixy = sep(src_p, kv_xg,  kh_xg )

    # 5) remove the padding so shape is [1,1,H,W]
    C,  Ix,  Iy,  Ixx,  Iyy,  Ixy = [
        M[..., polyN:-polyN, polyN:-polyN] for M in (C, Ix, Iy, Ixx, Iyy, Ixy)
    ]

    # squeeze to 2D arrays
    return [M.squeeze().cpu().numpy() for M in (C, Ix, Iy, Ixx, Iyy, Ixy)]



def get_raw_moments_py(frame, polyN=5, sigma=1.0):
    """
    Compute the 6 raw Farnebäck moments (C, Ix, Iy, Ixx, Iyy, Ixy)
    using OpenCV separable filters, with zero‐padding.
    """
    # 0) ensure float32
    src = frame.astype(np.float32)

    # build 1-D kernels
    x = np.arange(-polyN, polyN+1, dtype=np.float32)
    g   = np.exp(-(x**2)/(2*sigma*sigma))
    g  /= g.sum()
    xg  = x * g
    xxg = (x*x)*g

    # apply separable filters with constant (zero) border
    C   = cv2.sepFilter2D(src, cv2.CV_32F, g,  g,  borderType=cv2.BORDER_CONSTANT)
    Ix  = cv2.sepFilter2D(src, cv2.CV_32F, xg, g,  borderType=cv2.BORDER_CONSTANT)
    Iy  = cv2.sepFilter2D(src, cv2.CV_32F, g,  xg, borderType=cv2.BORDER_CONSTANT)
    Ixx = cv2.sepFilter2D(src, cv2.CV_32F, xxg,g,  borderType=cv2.BORDER_CONSTANT)
    Iyy = cv2.sepFilter2D(src, cv2.CV_32F, g,  xxg,borderType=cv2.BORDER_CONSTANT)
    Ixy = cv2.sepFilter2D(src, cv2.CV_32F, xg, xg, borderType=cv2.BORDER_CONSTANT)

    # crop off the polyN-pixel border to match your CUDA output
    return [M[polyN:-polyN, polyN:-polyN] for M in (C,Ix,Iy,Ixx,Iyy,Ixy)]

# Load the shared library
lib = ctypes.cdll.LoadLibrary("./PolyExp.so")

# Setup function prototype
lib.polynomialExpansion.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input frame
    ctypes.c_float,                  # sigma
    ctypes.c_int,                    # width
    ctypes.c_int,                    # height
    ctypes.c_int,                    # polyN
    ctypes.POINTER(ctypes.c_float),  # outC
    ctypes.POINTER(ctypes.c_float),  # outB
    ctypes.POINTER(ctypes.c_float),  # outA
    ctypes.POINTER(ctypes.c_float),  # outdC
    ctypes.POINTER(ctypes.c_float),  # outIx
    ctypes.POINTER(ctypes.c_float),  # outIy
    ctypes.POINTER(ctypes.c_float),  # outIxx
    ctypes.POINTER(ctypes.c_float),  # outIyy
    ctypes.POINTER(ctypes.c_float),  # outIxy
]
def process_frame_with_cuda(frame, polyN=5):
    h, w = frame.shape
    total = h * w
    pad = polyN
    # mode='edge' replicates the edge pixels
    padded = np.pad(frame, ((pad,pad),(pad,pad)), mode='constant', constant_values= 0)

    # 2) feed the padded image into your CUDA routine
    #    (your wrapper will see width = w + 2*pad, height = h + 2*pad)
    h_pad, w_pad = padded.shape
    total_pad = h_pad * w_pad
    frame_in = np.ascontiguousarray(padded, dtype=np.float32)
    dC   = np.empty((h, w), dtype=np.float32)
    Ix  = np.empty_like(dC)
    Iy  = np.empty_like(dC)
    Ixx = np.empty_like(dC)
    Iyy = np.empty_like(dC)
    Ixy = np.empty_like(dC)
    #frame_in = np.ascontiguousarray(frame, dtype=np.float32)
    sigma = 1.0

    # Allocate output arrays
    outC = np.zeros(total, dtype=np.float32)       # c scalar per pixel
    outB = np.zeros(total * 2, dtype=np.float32)   # B vector (2 floats) per pixel
    outA = np.zeros(total * 4, dtype=np.float32)   # A matrix (4 floats) per pixel

    lib.polynomialExpansion(
        frame_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(sigma), 
        ctypes.c_int(w_pad),
        ctypes.c_int(h_pad),
        ctypes.c_int(polyN),
        outC.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        outB.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        outA.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dC.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Ix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Iy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Ixx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Iyy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Ixy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    outC = outC.reshape((h, w))
    outB = outB.reshape((h, w, 2))
    outA = outA.reshape((h, w, 2, 2))

    return outA, outB, outC, dC, Ix, Iy, Ixx, Iyy, Ixy

def main(video):
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-9)
        pad =5 
        padded = np.pad(frame, ((pad,pad),(pad,pad)), mode='constant', constant_values= 0)
        print("starting processing")
        A_cuda, B_cuda, C_cuda, dC_cuda, Ix_cuda, Iy_cuda, Ixx_cuda, Iyy_cuda, Ixy_cuda = process_frame_with_cuda(frame)
        print("done processing")
        print("A matrix shape:", A_cuda.shape)
        print("B vector shape:", B_cuda.shape)
        print("C scalar shape:", C_cuda.shape)
        print("Get Python version:")
        c = np.ones_like(frame)
        A_py, B_py, C_py = farneback_polyexp_numpy(frame, n=5, sigma=1.0)
        A_py2, B_py2, C_py2 = poly_exp(frame, c, sigma = 1.0)
        #dC_py,  Ix_py,  Iy_py,  Ixx_py,  Iyy_py,  Ixy_py  = get_raw_moments_py(padded)
        dC_py,  Ix_py,  Iy_py,  Ixx_py,  Iyy_py,  Ixy_py  = get_raw_moments_torch(frame, polyN=5, sigma=1.0)

        def stats(diff):
            return {
                'mean_abs': np.mean(np.abs(diff)),
                'max_abs':  np.max(np.abs(diff)),
                'rmse':     np.sqrt(np.mean(diff**2))
            }

        err_A = stats(A_cuda - A_py2)
        err_B = stats(B_cuda - B_py2)
        err_C = stats(C_cuda - C_py2)
        names    = ['dC', 'Ix', 'Iy', 'Ixx', 'Iyy', 'Ixy']
        cuda_arr = [dC_cuda, Ix_cuda, Iy_cuda, Ixx_cuda, Iyy_cuda, Ixy_cuda]
        py_arr   = [dC_py,   Ix_py,   Iy_py,   Ixx_py,   Iyy_py,   Ixy_py]

        for name, c_arr, p_arr in zip(names, cuda_arr, py_arr):
            diff = c_arr - p_arr
            e = stats(diff)
            print(f"Error stats for {name}: mean={e['mean_abs']:.6f}, "
                f"max={e['max_abs']:.6f}, rmse={e['rmse']:.6f}")
        
        # err_A = stats(A_py2 - A_py)
        # err_B = stats(B_py2 - B_py)
        # err_C = stats(C_py2 - C_py)

        print("Error stats for A (2×2 matrix):", err_A)
        print("Error stats for B (2-vector):     ", err_B)
        print("Error stats for C (scalar):       ", err_C)
       
        breakpoint()

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main("walking.mp4")