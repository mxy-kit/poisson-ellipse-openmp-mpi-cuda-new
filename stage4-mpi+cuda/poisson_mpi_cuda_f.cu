#include <mpi.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <algorithm>

const double A1 = -1.0, B1 =  1.0;
const double A2 = -0.6, B2 =  0.6;
const double F_VAL = 1.0;

using Matrix = std::vector< std::vector<double> >;

/*
 * Simple CUDA error checking helper.
 */
inline void checkCuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << msg << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/*
 * Check if the point (x, y) lies inside the ellipse x^2 + 4 y^2 < 1.
 */
bool is_in_D(double x, double y) {
    return (x * x + 4.0 * y * y < 1.0);
}

/*
 * Compute the length of the intersection of a segment with the ellipse.
 *
 * If vertical == true:
 *   x = const_coord, y in [start_var, end_var]
 * Otherwise:
 *   y = const_coord, x in [start_var, end_var]
 *
 * This is used to compute the fractions of cell faces lying inside D.
 */
double calculate_segment_length_in_D(double const_coord, double start_var,
                                     double end_var, bool vertical)
{
    double lij;
    if (vertical) {
        double x0 = const_coord;
        if (std::abs(x0) >= 1.0) {
            lij = 0.0;
        } else {
            double val = (1.0 - x0 * x0) / 4.0;
            double root = std::sqrt(std::max<double>(0.0, val));
            double y_min = -root;
            double y_max =  root;
            lij = std::max<double>(0.0,
                     std::min(end_var, y_max) - std::max(start_var, y_min));
        }
    } else {
        double y0 = const_coord;
        if (std::abs(2.0 * y0) >= 1.0) {
            lij = 0.0;
        } else {
            double val = 1.0 - 4.0 * y0 * y0;
            double root = std::sqrt(std::max<double>(0.0, val));
            double x_min = -root;
            double x_max =  root;
            lij = std::max<double>(0.0,
                     std::min(end_var, x_max) - std::max(start_var, x_min));
        }
    }
    return lij;
}

/*
 * Choose a 2D process grid Px x Py such that Px * Py == size
 * and the grid is as "square" as possible.
 */
void choose_process_grid(int size, int &Px, int &Py) {
    Px = static_cast<int>(std::sqrt(static_cast<double>(size)));
    while (Px > 1 && size % Px != 0) --Px;
    Py = size / Px;
}

/*
 * 2D decomposition of interior indices:
 *   i = 1..M-1,  j = 1..N-1
 *
 * Px processes along x, Py processes along y.
 * For a given rank we compute its subdomain:
 *   i_start..i_end,  j_start..j_end
 * so that the numbers of points differ by at most 1.
 */
void decompose_2d(int M, int N,
                  int Px, int Py,
                  int rank,
                  int &i_start, int &i_end,
                  int &j_start, int &j_end)
{
    int px = rank % Px;    // coordinate in x-direction
    int py = rank / Px;    // coordinate in y-direction

    int total_i = M - 1;   // number of interior nodes in x
    int base_i  = total_i / Px;
    int rem_i   = total_i % Px;

    int total_j = N - 1;   // number of interior nodes in y
    int base_j  = total_j / Py;
    int rem_j   = total_j % Py;

    // Decomposition in x-direction
    int offset_i = 1;
    for (int k = 0; k < px; ++k) {
        int nk = base_i + (k < rem_i ? 1 : 0);
        offset_i += nk;
    }
    int local_i = base_i + (px < rem_i ? 1 : 0);
    i_start = offset_i;
    i_end   = offset_i + local_i - 1;

    // Decomposition in y-direction
    int offset_j = 1;
    for (int k = 0; k < py; ++k) {
        int nk = base_j + (k < rem_j ? 1 : 0);
        offset_j += nk;
    }
    int local_j = base_j + (py < rem_j ? 1 : 0);
    j_start = offset_j;
    j_end   = offset_j + local_j - 1;
}

/*
 * Assemble coefficients a_ij, b_ij and right-hand side B_ij
 * on the local subdomain:
 *
 *   global i in [i_start..i_end], global j in [j_start..j_end].
 *
 * Local indexing:
 *   li = 0..local_nx+1,  lj = 0..local_ny+1
 * li,lj = 1..local_nx/local_ny are interior nodes of this subdomain,
 * li,lj = 0 or local_nx+1/local_ny+1 are halo layers.
 */
void fictitious_regions_setup_local(
    Matrix &a, Matrix &b, Matrix &B,
    int M, int N, double h1, double h2, double eps,
    int i_start, int i_end,
    int j_start, int j_end,
    int local_nx, int local_ny)
{
    // 1) Coefficients a_ij, b_ij on vertical/horizontal edges
    for (int gi = i_start - 1; gi <= i_end + 1; ++gi) {
        for (int gj = j_start - 1; gj <= j_end + 1; ++gj) {
            int li = gi - (i_start - 1); // 0..local_nx+1
            int lj = gj - (j_start - 1); // 0..local_ny+1

            double x_i = A1 + gi * h1;
            double y_j = A2 + gj * h2;

            double lij_a = calculate_segment_length_in_D(
                x_i - 0.5 * h1, y_j - 0.5 * h2, y_j + 0.5 * h2, true);
            double lij_b = calculate_segment_length_in_D(
                y_j - 0.5 * h2, x_i - 0.5 * h1, x_i + 0.5 * h1, false);

            a[li][lj] = (std::abs(lij_a - h2) < 1e-9)
                        ? 1.0
                        : (lij_a < 1e-9
                           ? 1.0 / eps
                           : (lij_a / h2) + (1.0 - lij_a / h2) / eps);

            b[li][lj] = (std::abs(lij_b - h1) < 1e-9)
                        ? 1.0
                        : (lij_b < 1e-9
                           ? 1.0 / eps
                           : (lij_b / h1) + (1.0 - lij_b / h1) / eps);
        }
    }

    // 2) Right-hand side B_ij on interior nodes of this subdomain
    for (int gi = i_start; gi <= i_end; ++gi) {
        for (int gj = j_start; gj <= j_end; ++gj) {
            int li = gi - (i_start - 1); // 1..local_nx
            int lj = gj - (j_start - 1); // 1..local_ny

            double x_i = A1 + gi * h1;
            double y_j = A2 + gj * h2;
            B[li][lj] = is_in_D(x_i, y_j) ? F_VAL : 0.0;
        }
    }
}

/*
 * Local dot product on CPU（现在只保留备用，不再在主循环里用）
 *   (u, v)_E = sum_{li=1..local_nx, lj=1..local_ny} u_ij v_ij h1 h2
 */
double dot_local(const Matrix &u, const Matrix &v,
                 int local_nx, int local_ny,
                 double h1, double h2)
{
    double sum = 0.0;
    for (int li = 1; li <= local_nx; ++li)
        for (int lj = 1; lj <= local_ny; ++lj)
            sum += u[li][lj] * v[li][lj];

    return sum * h1 * h2;
}

/*
 * CPU diagonal preconditioner (保留做参考，目前主流程用 GPU 版本)
 */
void apply_Dinv_local(const Matrix &r,
                      const Matrix &a, const Matrix &b,
                      Matrix &z,
                      int local_nx, int local_ny,
                      double h1, double h2)
{
    for (int li = 1; li <= local_nx; ++li) {
        for (int lj = 1; lj <= local_ny; ++lj) {
            double D_ij = (a[li + 1][lj] + a[li][lj]) / (h1 * h1)
                        + (b[li][lj + 1] + b[li][lj]) / (h2 * h2);
            z[li][lj] = (D_ij != 0.0) ? r[li][lj] / D_ij : 0.0;
        }
    }
}

/*
 * Exchange halo layers of p with all four neighbours (left, right, down, up).
 * On global boundaries Dirichlet condition u = 0 is enforced.
 *
 * The process grid is Px x Py. Rank is mapped to (px, py) by:
 *   px = rank % Px,  py = rank / Px
 */
void exchange_halos_2d(Matrix &p,
                       int local_nx, int local_ny,
                       int Px, int Py,
                       int rank, MPI_Comm comm)
{
    int px = rank % Px;
    int py = rank / Px;

    int left_rank  = (px > 0)      ? (rank - 1)   : MPI_PROC_NULL;
    int right_rank = (px < Px - 1) ? (rank + 1)   : MPI_PROC_NULL;
    int down_rank  = (py > 0)      ? (rank - Px)  : MPI_PROC_NULL;
    int up_rank    = (py < Py - 1) ? (rank + Px)  : MPI_PROC_NULL;

    std::vector<double> send_left (local_ny + 2), recv_left (local_ny + 2);
    std::vector<double> send_right(local_ny + 2), recv_right(local_ny + 2);
    std::vector<double> send_down(local_nx + 2), recv_down(local_nx + 2);
    std::vector<double> send_up  (local_nx + 2), recv_up  (local_nx + 2);

    // Pack columns to send buffers (including halo rows)
    if (left_rank != MPI_PROC_NULL) {
        for (int lj = 0; lj <= local_ny + 1; ++lj)
            send_left[lj] = p[1][lj];              // first interior column
    }
    if (right_rank != MPI_PROC_NULL) {
        for (int lj = 0; lj <= local_ny + 1; ++lj)
            send_right[lj] = p[local_nx][lj];      // last interior column
    }

    // Pack rows to send buffers (including halo columns)
    if (down_rank != MPI_PROC_NULL) {
        for (int li = 0; li <= local_nx + 1; ++li)
            send_down[li] = p[li][1];              // first interior row
    }
    if (up_rank != MPI_PROC_NULL) {
        for (int li = 0; li <= local_nx + 1; ++li)
            send_up[li] = p[li][local_ny];         // last interior row
    }

    // Left-right exchange with blocking Sendrecv
    if (left_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(send_left.data(),  local_ny + 2, MPI_DOUBLE, left_rank, 0,
                     recv_left.data(),  local_ny + 2, MPI_DOUBLE, left_rank, 1,
                     comm, MPI_STATUS_IGNORE);
    } else {
        // Global left boundary: Dirichlet u = 0
        for (int lj = 0; lj <= local_ny + 1; ++lj) p[0][lj] = 0.0;
    }

    if (right_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(send_right.data(), local_ny + 2, MPI_DOUBLE, right_rank, 1,
                     recv_right.data(), local_ny + 2, MPI_DOUBLE, right_rank, 0,
                     comm, MPI_STATUS_IGNORE);
    } else {
        // Global right boundary: Dirichlet u = 0
        for (int lj = 0; lj <= local_ny + 1; ++lj) p[local_nx + 1][lj] = 0.0;
    }

    // Down-up exchange with blocking Sendrecv
    if (down_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(send_down.data(), local_nx + 2, MPI_DOUBLE, down_rank, 2,
                     recv_down.data(), local_nx + 2, MPI_DOUBLE, down_rank, 3,
                     comm, MPI_STATUS_IGNORE);
    } else {
        // Global bottom boundary: Dirichlet u = 0
        for (int li = 0; li <= local_nx + 1; ++li) p[li][0] = 0.0;
    }

    if (up_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(send_up.data(),   local_nx + 2, MPI_DOUBLE, up_rank, 3,
                     recv_up.data(),   local_nx + 2, MPI_DOUBLE, up_rank, 2,
                     comm, MPI_STATUS_IGNORE);
    } else {
        // Global top boundary: Dirichlet u = 0
        for (int li = 0; li <= local_nx + 1; ++li) p[li][local_ny + 1] = 0.0;
    }

    // Unpack received data into halo layers
    if (left_rank != MPI_PROC_NULL) {
        for (int lj = 0; lj <= local_ny + 1; ++lj)
            p[0][lj] = recv_left[lj];
    }
    if (right_rank != MPI_PROC_NULL) {
        for (int lj = 0; lj <= local_ny + 1; ++lj)
            p[local_nx + 1][lj] = recv_right[lj];
    }
    if (down_rank != MPI_PROC_NULL) {
        for (int li = 0; li <= local_nx + 1; ++li)
            p[li][0] = recv_down[li];
    }
    if (up_rank != MPI_PROC_NULL) {
        for (int li = 0; li <= local_nx + 1; ++li)
            p[li][local_ny + 1] = recv_up[li];
    }
}

/*
 * CUDA kernel: apply discrete operator A to p.
 * Input arrays are flattened with pitch = local_ny + 2.
 * We compute interior nodes li=1..local_nx, lj=1..local_ny.
 */
__global__
void apply_A_kernel(const double* p,
                    const double* a, const double* b,
                    double* Ap,
                    int local_nx, int local_ny,
                    double h1, double h2)
{
    int li = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int lj = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (li > local_nx || lj > local_ny) return;

    int pitch = local_ny + 2;
    int idx    = li      * pitch + lj;
    int idx_ip = (li + 1) * pitch + lj;
    int idx_im = (li - 1) * pitch + lj;
    int idx_jp = li       * pitch + (lj + 1);
    int idx_jm = li       * pitch + (lj - 1);

    double Ax = -1.0 / h1 * (
        a[idx_ip] * (p[idx_ip] - p[idx]) / h1
      - a[idx]    * (p[idx]    - p[idx_im]) / h1
    );
    double Ay = -1.0 / h2 * (
        b[idx_jp] * (p[idx_jp] - p[idx]) / h2
      - b[idx]    * (p[idx]    - p[idx_jm]) / h2
    );

    Ap[idx] = Ax + Ay;
}

/*
 * CUDA kernel: apply diagonal preconditioner z = D^{-1} r.
 */
__global__
void apply_Dinv_kernel(const double* r,
                       const double* a, const double* b,
                       double* z,
                       int local_nx, int local_ny,
                       double h1, double h2)
{
    int li = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int lj = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (li > local_nx || lj > local_ny) return;

    int pitch = local_ny + 2;
    int idx    = li * pitch + lj;
    int idx_ip = (li + 1) * pitch + lj;
    int idx_jp = li       * pitch + (lj + 1);

    double D_ij = (a[idx_ip] + a[idx]) / (h1 * h1)
                + (b[idx_jp] + b[idx]) / (h2 * h2);

    z[idx] = (D_ij != 0.0) ? (r[idx] / D_ij) : 0.0;
}

/*
 * CUDA kernel for dot product over interior nodes.
 *
 * - x, y: device arrays with halo, pitch = local_ny + 2
 * - partial: device array of size num_threads (gridDim.x * blockDim.x)
 *
 * 每个线程遍历若干个 interior 结点（stride 循环），
 * 把自己的部分和写入 partial[tid]。
 * 不用 shared memory，也不用 atomicAdd。
 */
__global__
void dot_kernel(const double* x,
                const double* y,
                double* partial,
                int local_nx, int local_ny,
                double h1h2)
{
    int pitch = local_ny + 2;
    int interior_size = local_nx * local_ny;

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;

    for (int k = tid; k < interior_size; k += stride) {
        int li = k / local_ny + 1;   // 1..local_nx
        int lj = k % local_ny + 1;   // 1..local_ny
        int idx = li * pitch + lj;
        sum += x[idx] * y[idx];
    }

    // 带上 h1*h2（积分权重）
    partial[tid] = sum * h1h2;
}
__global__
void zero_kernel(double* w, int local_nx, int local_ny)
{
    int li = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int lj = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (li > local_nx || lj > local_ny) return;

    int pitch = local_ny + 2;
    int idx   = li * pitch + lj;
    w[idx] = 0.0;
}

// p = z
__global__
void copy_kernel(const double* z, double* p,
                 int local_nx, int local_ny)
{
    int li = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int lj = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (li > local_nx || lj > local_ny) return;

    int pitch = local_ny + 2;
    int idx   = li * pitch + lj;
    p[idx] = z[idx];
}

// update w, r and accumulate ||w_new - w_old||^2
__global__
void update_w_r_kernel(double* w,
                       double* r,
                       const double* p,
                       const double* Ap,
                       double alpha,
                       int local_nx, int local_ny,
                       double h1h2,
                       double* partial_diff)
{
    int pitch = local_ny + 2;
    int interior_size = local_nx * local_ny;

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;

    for (int k = tid; k < interior_size; k += stride) {
        int li  = k / local_ny + 1;
        int lj  = k % local_ny + 1;
        int idx = li * pitch + lj;

        double w_old = w[idx];
        double w_new = w_old + alpha * p[idx];
        double r_new = r[idx] - alpha * Ap[idx];

        w[idx] = w_new;
        r[idx] = r_new;

        double diff = w_new - w_old;
        sum += diff * diff;
    }
    partial_diff[tid] = sum * h1h2;
}

// p = z + beta * p
__global__
void update_p_kernel(double* p,
                     const double* z,
                     double beta,
                     int local_nx, int local_ny)
{
    int li = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int lj = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (li > local_nx || lj > local_ny) return;

    int pitch = local_ny + 2;
    int idx   = li * pitch + lj;
    p[idx] = z[idx] + beta * p[idx];
}


/*
 * Preconditioned Conjugate Gradient (PCG) solver with MPI + CUDA.
 *
 *  - 2D domain decomposition (Px x Py) over MPI processes.
 *  - Each process stores only its local subdomain with halo layers.
 *  - The matrix-vector product Ap = A p and preconditioner z = D^{-1} r
 *    are computed on the GPU.
 *  - dot products (z,r) 和 (Ap,p) 也在 GPU 上计算（用 dot_kernel）。
 */
int gradient_solver_mpi(int M, int N, double delta, int max_iter,
                        int rank, int size, MPI_Comm comm)
{
    double h1 = (B1 - A1) / M;
    double h2 = (B2 - A2) / N;
    double eps = std::max(h1, h2) * std::max(h1, h2);

    double h1h2 = h1 * h2;

    // 1) Build 2D process grid and compute local ranges
    int Px, Py;
    choose_process_grid(size, Px, Py);

    int i_start, i_end, j_start, j_end;
    decompose_2d(M, N, Px, Py, rank, i_start, i_end, j_start, j_end);

    int local_nx = i_end - i_start + 1;
    int local_ny = j_end - j_start + 1;

    // 2) Allocate local arrays: size (local_nx+2) x (local_ny+2)
    Matrix a(local_nx + 2, std::vector<double>(local_ny + 2, 0.0));
    Matrix b(local_nx + 2, std::vector<double>(local_ny + 2, 0.0));
    Matrix B(local_nx + 2, std::vector<double>(local_ny + 2, 0.0));

    fictitious_regions_setup_local(a, b, B,
                                   M, N, h1, h2, eps,
                                   i_start, i_end,
                                   j_start, j_end,
                                   local_nx, local_ny);

    Matrix w(local_nx + 2, std::vector<double>(local_ny + 2, 0.0));
    Matrix r = B;   // initial residual r = f - A w, w = 0 so r = f
    Matrix z(local_nx + 2, std::vector<double>(local_ny + 2, 0.0));
    Matrix p(local_nx + 2, std::vector<double>(local_ny + 2, 0.0));
    Matrix Ap(local_nx + 2, std::vector<double>(local_ny + 2, 0.0));

    // 3) GPU-related arrays (flattened)
    int pitch = local_ny + 2;
    int local_size = (local_nx + 2) * (local_ny + 2);

    std::vector<double> h_a(local_size);
    std::vector<double> h_b(local_size);
    std::vector<double> h_p(local_size);
    std::vector<double> h_Ap(local_size);
    std::vector<double> h_r(local_size);
    std::vector<double> h_z(local_size);

    // Flatten a and b once and copy to device
    for (int li = 0; li <= local_nx + 1; ++li) {
        for (int lj = 0; lj <= local_ny + 1; ++lj) {
            int idx = li * pitch + lj;
            h_a[idx] = a[li][lj];
            h_b[idx] = b[li][lj];
        }
    }

    double *d_a  = nullptr;
    double *d_b  = nullptr;
    double *d_p  = nullptr;
    double *d_Ap = nullptr;
    double *d_r  = nullptr;
    double *d_z  = nullptr;

    size_t bytes = static_cast<size_t>(local_size) * sizeof(double);

    checkCuda(cudaMalloc((void**)&d_a,  bytes), "cudaMalloc d_a");
    checkCuda(cudaMalloc((void**)&d_b,  bytes), "cudaMalloc d_b");
    checkCuda(cudaMalloc((void**)&d_p,  bytes), "cudaMalloc d_p");
    checkCuda(cudaMalloc((void**)&d_Ap, bytes), "cudaMalloc d_Ap");
    checkCuda(cudaMalloc((void**)&d_r,  bytes), "cudaMalloc d_r");
    checkCuda(cudaMalloc((void**)&d_z,  bytes), "cudaMalloc d_z");

    checkCuda(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy a H2D");
    checkCuda(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy b H2D");

    // === dot product on GPU: partial buffer ===
    const int DOT_BLOCKS  = 1;     // 可以调大一点
    const int DOT_THREADS = 256;   // 每个 block 256 线程
    const int DOT_PARTIAL = DOT_BLOCKS * DOT_THREADS;

    double *d_partial = nullptr;
    checkCuda(cudaMalloc((void**)&d_partial,
                         DOT_PARTIAL * sizeof(double)),
              "cudaMalloc d_partial");

    std::vector<double> h_partial(DOT_PARTIAL, 0.0);

    // helper lambda: dot on GPU, 返回 local dot (未 MPI_reduce)
    auto dot_gpu = [&](const double* dx, const double* dy) -> double
    {
        dim3 block(DOT_THREADS);
        dim3 grid(DOT_BLOCKS);

        dot_kernel<<<grid, block>>>(dx, dy, d_partial,
                                    local_nx, local_ny, h1h2);
        checkCuda(cudaGetLastError(), "dot_kernel launch");
        checkCuda(cudaDeviceSynchronize(), "dot_kernel sync");

        checkCuda(cudaMemcpy(h_partial.data(), d_partial,
                             DOT_PARTIAL * sizeof(double),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy partial D2H");

        double sum = 0.0;
        for (int i = 0; i < DOT_PARTIAL; ++i) {
            sum += h_partial[i];
        }
        return sum;
    };

    int iter = 0;

    // Timing accumulators for GPU and copies
    double time_gpu_total  = 0.0;
    double time_copy_total = 0.0;

    // Timing accumulators for communication and CPU-side work
    double time_comm_total     = 0.0; // MPI halo exchange
    double time_precond_total  = 0.0; // CPU part of preconditioner (matrix unpack)
    double time_dot_total      = 0.0; // dot on GPU + copy partial + sum

    // CUDA launch configuration (simple 2D grid)
    dim3 blockDim(16, 16);
    dim3 gridDim((local_nx + blockDim.x - 1) / blockDim.x,
                 (local_ny + blockDim.y - 1) / blockDim.y);

    // Lambda: apply D^{-1} r on GPU and measure time
    auto apply_Dinv_gpu = [&](const Matrix &r_mat, Matrix &z_mat)
    {
        // 1) flatten r -> h_r
        for (int li = 0; li <= local_nx + 1; ++li) {
            for (int lj = 0; lj <= local_ny + 1; ++lj) {
                int idx = li * pitch + lj;
                h_r[idx] = r_mat[li][lj];
            }
        }

        // 2) H2D copy for r
        double t_h2d0 = MPI_Wtime();
        checkCuda(cudaMemcpy(d_r, h_r.data(), bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy r H2D (precond)");
        double t_h2d1 = MPI_Wtime();
        time_copy_total += (t_h2d1 - t_h2d0);

        // 3) kernel: z = D^{-1} r
        double t_k0 = MPI_Wtime();
        apply_Dinv_kernel<<<gridDim, blockDim>>>(d_r, d_a, d_b, d_z,
                                                 local_nx, local_ny, h1, h2);
        checkCuda(cudaGetLastError(), "apply_Dinv_kernel launch");
        checkCuda(cudaDeviceSynchronize(), "apply_Dinv_kernel sync");
        double t_k1 = MPI_Wtime();
        time_gpu_total += (t_k1 - t_k0);

        // 4) D2H copy for z
        double t_d2h0 = MPI_Wtime();
        checkCuda(cudaMemcpy(h_z.data(), d_z, bytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy z D2H (precond)");
        double t_d2h1 = MPI_Wtime();
        time_copy_total += (t_d2h1 - t_d2h0);

        // 5) unpack interior part back to Matrix and count as precond CPU time
        double t_cpu0 = MPI_Wtime();
        for (int li = 1; li <= local_nx; ++li) {
            for (int lj = 1; lj <= local_ny; ++lj) {
                int idx = li * pitch + lj;
                z_mat[li][lj] = h_z[idx];
            }
        }
        double t_cpu1 = MPI_Wtime();
        time_precond_total += (t_cpu1 - t_cpu0);
    };

    // 4) Apply preconditioner to the initial residual (GPU)
    apply_Dinv_gpu(r, z);

    // 5) initial direction p = z
    p = z;

    // 6) initial (z, r) —— 改用 GPU dot
    double t_dot0 = MPI_Wtime();
    double local_zr = dot_gpu(d_z, d_r);
    double t_dot1 = MPI_Wtime();
    time_dot_total += (t_dot1 - t_dot0);

    double zr_old = 0.0;
    MPI_Allreduce(&local_zr, &zr_old, 1, MPI_DOUBLE, MPI_SUM, comm);

    for (int k = 1; k <= max_iter; ++k) {
        iter = k;

        // 1) Update halo layers of p (CPU + MPI)
        double t_comm0 = MPI_Wtime();
        exchange_halos_2d(p, local_nx, local_ny, Px, Py, rank, comm);
        double t_comm1 = MPI_Wtime();
        time_comm_total += (t_comm1 - t_comm0);

        // 2) Flatten p (including halos) into h_p
        for (int li = 0; li <= local_nx + 1; ++li) {
            for (int lj = 0; lj <= local_ny + 1; ++lj) {
                int idx = li * pitch + lj;
                h_p[idx] = p[li][lj];
            }
        }

        double t0 = MPI_Wtime();
        checkCuda(cudaMemcpy(d_p, h_p.data(), bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy p H2D");
        double t1 = MPI_Wtime();

        // 3) Ap = A p on GPU
        apply_A_kernel<<<gridDim, blockDim>>>(d_p, d_a, d_b, d_Ap,
                                              local_nx, local_ny, h1, h2);
        checkCuda(cudaGetLastError(), "apply_A_kernel launch");
        checkCuda(cudaDeviceSynchronize(), "apply_A_kernel sync");
        double t2 = MPI_Wtime();

        checkCuda(cudaMemcpy(h_Ap.data(), d_Ap, bytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy Ap D2H");
        double t3 = MPI_Wtime();

        time_copy_total += (t1 - t0) + (t3 - t2);
        time_gpu_total  += (t2 - t1);

        // 4) denom = (Ap, p) —— 改用 GPU dot（在 unflatten 之前算）
        t_dot0 = MPI_Wtime();
        double local_den = dot_gpu(d_Ap, d_p);
        t_dot1 = MPI_Wtime();
        time_dot_total += (t_dot1 - t_dot0);

        double denom = 0.0;
        MPI_Allreduce(&local_den, &denom, 1, MPI_DOUBLE, MPI_SUM, comm);
        if (std::abs(denom) < 1e-15) break;

        double alpha = zr_old / denom;

        // 5) Unflatten interior part of Ap back to Matrix（用于更新 r）
        for (int li = 1; li <= local_nx; ++li) {
            for (int lj = 1; lj <= local_ny; ++lj) {
                int idx = li * pitch + lj;
                Ap[li][lj] = h_Ap[idx];
            }
        }

        // 6) Update w and r, and accumulate local norm of w(k+1) - w(k)
        double local_diff = 0.0;
        for (int li = 1; li <= local_nx; ++li) {
            for (int lj = 1; lj <= local_ny; ++lj) {
                double w_old = w[li][lj];
                w[li][lj] = w_old + alpha * p[li][lj];
                r[li][lj] -= alpha * Ap[li][lj];
                double diff = w[li][lj] - w_old;
                local_diff += diff * diff;
            }
        }

        // 7) z = D^{-1} r (GPU)
        apply_Dinv_gpu(r, z);

        // 8) zr_new = (z, r) —— 继续用 GPU dot
        t_dot0 = MPI_Wtime();
        double local_zr_new = dot_gpu(d_z, d_r);
        t_dot1 = MPI_Wtime();
        time_dot_total += (t_dot1 - t_dot0);

        double zr_new = 0.0;
        MPI_Allreduce(&local_zr_new, &zr_new, 1, MPI_DOUBLE, MPI_SUM, comm);

        // 9) Global convergence check: ||w(k+1) - w(k)||_E < delta
        double global_diff = 0.0;
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, comm);
        global_diff = std::sqrt(global_diff * h1h2);

        if (global_diff < delta) {
            if (rank == 0) {
                std::cout << "Converged after " << k
                          << " iterations (||w(k+1)-w(k)|| < " << delta << ").\n";
            }
            break;
        }

        double beta = zr_new / zr_old;
        zr_old = zr_new;

        // 10) p = z + beta * p
        for (int li = 1; li <= local_nx; ++li)
            for (int lj = 1; lj <= local_ny; ++lj)
                p[li][lj] = z[li][lj] + beta * p[li][lj];
    }

    // Free device memory
    checkCuda(cudaFree(d_a),  "cudaFree d_a");
    checkCuda(cudaFree(d_b),  "cudaFree d_b");
    checkCuda(cudaFree(d_p),  "cudaFree d_p");
    checkCuda(cudaFree(d_Ap), "cudaFree d_Ap");
    checkCuda(cudaFree(d_r),  "cudaFree d_r");
    checkCuda(cudaFree(d_z),  "cudaFree d_z");
    checkCuda(cudaFree(d_partial), "cudaFree d_partial");

    // Reduce timings across all ranks (take max as representative)
    double gpu_total_max     = 0.0;
    double copy_total_max    = 0.0;
    double comm_total_max    = 0.0;
    double precond_total_max = 0.0;
    double dot_total_max     = 0.0;

    MPI_Reduce(&time_gpu_total,     &gpu_total_max,     1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&time_copy_total,    &copy_total_max,    1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&time_comm_total,    &comm_total_max,    1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&time_precond_total, &precond_total_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&time_dot_total,     &dot_total_max,     1, MPI_DOUBLE, MPI_MAX, 0, comm);

    // Print timing info on rank 0
    if (rank == 0) {
        std::cout << "   GPU compute time (Ap + D^{-1}r, max over ranks) ~ "
                  << gpu_total_max  << " s\n";
        std::cout << "   Host<->Device copy time (max over ranks)        ~ "
                  << copy_total_max << " s\n";
        std::cout << "   MPI halo exchange time (max over ranks)         ~ "
                  << comm_total_max << " s\n";
        std::cout << "   Preconditioner CPU part time (max over ranks)   ~ "
                  << precond_total_max << " s\n";
        std::cout << "   Dot products time (max over ranks)              ~ "
                  << dot_total_max << " s\n";
    }

    return iter;
}

// ---------------- MAIN ----------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Program start time (after MPI_Init)
    double t_program_start = MPI_Wtime();

    int M = 40, N = 40;
    if (argc >= 3) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
    }
    if (rank == 0) {
        std::cout << "MPI + CUDA 2D run with " << size << " processes; "
                  << "M=" << M << ", N=" << N << std::endl;
    }

    double delta = 1e-6;
    int max_iter = (M - 1) * (N - 1);

    // Initialization phase ends before solver starts
    MPI_Barrier(MPI_COMM_WORLD);
    double t_before_solver = MPI_Wtime();

    int iterations = gradient_solver_mpi(M, N, delta, max_iter,
                                         rank, size, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_after_solver = MPI_Wtime();

    // Finalization phase: everything after solver, before MPI_Finalize
    double t_before_finalize = t_after_solver; // 当前 solver 后几乎无额外工作

    double time_init     = t_before_solver   - t_program_start;
    double time_solver   = t_after_solver    - t_before_solver;
    double time_finalize = t_before_finalize - t_after_solver;
    double elapsed       = t_before_finalize - t_program_start;

    if (rank == 0) {
        std::cout << "M=" << M << ", N=" << N
                  << " | Iter=" << iterations
                  << " | Total Time=" << std::fixed << std::setprecision(6)
                  << elapsed << " s\n";
        std::cout << "   Init time (program)      ~ " << time_init     << " s\n";
        std::cout << "   Solver time (MPI+CUDA)   ~ " << time_solver   << " s\n";
        std::cout << "   Finalization time        ~ " << time_finalize << " s\n";
    }

    MPI_Finalize();
    return 0;
}

