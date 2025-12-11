#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <algorithm>

const double A1 = -1.0, B1 = 1.0;
const double A2 = -0.6, B2 = 0.6;
const double F_VAL = 1.0;

using Matrix = std::vector<std::vector<double>>;

/*
 * Check if the point (x, y) lies inside the ellipse x^2 + 4 y^2 < 1.
 */
bool if_is_in_D(double x, double y) {
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
double cal_seg_len_in_D(double const_coord, double start_var,
                                     double end_var, bool is_ver)
{
    double lij;
    if (is_ver) {
        double x0 = const_coord;
        if (std::abs(x0) >= 1.0) lij = 0.0;
        else {
            double y_min = -std::sqrt(std::max(0.0, (1.0 - x0 * x0) / 4.0));
            double y_max =  std::sqrt(std::max(0.0, (1.0 - x0 * x0) / 4.0));
            lij = std::max(0.0, std::min(end_var, y_max) - std::max(start_var, y_min));
        }
    } else {
        double y0 = const_coord;
        if (std::abs(2.0 * y0) >= 1.0) lij = 0.0;
        else {
            double x_min = -std::sqrt(std::max(0.0, 1.0 - 4.0 * y0 * y0));
            double x_max =  std::sqrt(std::max(0.0, 1.0 - 4.0 * y0 * y0));
            lij = std::max(0.0, std::min(end_var, x_max) - std::max(start_var, x_min));
        }
    }
    return lij;
}

/*
 * Choose a 2D process grid Px x Py such that Px * Py == size
 * and the grid is as "square" as possible.
 */
void choose_process_grid(int size, int &Px, int &Py) {
    Px = static_cast<int>(std::sqrt(size));
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
void fic_reg_local(
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

            double lij_a = cal_seg_len_in_D(
                x_i - 0.5 * h1, y_j - 0.5 * h2, y_j + 0.5 * h2, true);
            double lij_b = cal_seg_len_in_D(
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
            B[li][lj] =if_is_in_D(x_i, y_j) ? F_VAL : 0.0;
        }
    }
}

/*
 * Local dot product:
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
 * Apply the discrete operator A to w locally:
 *   Aw = A w
 *
 * Requires up-to-date halo layers of w.
 */
void mat_A_local(const Matrix &w,
                   const Matrix &a, const Matrix &b,
                   Matrix &Aw,
                   int local_nx, int local_ny,
                   double h1, double h2)
{
    for (int li = 1; li <= local_nx; ++li) {
        for (int lj = 1; lj <= local_ny; ++lj) {
            double Ax = -1.0 / h1 * (
                a[li + 1][lj] * (w[li + 1][lj] - w[li][lj]) / h1
              - a[li][lj]     * (w[li][lj]     - w[li - 1][lj]) / h1
            );
            double Ay = -1.0 / h2 * (
                b[li][lj + 1] * (w[li][lj + 1] - w[li][lj]) / h2
              - b[li][lj]     * (w[li][lj]     - w[li][lj - 1]) / h2
            );
            Aw[li][lj] = Ax + Ay;
        }
    }
}

/*
 * Apply the diagonal preconditioner D^{-1} locally:
 *   z = D^{-1} r
 */
void mat_D(const Matrix &r,
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

    int left_rank  = (px > 0)        ? (rank - 1)   : MPI_PROC_NULL;
    int right_rank = (px < Px - 1)   ? (rank + 1)   : MPI_PROC_NULL;
    int down_rank  = (py > 0)        ? (rank - Px)  : MPI_PROC_NULL;
    int up_rank    = (py < Py - 1)   ? (rank + Px)  : MPI_PROC_NULL;

    std::vector<double> send_left (local_ny + 2), recv_left (local_ny + 2);
    std::vector<double> send_right(local_ny + 2), recv_right(local_ny + 2);
    std::vector<double> send_down(local_nx + 2), recv_down(local_nx + 2);
    std::vector<double> send_up  (local_nx + 2), recv_up  (local_nx + 2);

    MPI_Request reqs[8];
    int q = 0;

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

    // Asynchronous exchange with left neighbour
    if (left_rank != MPI_PROC_NULL) {
        MPI_Irecv(recv_left.data(), local_ny + 2, MPI_DOUBLE,
                  left_rank, 1, comm, &reqs[q++]);
        MPI_Isend(send_left.data(), local_ny + 2, MPI_DOUBLE,
                  left_rank, 0, comm, &reqs[q++]);
    } else {
        // Global left boundary: Dirichlet u = 0
        for (int lj = 0; lj <= local_ny + 1; ++lj) p[0][lj] = 0.0;
    }

    // Asynchronous exchange with right neighbour
    if (right_rank != MPI_PROC_NULL) {
        MPI_Irecv(recv_right.data(), local_ny + 2, MPI_DOUBLE,
                  right_rank, 0, comm, &reqs[q++]);
        MPI_Isend(send_right.data(), local_ny + 2, MPI_DOUBLE,
                  right_rank, 1, comm, &reqs[q++]);
    } else {
        // Global right boundary: Dirichlet u = 0
        for (int lj = 0; lj <= local_ny + 1; ++lj) p[local_nx + 1][lj] = 0.0;
    }

    // Asynchronous exchange with bottom neighbour
    if (down_rank != MPI_PROC_NULL) {
        MPI_Irecv(recv_down.data(), local_nx + 2, MPI_DOUBLE,
                  down_rank, 3, comm, &reqs[q++]);
        MPI_Isend(send_down.data(), local_nx + 2, MPI_DOUBLE,
                  down_rank, 2, comm, &reqs[q++]);
    } else {
        // Global bottom boundary: Dirichlet u = 0
        for (int li = 0; li <= local_nx + 1; ++li) p[li][0] = 0.0;
    }

    // Asynchronous exchange with top neighbour
    if (up_rank != MPI_PROC_NULL) {
        MPI_Irecv(recv_up.data(), local_nx + 2, MPI_DOUBLE,
                  up_rank, 2, comm, &reqs[q++]);
        MPI_Isend(send_up.data(), local_nx + 2, MPI_DOUBLE,
                  up_rank, 3, comm, &reqs[q++]);
    } else {
        // Global top boundary: Dirichlet u = 0
        for (int li = 0; li <= local_nx + 1; ++li) p[li][local_ny + 1] = 0.0;
    }

    if (q > 0) {
        MPI_Waitall(q, reqs, MPI_STATUSES_IGNORE);
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
 * Preconditioned Conjugate Gradient (PCG) solver.
 *
 *  - 2D domain decomposition (Px x Py) over MPI processes.
 *  - Each process stores only its local subdomain with halo layers.
 *  - No OpenMP here: all parallelism is through MPI.
 */
int solve_mpi(int M, int N, double delta, int max_iter,
                        int rank, int size, MPI_Comm comm)
{
    double h1 = (B1 - A1) / M;
    double h2 = (B2 - A2) / N;
    double eps = std::max(h1, h2) * std::max(h1, h2);

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

    fic_reg_local(a, b, B,
                                   M, N, h1, h2, eps,
                                   i_start, i_end,
                                   j_start, j_end,
                                   local_nx, local_ny);

    Matrix w(local_nx + 2, std::vector<double>(local_ny + 2, 0.0));
    Matrix r = B;   // initial residual r = f - A w, w = 0 so r = f
    Matrix z(local_nx + 2, std::vector<double>(local_ny + 2, 0.0));
    Matrix p(local_nx + 2, std::vector<double>(local_ny + 2, 0.0));
    Matrix Ap(local_nx + 2, std::vector<double>(local_ny + 2, 0.0));

    // 3) Apply preconditioner to the initial residual
    mat_D(r, a, b, z, local_nx, local_ny, h1, h2);
    p = z;

    double local_zr = dot_local(z, r, local_nx, local_ny, h1, h2);
    double zr_old = 0.0;
    MPI_Allreduce(&local_zr, &zr_old, 1, MPI_DOUBLE, MPI_SUM, comm);

    int iter = 0;

    for (int k = 1; k <= max_iter; ++k) {
        iter = k;

        // 1) Exchange halo layers of p with neighbouring processes
        exchange_halos_2d(p, local_nx, local_ny, Px, Py, rank, comm);

        // 2) Ap = A p
        mat_A_local(p, a, b, Ap, local_nx, local_ny, h1, h2);

        // 3) denom = (Ap, p)
        double local_den = dot_local(Ap, p, local_nx, local_ny, h1, h2);
        double denom = 0.0;
        MPI_Allreduce(&local_den, &denom, 1, MPI_DOUBLE, MPI_SUM, comm);
        if (std::abs(denom) < 1e-15) break;

        double alpha = zr_old / denom;

        // 4) Update w and r, and accumulate local norm of w(k+1) - w(k)
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

        // 5) z = D^{-1} r
        mat_D(r, a, b, z, local_nx, local_ny, h1, h2);

        // 6) zr_new = (z, r)
        double local_zr_new = dot_local(z, r, local_nx, local_ny, h1, h2);
        double zr_new = 0.0;
        MPI_Allreduce(&local_zr_new, &zr_new, 1, MPI_DOUBLE, MPI_SUM, comm);

        // 7) Global convergence check: ||w(k+1) - w(k)||_E < delta
        double global_diff = 0.0;
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, comm);
        global_diff = std::sqrt(global_diff * h1 * h2);

        if (global_diff < delta) {
            if (rank == 0) {
                std::cout << "Converged after " << k
                          << " iterations (||w(k+1)-w(k)|| < " << delta << ").\n";
            }
            break;
        }

        double beta = zr_new / zr_old;
        zr_old = zr_new;

        // 8) p = z + beta * p
        for (int li = 1; li <= local_nx; ++li)
            for (int lj = 1; lj <= local_ny; ++lj)
                p[li][lj] = z[li][lj] + beta * p[li][lj];
    }

    return iter;
}

// ---------------- MAIN ----------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M = 40, N = 40;
    if (argc >= 3) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
    }
    if (rank == 0) {
        std::cout << "Pure MPI 2D run with " << size << " processes; "
                  << "M=" << M << ", N=" << N << std::endl;
    }

    double delta = 1e-6;
    int max_iter = (M - 1) * (N - 1);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    int iterations = solve_mpi(M, N, delta, max_iter,
                                         rank, size, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    if (rank == 0) {
        std::cout << "M=" << M << ", N=" << N
                  << " | Iter=" << iterations
                  << " | Time=" << std::fixed << std::setprecision(6)
                  << elapsed << " s\n";
    }

    MPI_Finalize();
    return 0;
}
