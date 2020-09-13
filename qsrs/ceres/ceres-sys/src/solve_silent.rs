use crate::ceres::ceres_problem_t;
use cpp::cpp;
cpp! {{
    #include <ceres/c_api.h>
    #include <ceres/ceres.h>
}}

/// # Safety
/// The problem is initialized/used in lib.rs
pub unsafe fn ceres_solve_silent(c_problem: *mut ceres_problem_t, max_iters: usize, num_threads: usize, ftol: f64, gtol: f64) {
    cpp!([c_problem as "ceres_problem_t *", max_iters as "size_t", num_threads as "size_t", ftol as "double", gtol as "double"] {
        ceres::Problem* problem = reinterpret_cast<ceres::Problem*>(c_problem);

        ceres::Solver::Options options;
        options.max_num_iterations = max_iters;
        options.linear_solver_type = ceres::DENSE_QR;
        options.num_threads = num_threads;
        options.minimizer_progress_to_stdout = false;
        options.function_tolerance = ftol;
        options.gradient_tolerance = gtol;
        // Ceres outputs a *lot* of logs, so we silence them here for our own uses
        options.logging_type = ceres::SILENT;

        ceres::Solver::Summary summary;
        ceres::Solve(options, problem, &summary);
        // good for debugging
        //std::cout << summary.FullReport() << "\n";
    })
}
