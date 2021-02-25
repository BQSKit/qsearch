import qsearch
import numpy as np
from qsearch import unitaries, advanced_unitaries, evaluation, solvers_future, solvers_higher_order
import time

NUM_REPEATS = 128

outlogger = qsearch.logging.Logger(stdout_enabled=True, output_file="experiment_result.csv")
configs = [
        {
            "error_residuals" : qsearch.evaluation.residuals_product,
            "error_residuals_jac" : qsearch.evaluation.residuals_product_jac,
            "error_func" : qsearch.evaluation.error_distsq,
            "error_jac" : qsearch.comparison.matrix_distance_squared_jac,
            "solver" : qsearch.solvers_future.LeastSquares_Jac_Solver_Future(),
            "objective" : solvers_higher_order.MatrixDistanceObjective(),
            "runname" : "Futurefunc - LeastSquares",
            "eval_func" : qsearch.comparison.matrix_distance_squared
        },
        {
            "error_residuals" : qsearch.comparison.matrix_residuals,
            "error_residuals_jac" : qsearch.comparison.matrix_residuals_jac,
            "error_func" : qsearch.comparison.matrix_distance_squared,
            "error_jac" : qsearch.comparison.matrix_distance_squared_jac,
            "solver" : qsearch.solvers.LeastSquares_Jac_Solver(),
            "objective" : solvers_higher_order.MatrixDistanceObjective(),
            "runname" : "Original - LeastSquares",
            "eval_func" : qsearch.comparison.matrix_distance_squared
        },
        {
            "error_residuals" : qsearch.comparison.matrix_residuals,
            "error_residuals_jac" : qsearch.comparison.matrix_residuals_jac,
            "error_func" : qsearch.comparison.matrix_distance_squared,
            "error_jac" : qsearch.comparison.matrix_distance_squared_jac,
            "solver" : qsearch.solvers_higher_order.LeastSquares_Jac_Solver_Objective(),
            "objective" : solvers_higher_order.MatrixDistanceObjective(),
            "runname" : "Objective - LeastSquares",
            "eval_func" : qsearch.comparison.matrix_distance_squared
        }
    ]

if __name__ == "__main__":
    for config in configs:
        with qsearch.Project("benchmarks") as project:
            project.clear()
            project["error_residuals"] = config["error_residuals"]
            project["error_func"] = config["error_func"]
            project["error_jac" ] = config["error_jac"]
            project["error_residuals_jac"] = config["error_residuals_jac"]
            project["eval_func"] = config["eval_func"]
            project["solver"] = config["solver"]
            project["objective"] = config["objective"]

            project.add_compilation("qft2", unitaries.qft(4))
            #project.add_compilation("qft3", unitaries.qft(8))
            #project.add_compilation("fredkin", unitaries.fredkin)
            #project.add_compilation("toffoli", unitaries.toffoli)
            project.add_compilation("peres", unitaries.peres)
            #project.add_compilation("or", unitaries.logical_or)

            project.add_compilation("miro", advanced_unitaries.mirogate)
            project.add_compilation("hhl", advanced_unitaries.HHL)

            # re-seed numpy once per project.  This helps to add some consistency that might be useful for this particular timing experiment
            np.random.seed(2148461255) # this seed decided by dice roll, as a sum of dice of various sizes.
            print("The first number that comes to mind is {}".format(np.random.rand()))

            # run the benchmarks script with default settings 10x and average the timing results
            # reported times are between 0.1s and 5s on my 2018 Macbook Pro

            times = {}
            for compilation in project.compilations:
                times[compilation] = 0

            for _ in range(NUM_REPEATS):
                project.reset()
                project.run()
                for compilation in project.compilations:
                    times[compilation] += project.get_time(compilation)

            outlogger.logprint(config["runname"])
            for compilation in project.compilations:
                outlogger.logprint(f'{compilation}\t{times[compilation]/NUM_REPEATS}s')

