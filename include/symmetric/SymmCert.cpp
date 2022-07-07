

#include "SymmCert.h"
// #include "SymmCertUtils.h"
#include "SymmCertProblem.h"
#include "SymmCertProblem.cpp"

// R. optimizayion
#include <experimental/optional>
#include <Optimization/Riemannian/TNT.h>
#include "Optimization/Convex/Concepts.h"
#include "Optimization/Riemannian/GradientDescent.h"
#include "Optimization/Convex/ProximalGradient.h"

#include <functional>
#include <chrono>
#include <math.h>


#include <string>
#include <iomanip>
#include <algorithm>



using namespace std::chrono;
using namespace Optimization;
using namespace Optimization::Convex;




namespace SymmCert{

    template <typename Matrix, typename VectorX, typename VectorM>    
    SymmCertResult<Matrix, VectorX, VectorM> SymmCertClass<Matrix, VectorX, VectorM>::getResults( const VectorX & xm,
                                                                                                  const Matrix & costMatrix,
                                                                                                  const std::vector<Matrix> & constMatrices,
                                                                                                  const VectorM & init_mult, 
                                                                                                  const Matrix & init_Hessian) 
             {


               auto start_time_init = high_resolution_clock::now(); 


               VarCert cert_init(init_Hessian, init_mult);

               
               /// Define the problem
  
               SymmCertProblem<Matrix, VectorX, VectorM> problem(xm, costMatrix, constMatrices, init_mult, init_Hessian);

    
               // Cache these three matrices for gradient & hessian
               ProblemCachedMatrices<Matrix, VectorX, VectorM> problem_matrices;


               // compute precon if needed
               // set precon matrix
                problem.setMatrixPrecon();

                /// Function handles required by the TNT optimization algorithm
                // Preconditioning operator (optional)
                  std::experimental::optional<Optimization::Riemannian::LinearOperator<VarCert, VarCert, ProblemCachedMatrices<Matrix, VectorX, VectorM>>> precon;

                  
                    Optimization::Riemannian::LinearOperator<VarCert, VarCert, ProblemCachedMatrices<Matrix, VectorX, VectorM>> precon_op =
                        [&problem](const VarCert &Y, const VarCert &Ydot,
                                   const ProblemCachedMatrices<Matrix, VectorX, VectorM> & problem_matrices) {
                          return problem.precondition(Y, Ydot);
                        };
                    precon = precon_op;
                  
                  
                  
 
              // Objective
              Optimization::Objective<VarCert, double, ProblemCachedMatrices<Matrix, VectorX, VectorM>> F =
                  [&problem](const VarCert &Y, ProblemCachedMatrices<Matrix, VectorX, VectorM> & problem_matrices){
                    return problem.evaluate_objective(Y, problem_matrices);
                  };


              /// Gradient
              Optimization::Riemannian::VectorField<VarCert, VarCert, ProblemCachedMatrices<Matrix, VectorX, VectorM>> grad_F = 
                                                [&problem](const VarCert &Y,
                                                ProblemCachedMatrices<Matrix, VectorX, VectorM> & problem_matrices) {
                                                
                // Compute and cache Euclidean gradient at the current iterate
                problem_matrices.NablaF_Y = problem.Euclidean_gradient(Y, problem_matrices);
                // Compute Riemannian gradient from Euclidean one
                return problem.Riemannian_gradient(Y, problem_matrices.NablaF_Y);
              };
              
              

              // Local quadratic model constructor
              Optimization::Riemannian::QuadraticModel<VarCert, VarCert, ProblemCachedMatrices<Matrix, VectorX, VectorM>> QM =
                  [&problem](
                      const VarCert &Y, VarCert &grad,
                      Optimization::Riemannian::LinearOperator<VarCert, VarCert, ProblemCachedMatrices<Matrix, VectorX, VectorM>> &HessOp,
                      ProblemCachedMatrices<Matrix, VectorX, VectorM> & problem_matrices) {
                    // Compute and cache Euclidean gradient at the current iterate

                    
                    problem_matrices.NablaF_Y  = problem.Euclidean_gradient(Y, problem_matrices);
                    // Compute Riemannian gradient from Euclidean gradient
                    
                    grad = problem.Riemannian_gradient(Y, problem_matrices.NablaF_Y );

                    // Define linear operator for computing Riemannian Hessian-vector
                    // products
                    HessOp = [&problem](const VarCert &Y, const VarCert &Ydot,
                                        const ProblemCachedMatrices<Matrix, VectorX, VectorM> & problem_matrices) {
                                        VarCert Hss = problem.Riemannian_Hessian_vector_product(Y, problem_matrices, Ydot);

                                        return Hss;
                    };
                  };

                  // Riemannian metric

                  // We consider a realization of the product of Stiefel manifolds as an
                  // embedded submanifold of R^{r x dn}; consequently, the induced Riemannian
                  // metric is simply the usual Euclidean inner product
                  Optimization::Riemannian::RiemannianMetric<VarCert, VarCert, double, ProblemCachedMatrices<Matrix, VectorX, VectorM>>
                      metric = [&problem](const VarCert &Y, const VarCert &V1, const VarCert &V2,
                                          const ProblemCachedMatrices<Matrix, VectorX, VectorM> & problem_matrices) { 
                                         
                        return ((V1.H.transpose() * V2.H).trace() + (V1.mult.dot(V2.mult)));
                      };

              // Retraction operator
              Optimization::Riemannian::Retraction<VarCert, VarCert, ProblemCachedMatrices<Matrix, VectorX, VectorM>> retraction =
                  [&problem](const VarCert &Y, const VarCert &Ydot, const ProblemCachedMatrices<Matrix, VectorX, VectorM> & problem_matrices) {
                    return problem.retract(Y, Ydot);
                  };



              // Stop timer
              auto duration_init = duration_cast<microseconds>( high_resolution_clock::now() - start_time_init);



              // set up params for solver
              Optimization::Riemannian::TNTParams<double> params;
              params.gradient_tolerance = options_.tol_grad_norm;
              params.relative_decrease_tolerance = options_.rel_func_decrease_tol;
              params.max_iterations = options_.max_RTR_iterations;
              params.max_TPCG_iterations = options_.max_tCG_iterations;
              params.preconditioned_gradient_tolerance = options_.preconditioned_grad_norm_tol;
              params.stepsize_tolerance = options_.stepsize_tol;
              params.verbose = options_.verbose;
              
              
              /** An optional user-supplied function that can be used to instrument/monitor
               * the performance of the internal Riemannian truncated-Newton trust-region
               * optimization algorithm as it runs. */
              // std::experimental::optional<EssentialTNTUserFunction> user_fcn;

              /// Run optimization!
         
              auto start_opt = high_resolution_clock::now();

            
              Optimization::Riemannian::TNTResult<VarCert, double> TNTResults = Optimization::Riemannian::TNT<VarCert, VarCert, double, ProblemCachedMatrices<Matrix, VectorX, VectorM>>(
                        F, QM, metric, retraction, cert_init, problem_matrices, precon, params);
 

               auto duration_opt = duration_cast<microseconds>(high_resolution_clock::now() - start_opt);
               auto duration_total = duration_cast<microseconds>(high_resolution_clock::now() - start_time_init); 
                

   
                 
	           VarCert Rs_opt = TNTResults.x;
	           VectorM opt_mult = Rs_opt.mult; 
                   Matrix opt_Hessian = Rs_opt.H; 

                   SymmCertResult<Matrix, VectorX, VectorM> result;
	           //  assign all the results to this struct
	           result.f_hat = TNTResults.f;
	               	               
                   result.opt_mult = opt_mult;
                   result.opt_Hessian = opt_Hessian;
                   // Save time in microsecs
                   result.elapsed_init_time = duration_init.count();
                   result.elapsed_iterative_time = duration_opt.count();  // in microsecs
                   result.elapsed_estimation_time = duration_total.count();           


               return result;

    }; // end of fcn getResult

template <typename Matrix, typename VectorX, typename VectorM>    
void SymmCertClass<Matrix, VectorX, VectorM>::printResult(SymmCertResult<Matrix, VectorX, VectorM> & my_result)
{
    // print data

      std::cout << "Data from estimation with symmetric matrix\n--------------------\n";

      std::cout << "######## Times [in microseconds]:\n";
      std::cout << "- Total init: " << my_result.elapsed_init_time << std::endl;

      std::cout << "- Iterative Method: " << my_result.elapsed_iterative_time << std::endl;

      std::cout << "---------------------\n";
      std::cout << "- Total time: " << my_result.elapsed_estimation_time << std::endl << std::endl;

      std::cout << "\n Recovered multipliers:\n" << my_result.opt_mult << std::endl;
      std::cout << "\n Recovered Hessian:\n" << my_result.opt_Hessian << std::endl;          
    

};  //end of print fcn

}  // end of essential namespace
