
#include "RankYCert.h"
#include "RankYCertProblem.h"
#include "RankYCertProblem.cpp"

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




namespace RankYCert{


        
    template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
    bool RankYCertClass<Matrix, MatrixY, VectorX, VectorM>::escape_saddle_rank_y(const VectorX & xm,
                                                                                 const Matrix & costMatrix,
                                                                                 const std::vector<Matrix> & constMatrices,
                                                                                 const VectorM & init_mult,  
                                                                                 const MatrixY & Yprev, 
                                                                                 Eigen::MatrixXd & Yplus) 
    {
    
       
        // TODO IMPROVE THIS
        RankYCertProblem<Matrix, MatrixY, VectorX, VectorM> problem(xm, costMatrix, constMatrices, init_mult, 
                                                Yprev, options_.threshold_cost, options_.threshold_min_svd, options_.use_euc_simplification);

        // Cache these three matrices for gradient & hessian
        ProblemCachedMatrices<Matrix> problem_matrices;
        problem.setMatrixPrecon();
        
        
        int rows = Yprev.rows(); 
        int cols = Yprev.cols(); 
        int rank_y = cols+1;
        // variables
        Eigen::MatrixXd Ynext(rows, rank_y);
        Ynext.setZero(); 
        Ynext.leftCols(cols) = Yprev;
        
        Eigen::VectorXd v_min(rank_y);
        
        VarCert Xprev(Ynext, init_mult);     
        
        // 0. we need to call objective for problem_matrices
        double cost_prev = problem.evaluate_objective(Xprev, problem_matrices);
        // std::cout << "Cost at original point: " << cost_prev << std::endl;
        
        // 1. Get Hessian and compute descent direction
        double lambda_min = problem.getVectorEucHessian(Yprev, problem_matrices, v_min);
        Eigen::MatrixXd Ydot(rows, rank_y); 
        Ydot.setZero();
        Ydot.rightCols(1) = v_min;
        VarCert Xdot(Ydot, VectorM::Zero());                              
        // std::cout << "Minimum eigenvalue Hessian: " << lambda_min << std::endl; 
        //  std::cout << "Descent direction:\n" << v_min<< std::endl; 
         
                                      
          /* From here we follow Se-SynC */
          // Set the initial step length to the greater of 10 times the distance needed
          // to arrive at a trial point whose gradient is large enough to avoid
          // triggering the gradient norm tolerance stopping condition (according to the
          // local second-order model), or at least 2^4 times the minimum admissible
          // steplength,
          double alpha_min = 1e-6; // Minimum stepsize
          double alpha =
              std::max(16 * alpha_min, 100 * options_.tol_grad_norm / fabs(lambda_min));
              

        // TODO: eliminar computation innces. 
          // Vectors of trial stepsizes and corresponding function values
          std::vector<double> alphas;
          std::vector<double> fvals;

          /// Backtracking line search
          VarCert Xtest = Xprev;
          bool step_accepted = false;
          while (alpha >= alpha_min) {

            // Retract along the given tangent vector using the given stepsize
            Xtest = problem.retract(Xprev, alpha * Xdot);
            

            // Ensure that the trial point Ytest has a lower function value than
            // the current iterate Y, and that the gradient at Ytest is
            // sufficiently large that we will not automatically trigger the
            // gradient tolerance stopping criterion at the next iteration
            double FYtest = problem.evaluate_objective(Xtest, problem_matrices);
            
                         
            problem_matrices.NablaF_Y = problem.Euclidean_gradient(Xtest, problem_matrices);
                
                // Compute Riemannian gradient from Euclidean one
            VarCert grad_FYtest = problem.Riemannian_gradient(Xtest, problem_matrices.NablaF_Y);
                
            
            double grad_FYtest_norm = grad_FYtest.norm();
            double preconditioned_grad_FYtest_norm =
                problem.precondition(Xtest, grad_FYtest).norm();

            // Record trial stepsize and function value
            alphas.push_back(alpha);
            fvals.push_back(FYtest);
            // std::cout << "Step alpha: " << alpha << std::endl;
            // std::cout << "Cost in line search: " << FYtest << std::endl; 
            // std::cout << "grad norm: " << grad_FYtest_norm << std::endl; 
            // std::cout << "precon grad nomr: " << preconditioned_grad_FYtest_norm << std::endl;
            
            if ((FYtest < cost_prev)) // && (grad_FYtest_norm > options_.tol_grad_norm) &&
               //  (preconditioned_grad_FYtest_norm > options_.preconditioned_grad_norm_tol))
                 {
              // Accept this trial point and return success
              // std::cout << "Accepting solution in line search!!\n";
              Yplus = Xtest.Y;
              step_accepted = true;
              break;
            }
             
            alpha /= 2;
          }

          // If control reaches here, we failed to find a trial point that satisfied
          // *both* the function decrease *and* gradient bounds.  In order to make
          // forward progress, we will fall back to accepting the trial point that
          // simply minimized the objective value, provided that it strictly *decreased*
          // the objective from the current (saddle) point
          if (!step_accepted)
              {
              // try again
              // Find minimum function value from among the trial points
              auto fmin_iter = std::min_element(fvals.begin(), fvals.end());
              auto min_idx = std::distance(fvals.begin(), fmin_iter);

              double f_min = fvals[min_idx];
              double a_min = alphas[min_idx];
              // std::cout << "Min cost after: " << f_min << std::endl; 
              // std::cout << "Alpha with min cost: " << a_min << std::endl;
              
              if (f_min < cost_prev) {
                // If this trial point strictly decreased the objective value, accept it and
                // return success
                Xtest = problem.retract(Xprev, a_min * Xdot);
                Yplus = Xtest.Y;

                step_accepted = true;
              } 
          }
                    
           if (step_accepted)
           {
          
                Eigen::MatrixXd yty = Yplus.transpose() * Yplus; 
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_ss(yty);
                                
                Eigen::VectorXd lambdas = eig_ss.eigenvalues();
                double cond_number_y = lambdas(0) / lambdas(lambdas.rows()-1);
                // std::cout << "Condition number for solution: " << cond_number_y << std::endl;
                // std::cout << "Vector eigenvalues:\n" << lambdas << std::endl;
                                 
                // Only return the solution as valid
                // if it is not rank deficient
                if (cond_number_y > options_.threshold_min_svd)
                { 
                   //  std::cout << "Returning value from line search\n"; 
                    return true;
                }
                
                else
                {
                   //  std::cout << "NOOOO Returning value from line search\n"; 
                    return false;
                }    
                                                                           
          }
          // if step is not accepted
          return false;         
                                                                                 
    }
                    
    template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
    void RankYCertClass<Matrix, MatrixY, VectorX, VectorM>::checkFeasibilitySol(const VectorX & xm, const std::vector<Matrix> & constMatrices) 
    {
            
                int s_x = xm.rows(); 
                Matrix C = Matrix::Zero(s_x, s_x); 
                MatrixY init_H = MatrixY::Ones();
                VectorM init_m = VectorM::Zero();

               /// Define the problem
  
               RankYCertProblem<Matrix, MatrixY, VectorX, VectorM> problem(xm, C, constMatrices, init_m, init_H);                               
               
               problem.checkConstraints();  
    
    }
    
    template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
    void RankYCertClass<Matrix, MatrixY, VectorX, VectorM>::checkConstIndp(const std::vector<Matrix> & constMatrices) 
    {
            
                int s_x = constMatrices[0].rows(); 
                Matrix C = Matrix::Zero(s_x, s_x); 
                MatrixY init_H = MatrixY::Ones();
                VectorM init_m = VectorM::Zero();
                VectorX xm = VectorX::Zero(); 

               /// Define the problem
  
               RankYCertProblem<Matrix, MatrixY, VectorX, VectorM> problem(xm, C, constMatrices, init_m, init_H);                               
               
               problem.checkConstraintsIndependence();  
    
    }
    
    
    template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
    RankYCertResult<Matrix, MatrixY, VectorM> RankYCertClass<Matrix, MatrixY, VectorX, VectorM>::getResults(const VectorX & xm,
                                                                                                            const Matrix & costMatrix,
                                                                                                            const std::vector<Matrix> & constMatrices,
                                                                                                            const VectorM & init_mult, 
                                                                                                            const MatrixY & init_Hessian) 
    {

               auto start_time_init = high_resolution_clock::now(); 


               VarCert cert_init(init_Hessian, init_mult);

               RankYStatus opt_status = RankYStatus::Y_FULL_RANK;

               /// Define the problem
  
               RankYCertProblem<Matrix, MatrixY, VectorX, VectorM> problem(xm, costMatrix, constMatrices, init_mult, 
                                                init_Hessian, options_.threshold_cost, options_.threshold_cost_norm, 
                                                options_.threshold_min_svd, options_.use_euc_simplification);

               // Cache these three matrices for gradient & hessian
               ProblemCachedMatrices<Matrix> problem_matrices;

               
               // compute precon if needed
               // set precon matrix
                problem.setMatrixPrecon();

                /// Function handles required by the TNT optimization algorithm
                // Preconditioning operator (optional)
                  std::experimental::optional<Optimization::Riemannian::LinearOperator<VarCert, VarCert, ProblemCachedMatrices<Matrix>>> precon;

                  
                    Optimization::Riemannian::LinearOperator<VarCert, VarCert, ProblemCachedMatrices<Matrix>> precon_op =
                        [&problem](const VarCert &Y, const VarCert &Ydot,
                                   const ProblemCachedMatrices<Matrix> & problem_matrices) {
                          return problem.precondition(Y, Ydot);
                        };
                    precon = precon_op;
                    
                    
              // Objective
              Optimization::Objective<VarCert, double, ProblemCachedMatrices<Matrix>> F =
                  [&problem](const VarCert &Y, ProblemCachedMatrices<Matrix> & problem_matrices){
                    return problem.evaluate_objective(Y, problem_matrices);
                  };


             /// Gradient
              Optimization::Riemannian::VectorField<VarCert, VarCert, ProblemCachedMatrices<Matrix>> grad_F = 
                                                [&problem](const VarCert &Y,
                                                ProblemCachedMatrices<Matrix> & problem_matrices) {
                                                
                // Compute and cache Euclidean gradient at the current iterate
                problem_matrices.NablaF_Y = problem.Euclidean_gradient(Y, problem_matrices);
                
                // Compute Riemannian gradient from Euclidean one
                return problem.Riemannian_gradient(Y, problem_matrices.NablaF_Y);
              };
              
              

              // Local quadratic model constructor
              Optimization::Riemannian::QuadraticModel<VarCert, VarCert, ProblemCachedMatrices<Matrix>> QM =
                  [&problem](
                      const VarCert &Y, VarCert &grad,
                      Optimization::Riemannian::LinearOperator<VarCert, VarCert, ProblemCachedMatrices<Matrix>> &HessOp,
                      ProblemCachedMatrices<Matrix> & problem_matrices) {
                    // Compute and cache Euclidean gradient at the current iterate

                    
                    problem_matrices.NablaF_Y  = problem.Euclidean_gradient(Y, problem_matrices);
                    // Compute Riemannian gradient from Euclidean gradient
                    
                    grad = problem.Riemannian_gradient(Y, problem_matrices.NablaF_Y );
                    
                    // Define linear operator for computing Riemannian Hessian-vector
                    // products
                    HessOp = [&problem](const VarCert &Y, const VarCert &Ydot,
                                        const ProblemCachedMatrices<Matrix> & problem_matrices) {
                                        VarCert Hss = problem.Riemannian_Hessian_vector_product(Y, problem_matrices, Ydot);

                                        return Hss;
                    };
                  };

                  // Riemannian metric

                  // We consider a realization of the product of Stiefel manifolds as an
                  // embedded submanifold of R^{r x dn}; consequently, the induced Riemannian
                  // metric is simply the usual Euclidean inner product
                  Optimization::Riemannian::RiemannianMetric<VarCert, VarCert, double, ProblemCachedMatrices<Matrix>>
                      metric = [&problem](const VarCert &Y, const VarCert &V1, const VarCert &V2,
                                          const ProblemCachedMatrices<Matrix> & problem_matrices) { 
                                         
                        return ((V1.Y.transpose() * V2.Y).trace() + (V1.mult.dot(V2.mult)));
                      };

              // Retraction operator
              Optimization::Riemannian::Retraction<VarCert, VarCert, ProblemCachedMatrices<Matrix>> retraction =
                  [&problem](const VarCert &Y, const VarCert &Ydot, const ProblemCachedMatrices<Matrix> & problem_matrices) {
                    return problem.retract(Y, Ydot);
                  };



              // Stop timer
              auto duration_init = duration_cast<microseconds>( high_resolution_clock::now() - start_time_init);



              // set up params for solver
              Optimization::Riemannian::TNTParams<double> params = Optimization::Riemannian::TNTParams<double>();
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

              Optimization::Riemannian::TNTUserFunction<VarCert, VarCert, double, ProblemCachedMatrices<Matrix>> user_fcn = 
             [&problem](size_t i, double t, const VarCert &x, double f,
                       const VarCert &g,
                       const Optimization::Riemannian::LinearOperator<VarCert, VarCert, ProblemCachedMatrices<Matrix>> &HessOp,
                       double Delta, size_t num_STPCG_iters, const VarCert &h,
                       double df, double rho, bool accepted, ProblemCachedMatrices<Matrix> & problem_matrices)
                       {
                        bool stop_conv_cost = problem.stopFcnCost(i, f, df, problem.get_threshold_cost(), problem.get_threshold_cost_norm(), 1e-10);     
                        
                        bool stop_def_y = problem.stopFcnRank(i, x.Y, problem_matrices, problem.get_threshold_min_svd());     
                        
                        
                                       
                        return (stop_conv_cost || stop_def_y);                   
                       };
                       
             
             
             
              /// Run optimization!
         
              auto start_opt = high_resolution_clock::now();

              // user function after params
              Optimization::Riemannian::TNTResult<VarCert, double> TNTResults = Optimization::Riemannian::TNT<VarCert, VarCert, double, ProblemCachedMatrices<Matrix>>(
                        F, QM, metric, retraction, cert_init, problem_matrices, precon, params, user_fcn);
 

               auto duration_opt = duration_cast<microseconds>(high_resolution_clock::now() - start_opt);
               auto duration_total = duration_cast<microseconds>(high_resolution_clock::now() - start_time_init); 
                

               // check condition on cost
               
               if (TNTResults.status == Optimization::Riemannian::TNTStatus::UserFunction) 
               {
                
                    
                    if (TNTResults.f <= options_.threshold_cost)
                        opt_status = RankYStatus::STOP_COST;
                        
                    else if (problem_matrices.cond_number_Y <= options_.threshold_min_svd)
                    {
                        opt_status = RankYStatus::Y_RANK_DEF;
                        
                    }    
               }
               
               
               if (options_.verbose)
                    std::cout << "Cost : " << TNTResults.f << std::endl;
                    
	       VarCert Rs_opt = TNTResults.x;
	       Eigen::Matrix<double, Eigen::Dynamic, 1> opt_mult = Rs_opt.mult; 
               Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> opt_Y = Rs_opt.Y; 
               Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> opt_Hessian = Rs_opt.Y * Rs_opt.Y.transpose(); 

               RankYCertResult<Matrix, MatrixY, VectorM> result = RankYCertResult<Matrix, MatrixY, VectorM>();
	       //  assign all the results to this struct
	       result.f_hat = TNTResults.f;	               	               
               result.opt_mult = opt_mult;
               result.opt_Hessian = opt_Hessian;
               result.status_Y = opt_status;
               result.opt_Y = opt_Y;
               // Save time in microsecs
               result.elapsed_init_time = duration_init.count();
               result.elapsed_iterative_time = duration_opt.count();  // in microsecs
               result.elapsed_estimation_time = duration_total.count();           

                // Save svd Y                
                result.cond_number_Y = problem_matrices.cond_number_Y;
                result.absolute_min_svd_Y = problem_matrices.abs_min_svd_Y;
                
               return result;

    }; // end of fcn getResult
    
    

template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
void RankYCertClass<Matrix, MatrixY, VectorX, VectorM>::printResult(RankYCertResult<Matrix, MatrixY, VectorM> & my_result)
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
