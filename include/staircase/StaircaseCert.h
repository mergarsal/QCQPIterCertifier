#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>



namespace StaircaseCert
{



struct StaircaseCertOptions {

      /// OPTIMIZATION STOPPING CRITERIA
      /** Stopping tolerance for the norm of the Riemannian gradient */
      double tol_grad_norm = 1.e-9;

       /** Stopping criterion based upon the norm of an accepted update step */
      double preconditioned_grad_norm_tol = 1.e-9;

      /** Stopping criterion based upon the relative decrease in function value */
      double rel_func_decrease_tol = 1e-13;

       /** Stopping criterion based upon the norm of an accepted update step */
      double stepsize_tol = 1e-07;

       /** Gradient tolerance for the truncated preconditioned conjugate gradient
   * solver: stop if ||g|| < kappa * ||g_0||.  This parameter should be in the
   * range (0,1). */
      double STPCG_kappa = 0.7;

      /** Gradient tolerance based upon a fractional-power reduction in the norm of
   * the gradient: stop if ||g|| < ||kappa||^{1+ theta}.  This value should be
   * positive, and controls the asymptotic convergence rate of the
   * truncated-Newton trust-region solver: specifically, for theta > 0, the TNT
   * algorithm converges q-superlinearly with order (1+theta). */
      double STPCG_theta = .5;

    
      /** Maximum permitted number of (outer) iterations of the RTR algorithm */
      unsigned int max_RTR_iterations = 10;
      /** Maximum number of inner (truncated conjugate-gradient) iterations to
      * perform per out iteration */
      unsigned int max_tCG_iterations = 5;

      // verbose = {0, 1}
      unsigned int estimation_verbose = 1;
      
      unsigned int verbose = 1;  // for TNT
      
      /* For the staircase algorithm */
      unsigned int max_RTR_iterations_staircase = 10; 
      
      unsigned int max_n_iter_staircase = 10; 
      
      double threshold_cost = 1e-09;
      double threshold_cost_norm = 1e-11;      
      double threshold_min_svd = 1e-06;
      
      unsigned int initial_rank = 1;
      
      unsigned int min_rank = 1;
      
      unsigned int max_rank = 1;
      
      bool record_stairs = false;  
      // if true, we save in StaircaseCertResult 
      // the iterations
      
      // use euclidean simplification for S=YYt
      bool use_euc_simplification = false; 
      
      // use the original decomposition of Hinit 
      // for the next iteration 
      // during the Riem. staircase
      bool use_original_H_up = false;
      // ONLY if use_original_H_up is true, 
      // we use the next variables
      Eigen::MatrixXd matrixU; 
      Eigen::VectorXd singularValues; 
      
      /// DEFAULT CONSTRUCTOR with default values

      StaircaseCertOptions() {};
       

};  // end of struct: StaircaseCertOptions


enum class CertStatus
{
    MIN_RANK = 0,
    MAX_RANK,
    STRONG_DUALITY, 
    MAX_ITERS,
    INCONCLUSIVE, 
    ERROR_STAIR
}; // end of struct status



template <typename Matrix, typename MatrixY, typename VectorM>    
/** This struct contains the output of the Essential Estimation */
struct StaircaseCertResult {

  
   unsigned int number_stairs;  

   // Total times
   double total_init_time;
   // Elapsed time for the optimization on manifold
   double total_iterative_time;
   // Elapsed time for the whole estimation: initialisation + optimization + verification
   double total_estimation_time;
   
   // Primal objective value
   double f_hat;
   // Output lagrange multipliers
   VectorM opt_mult;
   // Output Y matrix: H = Y * Y ^t
   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> opt_Y;
   // Output Hessin matrix
   Matrix opt_Hessian;
   // Elapsed time for the initialisation
   double elapsed_init_time;
   // Elapsed time for the optimization on manifold
   double elapsed_iterative_time;
   // Elapsed time for the whole estimation: initialisation + optimization + verification
   double elapsed_estimation_time;
   // SVD decomposition of Y
   // min / max
   double cond_number_Y; 
   // min
   double absolute_min_svd_Y;


   std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> Y_stairs;
   std::vector<double> f_stairs;
   
   CertStatus status_cert; 
   
   /* Default constructor */
   StaircaseCertResult() {};

}; // end of struct



template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
class StaircaseCertClass
{
       
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
    /* Default constructor */
    StaircaseCertClass(void){};

                    
    StaircaseCertClass( const StaircaseCertOptions & options = StaircaseCertOptions()): options_(options){};        
                           

    ~StaircaseCertClass(void){};


    StaircaseCertResult<Matrix, MatrixY, VectorM> getResults(const VectorX & xm,
                                                             const Matrix & costMatrix,
                                                             const std::vector<Matrix> & constMatrices,
                                                             const VectorM & init_mult, 
                                                             const MatrixY & init_Hessian); 

    void printResult(StaircaseCertResult<Matrix, MatrixY, VectorM> & results);

    void checkFeasibilitySol(const VectorX & xm, const std::vector<Matrix> & constMatrices); 
    void checkConstIndp(const std::vector<Matrix> & constMatrices); 
    
    private:

        StaircaseCertOptions options_;                              
        

};  //end of StaircaseCert Class



}  // end of namespace
