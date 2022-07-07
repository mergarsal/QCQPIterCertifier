#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>


#include "SymmCertTypes.h"



namespace SymmCert
{

    
    /** This struct contains the various parameters that control the algorithm 
        Based on: SESync struct*/
    struct SymmCertOptions {

      /// OPTIMIZATION STOPPING CRITERIA
      /** Stopping tolerance for the norm of the Riemannian gradient */
      double tol_grad_norm = 1.e-9;

       /** Stopping criterion based upon the norm of an accepted update step */
      double preconditioned_grad_norm_tol = 1.e-9;

      /** Stopping criterion based upon the relative decrease in function value */
      double rel_func_decrease_tol = 1e-13;

       /** Stopping criterion based upon the norm of an accepted update step */
      double stepsize_tol = 1e-05;

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


      /// DEFAULT CONSTRUCTOR with default values

       SymmCertOptions() {};
       

};  // end of struct: EssentialEstimationOptions


    
/** This struct contains the output of the Essential Estimation */
template <typename Matrix, typename VectorX, typename VectorM>
struct SymmCertResult {

  
   // Primal objective value
   double f_hat;

 
   // Elapsed time for the initialisation
   double elapsed_init_time;
   // Elapsed time for the optimization on manifold
   double elapsed_iterative_time;
   // Elapsed time for the whole estimation: initialisation + optimization + verification
   double elapsed_estimation_time;

   
   // Output lagrange multipliers
   VectorM opt_mult;
   // Output Hessin matrix
   Matrix opt_Hessian;

   /* Default constructor */
   SymmCertResult() {};

}; // end of struct



template <typename Matrix, typename VectorX, typename VectorM>    
class SymmCertClass
{
       
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
    /* Default constructor */
    SymmCertClass(void){};

                    
    SymmCertClass( const SymmCertOptions & options = SymmCertOptions()): options_(options){};        
                           

    ~SymmCertClass(void){};


    SymmCertResult<Matrix, VectorX, VectorM> getResults(const VectorX & xm,
                                                        const Matrix & costMatrix,
                                                        const std::vector<Matrix> & constMatrices,
                                                        const VectorM & init_mult, 
                                                        const Matrix & init_Hessian); 

    void printResult(SymmCertResult<Matrix, VectorX, VectorM> & results);

    private:

        SymmCertOptions options_;                              
        

};  //end of RotPrior class


template <typename T>
VarCert operator*(const T b, const VarCert & a) 
{
      return VarCert(b * a.H, b * a.mult);
} ; 
     
       


}  // end of RotPrior namespace


