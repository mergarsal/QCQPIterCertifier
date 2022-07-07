#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>



namespace MinNormCert
{

        
/** This struct contains the output of the Essential Estimation */
template <typename Matrix, typename VectorX, typename VectorM>
struct MinNormCertResult {

  
   // Primal objective value
   double f_hat;

 
   // Elapsed time for the initialisation
   double elapsed_init_time;
   // Elapsed time for the optimization on manifold
   double elapsed_iterative_time;
   // Elapsed time for the whole estimation: initialisation + optimization 
   double elapsed_estimation_time;

   
   // Output lagrange multipliers
   VectorM opt_mult;
   // Output Hessin matrix
   Matrix opt_Hessian;

   /* Default constructor */
   MinNormCertResult() {};

}; // end of struct



template <typename Matrix, typename VectorX, typename VectorM>    
class MinNormCertClass
{
       
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
    /* Default constructor */
    MinNormCertClass(void){};
                          

    ~MinNormCertClass(void){};


    MinNormCertResult<Matrix, VectorX, VectorM> getResults(const VectorX & xm,
                                                        const Matrix & costMatrix,
                                                        const std::vector<Matrix> & constMatrices,
                                                        const VectorM & init_mult); 
                         
        

};  //end of RotPrior class



}  // end of RotPrior namespace


