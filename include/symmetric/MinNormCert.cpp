
#include "MinNormCert.h"

#include <chrono>
#include <math.h>


#include <string>
#include <iomanip>
#include <algorithm>


#define EIGEN_USE_LAPACKE_STRICT
// for eigendecomposition
#include <eigen3/Eigen/Dense>
#include <Eigen/Eigenvalues> 

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <sstream>



using namespace std::chrono;




namespace MinNormCert{

    // solve linear system minimum norm
    double solveLinearSystemMinNorm(const Eigen::MatrixXd & A, 
                                    const Eigen::VectorXd & b,
                                    Eigen::VectorXd & sol)
    {
            double tol_rank = 1e-03;
            Eigen::VectorXd y_sol;
            
            Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(A.rows(),
                                                    A.cols());
            cod.setThreshold(tol_rank);
            cod.compute(A);

            y_sol = cod.solve(b);
            
            Eigen::VectorXd e_lin = A * y_sol - b;
            double error_sol = e_lin.squaredNorm();
            sol.resize(A.cols(), 1); 
            sol = y_sol;
            return error_sol;
    }     



    template <typename Matrix, typename VectorX, typename VectorM>    
    MinNormCertResult<Matrix, VectorX, VectorM> MinNormCertClass<Matrix, VectorX, VectorM>::getResults( const VectorX & xm,
                                                                                                        const Matrix & costMatrix,
                                                                                                        const std::vector<Matrix> & constMatrices,
                                                                                                        const VectorM & init_mult) 
            {


                auto start_time_init = high_resolution_clock::now(); 


                // 1. Create matrix
                Eigen::MatrixXd A(xm.rows(), constMatrices.size()); 
                A.setZero(); 
                for(int i=0; i < constMatrices.size(); i++)
                {
                    A.col(i) = constMatrices[i]*xm;        
                }
                  
                Eigen::VectorXd b(xm.rows()); 
                b.setZero(); 
                b = costMatrix * xm - A * init_mult;
                
               
                // Stop timer
                auto duration_init = duration_cast<microseconds>( high_resolution_clock::now() - start_time_init);


                /// Run optimization!

                auto start_opt = high_resolution_clock::now();


                // 2. Solve linear system
                Eigen::VectorXd sol_mult;  

                double error_lin = solveLinearSystemMinNorm(A, b, sol_mult);        
                            
                // update initial estimation if required
                sol_mult += init_mult;

                auto duration_opt = duration_cast<microseconds>(high_resolution_clock::now() - start_opt);

                // 3. Form Hessian 
                Matrix Hess = costMatrix;

                for (int i=0; i < constMatrices.size(); i++)
                {
                    Hess -= sol_mult(i) * constMatrices[i];        
                }
               

                auto duration_total = duration_cast<microseconds>(high_resolution_clock::now() - start_time_init); 



                MinNormCertResult<Matrix, VectorX, VectorM> result;
                //  assign all the results to this struct
                result.f_hat = error_lin;
                           
                result.opt_mult = sol_mult;
                result.opt_Hessian = Hess;
                // Save time in microsecs
                result.elapsed_init_time = duration_init.count();
                result.elapsed_iterative_time = duration_opt.count();  // in microsecs
                result.elapsed_estimation_time = duration_total.count();           


               return result;

    }; // end of fcn getResult

}  // end of essential namespace
