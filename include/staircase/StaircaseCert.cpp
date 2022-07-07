#pragma once


#include "StaircaseCert.h"

#include "./rankY/RankYCert.h"
#include "./rankY/RankYCert.cpp"

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




namespace StaircaseCert{

    
    template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
    void StaircaseCertClass<Matrix, MatrixY, VectorX, VectorM>::checkFeasibilitySol(const VectorX & xm, const std::vector<Matrix> & constMatrices)
    {
        RankYCert::RankYCertOptions options;
        RankYCert::RankYCertClass<Matrix, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, VectorX, VectorM> cert_iter_i(options);     
        cert_iter_i.checkFeasibilitySol(xm, constMatrices);
    }
    
    
    template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
    void StaircaseCertClass<Matrix, MatrixY, VectorX, VectorM>::checkConstIndp(const std::vector<Matrix> & constMatrices)
    {
        RankYCert::RankYCertOptions options;
        RankYCert::RankYCertClass<Matrix, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, VectorX, VectorM> cert_iter_i(options);     
        cert_iter_i.checkConstIndp(constMatrices);
    }





    template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
    StaircaseCertResult<Matrix, MatrixY, VectorM> StaircaseCertClass<Matrix, MatrixY, VectorX, VectorM>::getResults(const VectorX & xm,
                                                                                                                    const Matrix & costMatrix,
                                                                                                                    const std::vector<Matrix> & constMatrices,
                                                                                                                    const VectorM & init_mult, 
                                                                                                                    const MatrixY & init_Hessian) 
    {

               // get some values from the data
               const int n_vars = xm.size(); 
               const int m_const = constMatrices.size();
               
               const int init_rank = init_Hessian.cols(); 
               
                StaircaseCertResult<Matrix, MatrixY, VectorM> result = StaircaseCertResult<Matrix, MatrixY, VectorM>();
                CertStatus status = CertStatus::MIN_RANK;
                
                // clean std vectors 
                result.Y_stairs.clear(); 
                result.f_stairs.clear();
                
                int iter = 1; 
                
                // Initialize
                int rank_i = init_rank;
                VectorM mult_i = init_mult;
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Y_i = init_Hessian;
                
                if (options_.verbose)
                    std::cout << "####    Riemmanian staircase\n";
                
                // fill options for inner optimization
                RankYCert::RankYCertOptions options = RankYCert::RankYCertOptions();
                options.tol_grad_norm = options_.tol_grad_norm; 
                options.preconditioned_grad_norm_tol = options_.preconditioned_grad_norm_tol;
                options.rel_func_decrease_tol = options_.rel_func_decrease_tol;
                options.stepsize_tol = options_.stepsize_tol;
                options.STPCG_kappa = options_.STPCG_kappa;
                options.STPCG_theta = options_.STPCG_theta;
                options.max_RTR_iterations = options_.max_RTR_iterations_staircase;
                options.max_tCG_iterations = options_.max_tCG_iterations;
                options.estimation_verbose = options_.estimation_verbose;
                options.verbose = options_.verbose;
                options.threshold_min_svd = options_.threshold_min_svd;
                options.threshold_cost = options_.threshold_cost;
                options.threshold_cost_norm = options_.threshold_cost_norm;
                options.use_euc_simplification = options_.use_euc_simplification;
                
                double total_init_time = 0; 
                double total_iterative_time = 0; 
                double total_estimation_time = 0;
                
                bool rank_y_has_been_decreased = false; 
                bool rank_y_has_been_increased = false; 
                
                double prev_cost = -10000;                            
                
                while (true) 
                {
                    if (options_.verbose)
                    {
                        std::cout << "Iteration #" << iter << "           "; 
                        std::cout << "Rank Y: " << rank_i << "           "; 
                        std::cout << "Dimension Y: (" << Y_i.rows() << "," << Y_i.cols() << ")\n"; 
                    }
                                                       
                    RankYCert::RankYCertClass<Matrix, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, VectorX, VectorM> cert_iter_i(options);                   
                   
                    RankYCert::RankYCertResult<Matrix, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, VectorM> my_result = cert_iter_i.getResults(xm, costMatrix, constMatrices, mult_i, Y_i);
                    
                    // save iteration if required
                    if (options_.record_stairs)
                    {
                        result.f_stairs.push_back(my_result.f_hat); 
                        result.Y_stairs.push_back(my_result.opt_Y);
                    }
                    
                    // save times 
                    total_init_time += my_result.elapsed_init_time; 
                    total_iterative_time += my_result.elapsed_iterative_time; 
                    total_estimation_time += my_result.elapsed_estimation_time;
                
                    // reasons to stop:
                    //  Y is full rank AND cost below threshold
                    // if ((my_result.status_Y == RankYCert::RankYStatus::Y_FULL_RANK) && (my_result.f_hat < options_.threshold_cost))
                    if ((my_result.f_hat < options_.threshold_cost) || (my_result.f_hat / (n_vars*n_vars) < options_.threshold_cost_norm))
                        {
                            if (options_.verbose)
                                std::cout << "Y was full rank and cost is below threshold\nReturning optimal result\n" << std::endl; 
                            status = CertStatus::STRONG_DUALITY; 
                            // save results
                            result.f_hat = my_result.f_hat;
                            result.opt_mult = my_result.opt_mult;
                            result.opt_Y = my_result.opt_Y;
                            result.opt_Hessian = my_result.opt_Hessian;
                            result.elapsed_init_time = my_result.elapsed_init_time;
                            result.elapsed_iterative_time = my_result.elapsed_iterative_time;  // in microsecs
                            result.elapsed_estimation_time = my_result.elapsed_estimation_time; 
                            result.cond_number_Y = my_result.cond_number_Y; 
                            result.absolute_min_svd_Y = my_result.absolute_min_svd_Y;
                            break;                        
                        }
                    // if the cost is above the threshold
                    // we try to modify the matrix
                    
                    // 1. Y is rank deficient
                    else if (my_result.status_Y == RankYCert::RankYStatus::Y_RANK_DEF)
                    {
                        if (options_.verbose)
                            std::cout << "Y was rank deficient\nReducing rank!!!!!!!!!!\n" << std::endl; 
                            
                        // 3. Y will have rank below MIN_RANK
                        if (rank_i - 1 < options_.min_rank)
                        {
                            if (options_.verbose)
                                std::cout << "Future rank was below threshold\nExiting staircase" << std::endl; 
                            status = CertStatus::MIN_RANK; 
                            // save results in any case
                            result.f_hat = my_result.f_hat;
                            result.opt_mult = my_result.opt_mult;
                            result.opt_Y = my_result.opt_Y;
                            result.opt_Hessian = my_result.opt_Hessian;
                            result.elapsed_init_time = my_result.elapsed_init_time;
                            result.elapsed_iterative_time = my_result.elapsed_iterative_time;  // in microsecs
                            result.elapsed_estimation_time = my_result.elapsed_estimation_time; 
                            result.cond_number_Y = my_result.cond_number_Y; 
                            result.absolute_min_svd_Y = my_result.absolute_min_svd_Y;
                            break;
                        } 
                        // 4. Max number iterations
                        else if (iter > options_.max_n_iter_staircase)
                        {
                            if (options_.verbose)
                                std::cout << "Maximum number of stairs reached!\n" << std::endl; 
                            status = CertStatus::MAX_ITERS;
                            result.f_hat = my_result.f_hat;
                            result.opt_mult = my_result.opt_mult;
                            result.opt_Y = my_result.opt_Y;
                            result.opt_Hessian = my_result.opt_Hessian;
                            result.elapsed_init_time = my_result.elapsed_init_time;
                            result.elapsed_iterative_time = my_result.elapsed_iterative_time;  // in microsecs
                            result.elapsed_estimation_time = my_result.elapsed_estimation_time; 
                            result.cond_number_Y = my_result.cond_number_Y; 
                            result.absolute_min_svd_Y = my_result.absolute_min_svd_Y;
                            break;                    
                        }
                        else if (rank_y_has_been_increased)
                        {
                            if (options_.verbose)
                                std::cout << "Certifier was inconclusive!\n" << std::endl; 
                            status = CertStatus::INCONCLUSIVE;
                            result.f_hat = my_result.f_hat;
                            result.opt_mult = my_result.opt_mult;
                            result.opt_Hessian = my_result.opt_Hessian;
                            result.opt_Y = my_result.opt_Y;
                            result.elapsed_init_time = my_result.elapsed_init_time;
                            result.elapsed_iterative_time = my_result.elapsed_iterative_time;  // in microsecs
                            result.elapsed_estimation_time = my_result.elapsed_estimation_time; 
                            result.cond_number_Y = my_result.cond_number_Y; 
                            result.absolute_min_svd_Y = my_result.absolute_min_svd_Y;
                            break;
                        }
                        else
                        {
                            rank_y_has_been_decreased = true;
                            // if not, decrease rank
                            rank_i--;      
                            mult_i = my_result.opt_mult;    
                            // Increase number iterations
                            iter++;
                            // get initial Y with rank rank_i
                            Eigen::JacobiSVD<Eigen::MatrixXd> svd(my_result.opt_Y, Eigen::ComputeThinU | Eigen::ComputeThinV);
                            // Check svd Y                    
                            Eigen::MatrixXd Ured = svd.matrixU().leftCols(rank_i);  
                            Eigen::MatrixXd Vred = svd.matrixV().topRightCorner(rank_i, rank_i);                     
                            Eigen::MatrixXd D = (svd.singularValues().topRows(rank_i)).asDiagonal();
                            Y_i = Eigen::MatrixXd(n_vars, rank_i);
                            // std::cout << "[IN DECRE] Size Y: (" << Y_i.rows() << "," << Y_i.cols() << ")\n";                           
                            Y_i = Ured * D * Vred.transpose();
                            
                            /*
                            std::cout << "Singular values:\n" << svd.singularValues() << std::endl;
                            std::cout << "matrix D:\n" << D << std::endl; 
                            std::cout << "Matrix Vred:\n" << Vred << std::endl;
                            std::cout << "Vred:\n" << svd.matrixV() << std::endl; 
                            */
                            /*
                            std::cout << "[IN DECRE] After Size Vred: (" << svd.matrixV().rows() << "," << svd.matrixV().cols() << ")\n";
                            std::cout << "[IN DECRE] After Size U: (" << Ured.rows() << "," << Ured.cols() << ")\n";
                            std::cout << "[IN DECRE] After Size Vred: (" << Vred.rows() << "," << Vred.cols() << ")\n";
                            std::cout << "[IN DECRE] After Size D: (" << D.rows() << "," << D.cols() << ")\n";
                            std::cout << "[IN DECRE] After Size Y: (" << Y_i.rows() << "," << Y_i.cols() << ")\n";
                            */
                        }                            
                    }
                    
                    else if (!rank_y_has_been_decreased)
                    {
                        // if the next rank gives us a nxn matrix, exit
                        if (rank_i + 1 >= options_.max_rank)
                        {
                            if (options_.verbose)
                                std::cout << "Maximum rank reached!\n" << std::endl; 
                            status = CertStatus::MAX_RANK;
                            result.f_hat = my_result.f_hat;
                            result.opt_mult = my_result.opt_mult;
                            result.opt_Hessian = my_result.opt_Hessian;
                            result.opt_Y = my_result.opt_Y;
                            result.elapsed_init_time = my_result.elapsed_init_time;
                            result.elapsed_iterative_time = my_result.elapsed_iterative_time;  // in microsecs
                            result.elapsed_estimation_time = my_result.elapsed_estimation_time; 
                            result.cond_number_Y = my_result.cond_number_Y; 
                            result.absolute_min_svd_Y = my_result.absolute_min_svd_Y;
                            break;
                        }
                        else if (iter > options_.max_n_iter_staircase)
                        {
                            if (options_.verbose)
                                std::cout << "Maximum number of stairs reached!\n" << std::endl; 
                            status = CertStatus::MAX_ITERS;
                            result.f_hat = my_result.f_hat;
                            result.opt_mult = my_result.opt_mult;
                            result.opt_Hessian = my_result.opt_Hessian;
                            result.opt_Y = my_result.opt_Y;
                            result.elapsed_init_time = my_result.elapsed_init_time;
                            result.elapsed_iterative_time = my_result.elapsed_iterative_time;  // in microsecs
                            result.elapsed_estimation_time = my_result.elapsed_estimation_time; 
                            result.cond_number_Y = my_result.cond_number_Y; 
                            result.absolute_min_svd_Y = my_result.absolute_min_svd_Y;
                            break;                    
                        }
                        
                        // else
                        
                        if (options_.verbose)
                            std::cout << "Increasing rank!\n----------\n" << std::endl; 
                            
                        rank_y_has_been_increased = true;
                        
                        rank_i++;
                        mult_i = my_result.opt_mult;    
                        // std::cout << "Matrix Y:\n" << my_result.opt_Y << std::endl;
                        // Increase number iterations
                        iter++;
                        
                        bool step_descent_done = false;
                        // update Y prev
                        if (!options_.use_original_H_up)
                        {
                            // we compute a descent direction                           
                            // implement the escape in rankY and here call it 
                            // we need the problem that rankY has!!
                            Eigen::MatrixXd Yplus = Eigen::MatrixXd(n_vars, rank_i);
                            step_descent_done = cert_iter_i.escape_saddle_rank_y(xm, costMatrix, constMatrices, my_result.opt_mult, my_result.opt_Y, Yplus);
                            
                            // if the computation succeed
                            // we have step_descent_done = true.  this avoids the next `if`
                            if (step_descent_done) Y_i = Yplus;  // update
                        }
                        
                        // if the previous step has not been successful or 
                        // the user chose the original H
                        if (!step_descent_done)
                        {
                            // Use the original decomposition of H
                            // --- --- --- ---
                            // check dimensions
                            if ((options_.matrixU.rows() == 0) || (options_.matrixU.cols() == 0) 
                                    || (options_.singularValues.rows() == 0) )
                                    {
                                        std::cout << "[STAIR] ERROR: You choose the original H for the next iteration of staircase, but provided matrices are empty!!\n";
                                        status = CertStatus::ERROR_STAIR;
                                        break;
                                    }
                            // Check svd Y                    
                            Eigen::MatrixXd Ured = options_.matrixU.rightCols(rank_i);                     
                            Eigen::MatrixXd D = (options_.singularValues.bottomRows(rank_i)).asDiagonal();
                            Y_i = Eigen::MatrixXd(n_vars, rank_i);
                            Y_i.setZero();
                            // std::cout << "[IN DECRE] Size Y: (" << Y_i.rows() << "," << Y_i.cols() << ")\n";                           
                            Y_i = Ured * D;
                        
                        }
                        // go for the next iteration                           
                    }
                    
                    
                    // if the rank is not deficient nor the cost is below threshold 
                    else
                    {
                        // Unconclusive certifier
                        if (options_.verbose)
                            std::cout << "Certifier was inconclusive!\n" << std::endl; 
                        status = CertStatus::INCONCLUSIVE;
                        result.f_hat = my_result.f_hat;
                        result.opt_mult = my_result.opt_mult;
                        result.opt_Hessian = my_result.opt_Hessian;
                        result.opt_Y = my_result.opt_Y;
                        result.elapsed_init_time = my_result.elapsed_init_time;
                        result.elapsed_iterative_time = my_result.elapsed_iterative_time;  // in microsecs
                        result.elapsed_estimation_time = my_result.elapsed_estimation_time; 
                        result.cond_number_Y = my_result.cond_number_Y; 
                        result.absolute_min_svd_Y = my_result.absolute_min_svd_Y;
                        break;
                    }
                }            
        
                result.total_init_time = total_init_time; 
                result.total_iterative_time = total_iterative_time; 
                result.total_estimation_time = total_estimation_time;
	                       	               
                result.number_stairs = iter;
                result.status_cert = status;

                if (options_.verbose)
                            std::cout << "Number iteration: " << iter << std::endl; 
               return result;

    }; // end of fcn getResult
    
    

template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
void StaircaseCertClass<Matrix, MatrixY, VectorX, VectorM>::printResult(StaircaseCertResult<Matrix, MatrixY, VectorM> & my_result)
{
    // print data

      std::cout << "Data from estimation with staircase algorithm\n--------------------\n";

      std::cout << "######## Times [in microseconds]:\n";
      std::cout << "- Total init: " << my_result.total_init_time << std::endl;

      std::cout << "- Iterative Method: " << my_result.total_iterative_time << std::endl;

      std::cout << "---------------------\n";
      std::cout << "- Total time: " << my_result.total_estimation_time << std::endl << std::endl;

      std::cout << "\n Recovered multipliers:\n" << my_result.opt_mult << std::endl;
      std::cout << "\n Recovered Hessian:\n" << my_result.opt_Hessian << std::endl;    
      std::cout << "\n Number of iterations: " << my_result.number_stairs << std::endl;      
    

};  //end of print fcn

}  // end of essential namespace
