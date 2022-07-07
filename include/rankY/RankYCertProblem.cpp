#pragma once

#include "RankYCertProblem.h"


#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <Eigen/Cholesky> // to estimate dependent constraints

#define SQRT2 1.4142135623731

namespace RankYCert{
        
        Eigen::VectorXd svec(const Eigen::MatrixXd& X)
        {
            const int s_x = X.rows(); 
            Eigen::VectorXd xx( (s_x + 1) * s_x / 2 ); 
            
            int id_s = 0; 
            for (int i=0; i < s_x; i++)
            {
                xx(id_s) = X(i,i); 
                id_s++;
                for (int j=i+1; j < s_x; j++)
                {                    
                    xx(id_s) = X(i,j) * SQRT2; 
                    id_s++;
                }            
            }
            return xx;       
        
        }

        template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
        RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::RankYCertProblem(const VectorX & xm,
                                                                              const Matrix & costMatrix,
                                                                              const std::vector<Matrix> & constMatrices,
                                                                              const VectorM & init_mult, 
                                                                              const MatrixY & init_Hessian, 
                                                                              const double tol_cost, 
                                                                              const double tol_cost_norm,
                                                                              const double tol_svd, 
                                                                              const bool use_euc_simplification) 
        {

               xm_ = xm; 
               costMatrix_ = costMatrix; 
               constMatrices_ = constMatrices;
               init_mult_ = init_mult; 
               init_Hessian_ = init_Hessian;
               int n = init_mult.rows(); 
               precon_mult_ = 1.0; 
               precon_Y_ = 1.0;
               // create trace_constr_
               trace_constr_.setZero(n, n);              
               for (int i=0;i<n;i++)
               {
                for(int j=i;j<n;j++)
                {
                     trace_constr_(i,j) = (constMatrices[i] * constMatrices[j]).trace();  
                     trace_constr_(j,i) = (constMatrices[j] * constMatrices[i]).trace();  
                } 
               }
               
               // save params 
               threshold_cost_ = tol_cost; 
               threshold_cost_norm_ = tol_cost_norm;
               threshold_min_svd_ = tol_svd; 
               use_euc_simplification_ = use_euc_simplification;
               // setup simplification for domain
               domain_.setEucSimplification(use_euc_simplification);
             

        }; //end of constructor

        template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
        void RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::checkConstraints(void)
        {
                int n_const = constMatrices_.size();
                for (int i=0;i<n_const;i++)
                {
                        std::cout << "Constraint #" << i << " with value: " << xm_.dot(constMatrices_[i] * xm_) << std::endl;
                }
        }
        
        
        template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
        void RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::checkConstraintsIndependence(void)
        {
                int n_const = constMatrices_.size();
                int m_size = constMatrices_[0].rows(); 
                
                Eigen::MatrixXd vec_const((m_size + 1 )* m_size / 2, n_const); 
                
                for (int i=0;i<n_const;i++)
                {                                   
                        vec_const.col(i) = svec(constMatrices_[i]);                              
                }
                // std::cout << "Matrix constraint:\n" << vec_const << std::endl; 
                // std::cout << "Singular values of the constraints:\n"; 
                /*
                const Eigen::JacobiSVD<Eigen::MatrixXd> svd(vec_const.transpose()*vec_const);
                std::cout << svd.singularValues() << std::endl;
                // std::cout << "Singular values size: " << svd.singularValues().rows() << std::endl;
                int n_dep_const = 0; 
                for (int i=n_const-1;i>=0;i--)
                {                                   
                        if (svd.singularValues()[i] < 1e-10)  n_dep_const++; 
                        else break;                          
                }
                */
                // std::cout << "We found " << n_dep_const << " DEPENDENT constraints\n"; 
                // std::cout << "Matrix V:\n" << svd.matrixV() << std::endl;
                
                // Eigen::LDLT<Eigen::MatrixXd> lltOfA(vec_const.transpose()*vec_const); // compute the Cholesky decomposition of A
                // Eigen::MatrixXd L_m = lltOfA.matrixL();
                // Eigen::VectorXd d_m = lltOfA.vectorD();
                // Eigen::Transpositions Pp = lltOfA.transpositionsP(); 
                 
                 
                 
                // std::cout << "diff:\n" << vec_const.transpose()*vec_const - Pp.transpose() * L_m.transpose() * d_m * L_m * Pp<< std::endl;
             
                // std::cout << "Above should be zero\n"; 
                
                // std::cout << "L form LDL:\n"; //  << L_m << std::endl; 
                
                // Eigen::VectorXd dd_m = (Pp.transpose() * L_m * d_m).diagonal(); 
                /*
                int n_dep_const_chol = 0; 
                for (int i=0;i<n_const;i++)
                {                                   
                        if (d_m(i) < 1e-10)  n_dep_const_chol++; 
                        std::cout << "i = " << i << " : " << d_m(i) << std::endl;
                        // else break;                          
                }
                */
                // std::cout << "[CHOL] We found " << n_dep_const_chol << " DEPENDENT constraints\n"; 
                
                
                // QR
                Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qrA(vec_const.transpose()*vec_const); // compute the QR decomposition of A
                Eigen::MatrixXd qr_R = qrA.matrixR().triangularView<Eigen::Upper>();
                auto qr_P = qrA.colsPermutation(); 
                
                int n_dep_const_qr = 0; 
                for (int i=0;i<n_const;i++)
                {                                   
                        if (std::abs(qr_R(i,i)) < 1e-10)  n_dep_const_qr++; 
                        std::cout << "i = " << i << " : " << qr_R(i,i) << std::endl;
                        // else break;                          
                }
                std::cout << "[QR] We found " << n_dep_const_qr << " DEPENDENT constraints\n"; 
                auto ind_perm = qr_P.indices();
                auto ind_dep = ind_perm.bottomRows(n_dep_const_qr); 
                std::cout << "List of dependent constraints:\n" << ind_dep << std::endl;
                
                
        }
        
        
        
        
        template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
        double RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::evaluate_objective(const VarCert &Y) const {
            
          
            MatrixY YY = Y.Y; 
            VectorM mult = Y.mult;
            Matrix H = YY.transpose() * YY; 
            
            
            double f1 = xm_.transpose() * H * H * xm_; 
            double f2 = 0; 
            Matrix Al = H - costMatrix_; 
         
            for (int i=0;i < H.cols(); i++)
            {
                Al += mult(i) * constMatrices_[i]; 
            }
            f2 = (Al*Al.transpose()).trace(); 
            return (f1 + f2);

        }
        
         template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
         double RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::evaluate_objective(const VarCert &Y, ProblemCachedMatrices<Matrix> & problem_matrices) const {

          
            MatrixY YY = Y.Y; 
            VectorM mult = Y.mult;
            Matrix H = YY * YY.transpose();
            
            double f1 = xm_.transpose() * H * H * xm_; 
            double f2 = 0; 
            Matrix Al = H - costMatrix_; 
   
            for (int i=0;i < mult.rows(); i++)
            {
                Al += mult(i) * constMatrices_[i]; 
            }
            problem_matrices.Al = Al;
            problem_matrices.H = H; 
            f2 = (Al*Al).trace(); 
            return (f1 + f2);
        }


             template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
             VarCert RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::Euclidean_gradient(const VarCert &Y, const ProblemCachedMatrices<Matrix> & problem_matrices) const
             {
             
                   MatrixY YY = Y.Y; 
                   VectorM mult = Y.mult;
                   VarCert G(YY, mult); 
                   // std::cout << "Y in gradient:\n" << YY << std::endl;     
                   
                   Matrix Hj; 
                   for (int i=0;i < mult.rows(); i++)
                   {     
                       Hj = problem_matrices.Al - mult(i) * constMatrices_[i];
                     
                       G.mult(i) = 2 * mult(i) * trace_constr_(i,i) + 2 * (constMatrices_[i]*Hj).trace();
                    
                   }
                   
        
                  G.Y = 2 * problem_matrices.H * xm_*xm_.transpose() * YY + 4 * problem_matrices.Al * YY + 2 * xm_*xm_.transpose()*problem_matrices.H*YY;
                  // std::cout << "gradient Y:\n" << G.Y << std::endl; 
                  
                return G;
                
                
            }
            
            template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
            VarCert RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::Riemannian_gradient(const VarCert &Y, const VarCert &nablaF_Y) const
             {
              return tangent_space_projection(Y, nablaF_Y);
             }
             
            
            template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
            VarCert RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::Riemannian_gradient(const VarCert &Y, 
                                                        const ProblemCachedMatrices<Matrix> & problem_matrices) const {
     
              problem_matrices.NablaF_Y = Euclidean_gradient(Y, problem_matrices); 
              return tangent_space_projection(Y, problem_matrices.NablaF_Y);
            }


       /** Given a matrix Y in the domain D of the SE-Sync optimization problem, and
           * a tangent vector dotY in T_D(Y), the tangent space of the domain of the
           * optimization problem at Y, this function computes and returns Hess
           * F(Y)[dotY], the action of the Riemannian Hessian on dotX */                                               
     template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
     VarCert RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::Riemannian_Hessian_vector_product(const VarCert &Y,
                                                   const ProblemCachedMatrices<Matrix> & problem_matrices,
                                                   const VarCert &dotY) const
                                                   {
                                                 
                                                   // Euclidean Hessian-vector product
                                                   VarCert HessRiemannian(Y.Y, Y.mult); 
                                                   VarCert HessEuc = VarCert::Zero(Y.rows(), Y.cols());
                                                    
                                                   int n = Y.mult.rows(); 
                                                   MatrixY YY = Y.Y, VYY = dotY.Y; 
                                                   VectorM mult = Y.mult, Vmult = dotY.mult;
             
                                                   Matrix Cl = problem_matrices.Al - problem_matrices.H;
                                                   
                                            
                                                   int m = xm_.rows();
                                                  
                                                   // std::cout << "\nHessH:\n" << HessEuc.Y << std::endl;
                                                   // std::cout << "VYY:\n" << VYY << std::endl;
                                                    
                                                    for (int i=0;i < n; i++)
                                                    {
                                                            
                                                              HessEuc.mult(i) = 2 * trace_constr_.row(i)*Vmult + 4 * (constMatrices_[i] * YY * VYY.transpose()).trace();
                                                       
                                                              // std::cout << "2 * trace_constr_.row(i)*Vmult: " << 2 * trace_constr_.row(i)*Vmult << std::endl; 
                                                              // std::cout << "4 * (constMatrices_[i] * YY * VYY.transpose()).trace(): " << 4 * (constMatrices_[i] * YY * VYY.transpose()).trace() << std::endl; 
                                                              HessEuc.Y += 4 * Vmult(i) * constMatrices_[i] * YY;
                                                              // std::cout << "HessH SUM:\n" << 4 * Vmult(i) * constMatrices_[i] * YY << std::endl;
                                                    }
                                                   
                                               
                                                    HessEuc.Y += 4 * (YY * VYY.transpose() * YY + VYY*YY.transpose() * YY + problem_matrices.H * VYY);
                                                    // std::cout << "t1: " << 4 * (YY * VYY.transpose() * YY + VYY*YY.transpose() * YY + problem_matrices.H * VYY) << std::endl;
                                               
                                                    HessEuc.Y += 4 * Cl.transpose() * VYY;
                                                    // std::cout << "t2: " << 4 * Cl.transpose() * VYY << std::endl;
                                                    
                                                    HessEuc.Y += 2 * (VYY * YY.transpose() * xm_*xm_.transpose() * YY + YY * VYY.transpose() * xm_*xm_.transpose() * YY + problem_matrices.H * xm_ * xm_.transpose() * VYY); 
                                                  
                                                    // std::cout << "t3: " << 2 * (VYY * YY.transpose() * xm_*xm_.transpose() * YY + YY * VYY.transpose() * xm_*xm_.transpose() * YY + problem_matrices.H * xm_ * xm_.transpose() * VYY) << std::endl; 
                                                    
                                                    HessEuc.Y += 2 * xm_ * xm_.transpose() * (VYY * YY.transpose() * YY + YY * VYY.transpose() * YY + problem_matrices.H * VYY);
                                                    // std::cout << "t4: " << 2 * xm_ * xm_.transpose() * (VYY * YY.transpose() * YY + YY * VYY.transpose() * YY + problem_matrices.H * VYY) << std::endl; 
                                                   
                                                    // Riemannain Hessian for H
                                                    // std::cout << "Y in Hessian:\n" << Y.Y << std::endl; 
                                                    // std::cout << "Mult in Hessian:\n" << Y.mult << std::endl;
                                                    // std::cout << "\nHessian Y:\n" << HessEuc.Y << std::endl;
                                                    // std::cout << "Hessian mult:\n" << HessEuc.mult << std::endl; 
                                                    HessRiemannian.Y = domain_.ProjYYt(YY, HessEuc.Y); 
                                                    HessRiemannian.mult = HessEuc.mult;
                                                   
                                                    return HessRiemannian;
                           }


           template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
           double RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::getVectorEucHessian(const MatrixY &YY,
                                                                  const ProblemCachedMatrices<Matrix> & problem_matrices, 
                                                                  Eigen::VectorXd & v_min) const
           {
                              
                              // We only need the bottom-right (cols+1)x(cols+1)-block
                              // We also returns the minimum eigenvalue

                              Matrix H = YY * YY.transpose();
                              int rows = YY.rows(); 
                              int cols = YY.cols(); 
                              int rank_y = cols+1;
                              
                              v_min.setZero(rank_y,1);
                              double l_min = 10;
                              // Set Yplus = [Y, 0]T
                              Eigen::MatrixXd Yplus(rows, rank_y); 
                              Yplus.setZero(); 
                              
                              Yplus.leftCols(rank_y) = YY;
                              Yplus.rightCols(1) = Eigen::VectorXd::Zero(rows);
                                                    
                              // we use our simplification
                              // NOTE: we do not take into account the 
                              // influence of mult !!
                              Matrix XM = xm_*xm_.transpose(); 
                              
                              Eigen::MatrixXd blockHessian = Eigen::MatrixXd::Zero(rows, rows); 
                              blockHessian = 4 * problem_matrices.Al + 2 * H * XM + 2 * XM * H;
                              
                              
                              Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_ss(0.5 * (blockHessian + blockHessian.transpose()));
                                
                              l_min = eig_ss.eigenvalues()(0);
                              v_min = eig_ss.eigenvectors().col(0);
                                 
                              // std::cout << "Hessian:\n" << blockHessian << std::endl;       
                              // std::cout << "Eigenvalues block Hessian:\n" << eig_ss.eigenvalues() << std::endl;
                                                            
                              return l_min;                      
                                                 
           }                                      
        
          
     
           template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
           VarCert RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::tangent_space_projection(const VarCert &Y,
                                                            const VarCert &dotY) const 
                                                            { 
                                                            VarCert Yret(Y.Y, Y.mult); 
                                                            Yret = domain_.Proj(Y, dotY);
                                                            
                                                            return Yret; }

            template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
            VarCert RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::retract(const VarCert &Y, const VarCert &dotY) const
            {
                return domain_.retract(Y, dotY);

            }
            
            template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
            void RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::setMatrixPrecon(void)
             {
                // set matrix precon 
                precon_mult_ = init_mult_.rows(); 
                precon_Y_ =  init_mult_.rows(); 
             }
             
            // apply precon to the var Y
            template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
            VarCert RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::precondition(const VarCert& X, const VarCert & Xdot) const
            {
                VarCert Xprec = Xdot; 
                Xprec.mult /= precon_mult_; 
                Xprec.Y /= precon_Y_;               
                return tangent_space_projection(X,  Xprec);
            }
        
            
            template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
            bool RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::stopFcnCost(unsigned long i, double f, double df, double tol_f, double tol_fnorm, double tol_df) const
                             {                                
                                // std::cout << "[STOP-FCN] Cost f: " << f << std::endl; 
                                // std::cout << "[STOP-FCN] Normalized cost f: " << f / (init_Hessian_.rows() * init_Hessian_.rows()) << std::endl;
                                double f_norm = f / (init_Hessian_.rows() * init_Hessian_.rows());
                                // std::cout << "[STOP-FCN] Tol cost: " << tol_f << std::endl; 
                                // std::cout << "[STOP-FCN] Iteration number :" << i << std::endl;
                                if ((i > 1) && ((f < tol_f) || (f_norm < tol_fnorm) ) && (df < tol_df))  return true;
                                // else 
                                return false;                             
                             }
 
             template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
             bool RankYCertProblem<Matrix, MatrixY, VectorX, VectorM>::stopFcnRank(unsigned long i, const MatrixY & Y, 
                                            ProblemCachedMatrices<Matrix> & problem_matrices, double tol_svd) const
                             {  
                                // we compute again the svd decomposition of Y 
                                // we contemplate to simplify ProjYYt                                
                                Eigen::MatrixXd yty = Y.transpose() * Y; 
                                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_ss(yty);
                                
                                 Eigen::VectorXd lambdas = eig_ss.eigenvalues();
                                 double cond_number_y = lambdas(0) / lambdas(lambdas.rows()-1);
                                 
                                 // Save !
                                 problem_matrices.abs_min_svd_Y = lambdas(0); 
                                 problem_matrices.cond_number_Y = cond_number_y;
                                 
                                if ((i >= 1) && (!use_euc_simplification_) && (cond_number_y < tol_svd)) return true;
                                
                                // else 
                                return false;   
                                
                             }
} // end of Essential namespace
