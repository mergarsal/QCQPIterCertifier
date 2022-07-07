#pragma once


#include "SymmCertProblem.h"


#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <sstream>


namespace SymmCert{
        template <typename Matrix, typename VectorX, typename VectorM>
        SymmCertProblem<Matrix, VectorX, VectorM>::SymmCertProblem(   const VectorX & xm,
                                                                      const Matrix & costMatrix,
                                                                      const std::vector<Matrix> & constMatrices,
                                                                      const VectorM & init_mult, 
                                                                      const Matrix & init_Hessian) 
        {

               // std::cout << "Inside constructor problem\n"; 
               xm_ = xm; 
               costMatrix_ = costMatrix; 
               constMatrices_ = constMatrices;
               init_mult_ = init_mult; 
               init_Hessian_ = init_Hessian;
               int n = init_mult.rows(); 
               precon_mult_ = 1.0; 
               precon_H_ = 1.0;
               
               // std::cout << "Number constraints: " << n << std::endl;  
               // create trace_constr_
               trace_constr_.setZero(n, n);   
               // std::cout << "Running constraints matrices\n";   
               // std::cout << "Size constraints: " << constMatrices.size() << std::endl;        
               for (int i=0;i<n;i++)
               {
               // std::cout << "Going for row: " << i << std::endl;
                for(int j=i;j<n;j++)
                {
                    // std::cout << "Entering column: " << j << std::endl; 
                    // std::cout << "Matrix i:\n" << constMatrices[i] << std::endl; 
                    // std::cout << "Matrix j:\n" << constMatrices[j] << std::endl; 
                    // std::cout << "Value (i,j) = " << (constMatrices[i] * constMatrices[j]).trace() << std::endl;
                     trace_constr_(i,j) = (constMatrices[i] * constMatrices[j]).trace();  
                     // std::cout << "value assigned!\n";
                     trace_constr_(j,i) = (constMatrices[j] * constMatrices[i]).trace();  
                     // std::cout << "End of loop\n"; 
                } 
               }
             
// std::cout << "End of constructor\n"; 
        }; //end of constructor

        template <typename Matrix, typename VectorX, typename VectorM>
        void SymmCertProblem<Matrix, VectorX, VectorM>::checkConstraints(void)
        {
                int n_const = constMatrices_.size();
                for (int i=0;i<n_const;i++)
                {
                        std::cout << "Constraint #" << i << " with value: " << xm_.dot(constMatrices_[i] * xm_) << std::endl;
                }
        }
        
         
        // apply precon to the var Y
        template <typename Matrix, typename VectorX, typename VectorM>
        VarCert SymmCertProblem<Matrix, VectorX, VectorM>::precondition(const VarCert& X, const VarCert & Xdot) const
        {
            VarCert Xprec = Xdot; 
            Xprec.mult /= precon_mult_; 
            Xprec.H /= precon_H_;
            return tangent_space_projection(X,  Xprec);
        }
        
        
        template <typename Matrix, typename VectorX, typename VectorM>
        double SymmCertProblem<Matrix, VectorX, VectorM>::evaluate_objective(const VarCert &Y) const {
            
            Matrix H = Y.H; 
            VectorM mult = Y.mult;
            
            double f1 = xm_.transpose() * H * H * xm_; 
            double f2 = 0; 
            Matrix Al = H - costMatrix_; 
         
            for (int i=0;i < H.cols(); i++)
            {
                Al += mult(i) * constMatrices_[i]; 
            }
            f2 = (Al*Al.transpose()).trace(); 
            
            // TODO regul
            // double f3 = xm_.transpose() * H * xm_;
            
            return (f1 + f2); // + 0.00001 * f3);

        }
        
         template <typename Matrix, typename VectorX, typename VectorM>
         void SymmCertProblem<Matrix, VectorX, VectorM>::setMatrixPrecon(void)
         {
            // set matrix precon 
            precon_mult_ = init_mult_.rows(); 
            precon_H_ = init_mult_.rows();         
         }
         
         
         
         template <typename Matrix, typename VectorX, typename VectorM>
         double SymmCertProblem<Matrix, VectorX, VectorM>::evaluate_objective(const VarCert &Y, 
                                        ProblemCachedMatrices<Matrix, VectorX, VectorM> & problem_matrices) const {

            Matrix H = Y.H; 
            VectorM mult = Y.mult;
            double f1 = xm_.transpose() * H * H * xm_; 
            double f2 = 0; 
            Matrix Al = H - costMatrix_; 
   
            for (int i=0;i < mult.rows(); i++)
            {
                Al += mult(i) * constMatrices_[i]; 
            }
            problem_matrices.Al = Al;
            f2 = (Al*Al).trace();
            
            // TODO regul
            // double f3 = xm_.transpose() * H * xm_;
             
            return (f1 + f2); // + 0.00001 * f3);
        }


             template <typename Matrix, typename VectorX, typename VectorM>
             VarCert SymmCertProblem<Matrix, VectorX, VectorM>::Euclidean_gradient(const VarCert &Y, 
                                        const ProblemCachedMatrices<Matrix, VectorX, VectorM> & problem_matrices) const
             {
                   Matrix H = Y.H; 
                   VectorM mult = Y.mult;
                   VarCert G(H, mult); 
                   
                   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Hj; 
                  for (int i=0;i < mult.rows(); i++)
                  {     
                      Hj = problem_matrices.Al - mult(i) * constMatrices_[i];
                     
                      G.mult(i) = 2 * mult(i) * trace_constr_(i,i) + 2 * (constMatrices_[i]*Hj).trace();
                    
                  }
                   
                  // TODO last term from f3
                  // std::cout << "Grad mult:\n" << G.mult << std::endl;
                  G.H = 2 * H * xm_*xm_.transpose() + 2 * problem_matrices.Al; // + 0.00001 * 2 * (xm_*xm_.transpose());
                  // std::cout << "Grad H:\n" << G.H << std::endl;
                return G;
                
                
             }
            template <typename Matrix, typename VectorX, typename VectorM>
            VarCert SymmCertProblem<Matrix, VectorX, VectorM>::Riemannian_gradient(const VarCert &Y, const VarCert &nablaF_Y) const
             {
              return tangent_space_projection(Y, nablaF_Y);
             }
             
            template <typename Matrix, typename VectorX, typename VectorM>
            VarCert SymmCertProblem<Matrix, VectorX, VectorM>::Riemannian_gradient(const VarCert &Y, 
                                const ProblemCachedMatrices<Matrix, VectorX, VectorM> & problem_matrices) const {
            
            VarCert Gg = Euclidean_gradient(Y, problem_matrices); 
              return tangent_space_projection(Y, Gg);
            }


       /** Given a matrix Y in the domain D of the SE-Sync optimization problem, and
           * a tangent vector dotY in T_D(Y), the tangent space of the domain of the
           * optimization problem at Y, this function computes and returns Hess
           * F(Y)[dotY], the action of the Riemannian Hessian on dotX */                                               

     template <typename Matrix, typename VectorX, typename VectorM>
     VarCert SymmCertProblem<Matrix, VectorX, VectorM>::Riemannian_Hessian_vector_product(const VarCert &Y,
                                                   const ProblemCachedMatrices<Matrix, VectorX, VectorM> & problem_matrices,
                                                   const VarCert &dotY) const
                                                   {
                                                 
                                                   // Euclidean Hessian-vector product
                                                   VarCert HessRiemannian(Y.H, Y.mult), HessEuc(Y.H, Y.mult);
                                                  
                                                   int n = xm_.rows(); 
                                                   Matrix H = Y.H, VH = dotY.H; 
                                                   VectorM mult = Y.mult, Vmult = dotY.mult;
             
                                     
                                                   HessEuc.H = 2 * Eigen::MatrixXd::Identity(n,n) * VH + 2*xm_*xm_.transpose() * VH;
                                           //  std::cout << "HessH:\n" << HessEuc.H << std::endl;
                                                  
                                                    for (int i=0;i < mult.rows(); i++)
                                                    {
                                                            
                                                              HessEuc.mult(i) = 2 * trace_constr_.row(i)*Vmult + 2 * (constMatrices_[i] * VH).trace();
                                                       
                                                              HessEuc.H += 2 * Vmult(i) * constMatrices_[i];
                                                              // std::cout << "HessH SUM:\n" << 2 * Vmult(i) * constMatrices_[i] << std::endl;
                                                    }
                                                    
                                                   
                                                    // Riemannain Hessian for H
                                                    HessRiemannian.H = domain_.ProjSymm(H, HessEuc.H); 
                                                    HessRiemannian.mult = HessEuc.mult;
                                                   
                                                    return HessRiemannian;
                           }


     
           template <typename Matrix, typename VectorX, typename VectorM>
           VarCert SymmCertProblem<Matrix, VectorX, VectorM>::tangent_space_projection(const VarCert &Y,
                                                            const VarCert &dotY) const 
                                                            { 
                                                            VarCert Yret(Y.H, Y.mult); 
                                                            Yret = domain_.Proj(Y, dotY);
                                                            
                                                            return Yret; }

            template <typename Matrix, typename VectorX, typename VectorM>
            VarCert SymmCertProblem<Matrix, VectorX, VectorM>::retract(const VarCert &Y, const VarCert &dotY) const
            {
                return domain_.retract(Y, dotY);

            }

      

} // end of Essential namespace
