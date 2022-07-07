#include "./rankY/RankYCertManifold.h" 
#include "./rankY/RankYCertProblem.h" 

#include <Eigen/Eigenvalues> 

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace RankYCert{
        /**
                 M.egrad2rgrad = @(Y, eta) eta;
                 M.ehess2rhess = @(Y, egrad, ehess, U) M.proj(Y, ehess);
        **/
    
            VarCert RankYCertManifold::project(const VarCert & P) const
            {
                 VarCert A;                 
                 // 1. Project Euclidean
                 A.mult = P.mult;                  
                 // 2. Project YY^T matrix
                 //  exponential map
                 A.Y = P.Y;
                 return A;
            }
            

                      
            // Most expensive step
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> RankYCertManifold::ProjYYt(
                                const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &t, 
                                const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &Vt) const
            {
                 
                 if (use_euc_simplification_)
                 {
                    return Vt;
                 }
                 // else: use YYt param
                 // Here t is already a mxn matrix
                 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> SS = t.transpose() * t;
                 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> AS = t.transpose() * Vt - Vt.transpose() * t;
                 // Omega = lyapunov_symmetric(SS, AS); 
                 const int n_size = SS.rows();
                 Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_ss(SS);
                 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> M = eig_ss.eigenvectors().transpose() * AS * eig_ss.eigenvectors();
                 
                 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W(t.cols(), t.cols()); 
                 Eigen::VectorXd lambdas(t.cols());
                 // std::cout << "---  INIT ---\n";
                 // std::cout << "Y:\n" << t << std::endl;
                 // std::cout << "Matrix YtY:\n" << SS << std::endl;
                 lambdas = eig_ss.eigenvalues();
                 // std::cout << "EIgenvalues SS:\n" << lambdas << std::endl; 
                 for (int i=0;i<t.cols();i++)
                 {
                    for (int j=i;j<t.cols();j++)
                    {
                        W(i,j) = lambdas(i) + lambdas(j); 
                        W(j,i) = W(i,j);  
                    }
                 } 
                 
                 // std::cout << "Matrix W:\n" << W << std::endl;
                 // std::cout << "Matrix M:\n" << M << std::endl;
                 double tol_y = 1e-10;
                 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> solY = M; 
                 for (int i=0;i<M.rows();i++)
                 {
                        for (int j=0;j<M.cols();j++)
                        {
                                if ((W(i,j) < tol_y) && (-tol_y < W(i,j)))
                                    solY(i,j) = 0.0;
                                else
                                {
                                    solY(i,j) /= W(i,j);
                                    if ((solY(i,j) < tol_y) && (-tol_y < solY(i,j)))
                                        solY(i,j) = 0.0;
                                }
                        }
                 } 
                 
                 
                 // std::cout << "Eigenvectors:\n" << eig_ss.eigenvectors() << std::endl; 
                 
                 // std::cout << "Omega:\n" << eig_ss.eigenvectors() * solY * eig_ss.eigenvectors().transpose() << std::endl;
                 // std::cout << "Eta:\n" << Vt << std::endl; 
                 // std::cout << "t:\n" << t << std::endl;
                 // std::cout << "--- END ---\n";
                 
                 
                 
                 return (Vt - t * eig_ss.eigenvectors() * solY * eig_ss.eigenvectors().transpose());  
            }
        
            
    
            VarCert RankYCertManifold::retract(const VarCert &Y, const VarCert &V) const {

             
              // We use projection-based retraction, as described in "Projection-Like
              // Retractions on Matrix Manifolds" by Absil and Malick
              VarCert projY = project(V+Y);  
              return projY; 
        }
        
        

         // We just call here to ProjEuc & ProjSymm
         VarCert RankYCertManifold::Proj(const VarCert &Y, const VarCert & Vy ) const{
         
   
            VarCert V_tan(Y.Y, Y.mult);
                         
            // for the multipliers
            V_tan.mult = Vy.mult;

            // for the Hessian
            V_tan.Y = ProjYYt(Y.Y, Vy.Y);

            return V_tan;
         }


} // end of essential namespace
