#include "./symmetric/SymmCertManifold.h" 
#include "./symmetric/SymmCertProblem.h" 

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace SymmCert{


            VarCert SymmCertManifold::project(const VarCert & P) const
            {
                 VarCert A;                 
                 // 1. Project Euclidean
                 A.mult = P.mult;                  
                 // 2. Project symmetric matrix
                 A.H = 0.5 * P.H + 0.5 * P.H.transpose(); 
                 return A;
            }
            


            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> SymmCertManifold::ProjSymm(
                                const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &t, 
                                const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &Vt) const
            {
                return  (0.5 * Vt + 0.5 * Vt.transpose()) ; 
            }



            VarCert SymmCertManifold::retract(const VarCert &Y, const VarCert &V) const {

             
              // We use projection-based retraction, as described in "Projection-Like
              // Retractions on Matrix Manifolds" by Absil and Malick
              VarCert projY = project(V+Y);  //project(sum(Y, V));
              return projY; 
        }
        
        

         // We just call here to ProjEuc & ProjSymm
         VarCert SymmCertManifold::Proj(const VarCert &Y, const VarCert & Vy) const{
         
   
            VarCert V_tan(Y.H, Y.mult);
                         
            // for the multipliers
            V_tan.mult = Vy.mult;

            // for the Hessian
            V_tan.H = ProjSymm(Y.H, Vy.H);

            return V_tan;
         }


} // end of essential namespace
