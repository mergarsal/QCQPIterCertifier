#pragma once

#include <Eigen/Dense>

#include "RankYCertTypes.h"

/*Define the namespace*/
namespace RankYCert{

  class RankYCertManifold{
  

      public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
        /* Default constructor */
        RankYCertManifold(void){use_euc_simplification_ = false;};
        
        RankYCertManifold(const bool use_euc_simplification):
                use_euc_simplification_(use_euc_simplification){};
    
        /*Delete each component manifold*/
        ~RankYCertManifold(void){};
        
        /// GEOMETRY
        /** Given a generic matrix A in R^{3 x 3}, this function computes the
        * projection of A onto R (closest point in the Frobenius norm sense).  */
        VarCert project(const VarCert &A) const;
        
        
        /** Given an element Y in M and a tangent vector V in T_Y(M), this function
       * computes the retraction along V at Y using the QR-based retraction
       * specified in eq. (4.8) of Absil et al.'s  "Optimization Algorithms on
       * Matrix Manifolds").
       */
      VarCert retract(const VarCert &Y, const VarCert &V) const;
            
      /* Projections for each manifold */
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ProjYYt(
                                const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &t, 
                                const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &Vt) const;
                               
       /** Given an element Y in M and a matrix V in T_X(R^{p x kn}) (that is, a (p
       * x kn)-dimensional matrix V considered as an element of the tangent space to
       * the *entire* ambient Euclidean space at X), this function computes and
       * returns the projection of V onto T_X(M), the tangent space of M at X (cf.
       * eq. (42) in the SE-Sync tech report).*/
      VarCert Proj(const VarCert &Y, const VarCert &V) const;
    
        
      void setEucSimplification(const bool use_euc_simplification) {use_euc_simplification_ = use_euc_simplification;}; 
        
      
        private: 
            
            bool use_euc_simplification_;
    
      };
} /*end of namespace*/

