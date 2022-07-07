#pragma once


/* RotPrior manifold related */
#include "SymmCertTypes.h"
#include "SymmCertManifold.h"
// #include "SymmCertProblem.hpp"

#include <Eigen/Dense>



namespace SymmCert {

template <typename Matrix, typename VectorX, typename VectorM>
struct ProblemCachedMatrices{
EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
        VarCert NablaF_Y;
        Matrix Al; 
        
        /// DEFAULT CONSTRUCTOR with default values
        ProblemCachedMatrices(){}; 
        

        ProblemCachedMatrices(  const VarCert & NablaF_Y,
                                const Matrix & Al) :
                                NablaF_Y( NablaF_Y), Al(Al){}
};



template <typename Matrix, typename VectorX, typename VectorM>
class SymmCertProblem{
public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
        SymmCertProblem(){};  // Default
        
 
        // Constructor using two vectors of 3XN corresponding features
        SymmCertProblem(const VectorX & xm,
                        const Matrix & costMatrix,
                        const std::vector<Matrix> & constMatrices,
                        const VectorM & init_mult, 
                        const Matrix & init_Hessian); 

                      
        ~SymmCertProblem(){};


           void checkConstraints(void);      
           /// OPTIMIZATION AND GEOMETRY

          /** Given a matrix Y, this function computes and returns F(Y), the value of
           * the objective evaluated at X */
          double evaluate_objective(const VarCert &Y) const;

          double evaluate_objective(const VarCert &Y, ProblemCachedMatrices<Matrix, VectorX, VectorM> & problem_matrices) const;

            /** Given a matrix Y, this function computes and returns nabla F(Y), the
           * *Euclidean* gradient of F at Y. */

          VarCert Euclidean_gradient(const VarCert &Y, const ProblemCachedMatrices<Matrix, VectorX, VectorM> & problem_matrices) const;

          /** Given a matrix Y in the domain D of the SE-Sync optimization problem and
           * the *Euclidean* gradient nabla F(Y) at Y, this function computes and
           * returns the *Riemannian* gradient grad F(Y) of F at Y */

           VarCert Riemannian_gradient(const VarCert &Y, const VarCert &nablaF_Y) const;

           VarCert Riemannian_gradient(const VarCert &Y, const ProblemCachedMatrices<Matrix, VectorX, VectorM> & problem_matrices) const;


          /** Given a matrix Y in the domain D of the SE-Sync optimization problem, the
           * *Euclidean* gradient nablaF_Y of F at Y, and a tangent vector dotY in
           * T_D(Y), the tangent space of the domain of the optimization problem at Y,
           * this function computes and returns Hess F(Y)[dotY], the action of the
           * Riemannian Hessian on dotY */

           VarCert Riemannian_Hessian_vector_product(const VarCert &Y,
                                                     const ProblemCachedMatrices<Matrix, VectorX, VectorM> & problem_matrices,
                                                     const VarCert &dotY) const;

 

           /** Given a matrix Y in the domain D of the SE-Sync optimization problem and a
          tangent vector dotY in T_Y(E), the tangent space of Y considered as a generic
          matrix, this function computes and returns the orthogonal projection of dotY
          onto T_D(Y), the tangent space of the domain D at Y*/
          VarCert tangent_space_projection(const VarCert &Y, const VarCert &dotY) const;

          /** Given a matrix Y in the domain D of the SE-Sync optimization problem and a
           * tangent vector dotY in T_D(Y), this function returns the point Yplus in D
           * obtained by retracting along dotY */
          VarCert retract(const VarCert &Y, const VarCert &dotY) const;

          VarCert precondition(const VarCert& X, const VarCert & Xdot) const; 
          
          void setMatrixPrecon(void);

private:
        VectorX xm_;
        Matrix costMatrix_;
        std::vector<Matrix> constMatrices_;
        VectorM init_mult_; 
        Matrix init_Hessian_; 
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> trace_constr_;

        /** The product manifold of SO(2) X S(2) ~ E(3) that is the domain of our method */
        SymmCertManifold domain_;
        
        double precon_mult_; 
        double precon_H_; 
        

}; // end of problem class

}  // end of  namespace


