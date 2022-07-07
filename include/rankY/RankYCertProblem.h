#pragma once


/* RotPrior manifold related */
#include "RankYCertTypes.h"
#include "RankYCertManifold.h"


#include <Eigen/Dense>



namespace RankYCert {

template <typename Matrix>
struct ProblemCachedMatrices{
EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
        VarCert NablaF_Y;
        Matrix Al; 
        Matrix H; 
        
        double cond_number_Y;
        double abs_min_svd_Y;
        /// DEFAULT CONSTRUCTOR with default values
        ProblemCachedMatrices(){}; 
        
        ProblemCachedMatrices(  const VarCert & NablaF_Y,
                                const Matrix & Al) :
                                NablaF_Y( NablaF_Y), Al(Al){}
};


template <typename Matrix, typename MatrixY, typename VectorX, typename VectorM>
class RankYCertProblem{
public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
        RankYCertProblem(){};  // Default
        
 
        // Constructor using two vectors of 3XN corresponding features
        RankYCertProblem(const VectorX & xm,
                         const Matrix & costMatrix,
                         const std::vector<Matrix> & constMatrices,
                         const VectorM & init_mult, 
                         const MatrixY & init_Hessian, 
                         const double tol_cost = 1e-09, 
                         const double tol_cost_norm = 5e-11, 
                         const double tol_svd = 1e-04, 
                         const bool use_euc_simplification = false) ; 

                      
        ~RankYCertProblem(){};



           void checkConstraints(void); 
        
           /// OPTIMIZATION AND GEOMETRY

          /** Given a matrix Y, this function computes and returns F(Y), the value of
           * the objective evaluated at X */
          double evaluate_objective(const VarCert &Y) const;

          double evaluate_objective(const VarCert &Y, ProblemCachedMatrices<Matrix> & problem_matrices) const;

            /** Given a matrix Y, this function computes and returns nabla F(Y), the
           * *Euclidean* gradient of F at Y. */

          VarCert Euclidean_gradient(const VarCert &Y, const ProblemCachedMatrices<Matrix> & problem_matrices) const;

          /** Given a matrix Y in the domain D of the SE-Sync optimization problem and
           * the *Euclidean* gradient nabla F(Y) at Y, this function computes and
           * returns the *Riemannian* gradient grad F(Y) of F at Y */

           VarCert Riemannian_gradient(const VarCert &Y, const VarCert &nablaF_Y) const;

           VarCert Riemannian_gradient(const VarCert &Y, const ProblemCachedMatrices<Matrix> & problem_matrices) const;


          /** Given a matrix Y in the domain D of the SE-Sync optimization problem, the
           * *Euclidean* gradient nablaF_Y of F at Y, and a tangent vector dotY in
           * T_D(Y), the tangent space of the domain of the optimization problem at Y,
           * this function computes and returns Hess F(Y)[dotY], the action of the
           * Riemannian Hessian on dotY */

           VarCert Riemannian_Hessian_vector_product(const VarCert &Y,
                                                     const ProblemCachedMatrices<Matrix> & problem_matrices,
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

         bool stopFcnCost(unsigned long i, double f, double df, double tol_f, double tol_fnorm, double tol_df) const;
         
         bool stopFcnRank(unsigned long i, const MatrixY & Y, ProblemCachedMatrices<Matrix> & problem_matrices, double told_svd) const;
       
         double get_threshold_cost(void) {return threshold_cost_;};
         double get_threshold_cost_norm(void) {return threshold_cost_norm_;};
         double get_threshold_min_svd(void) {return threshold_min_svd_;};
         
         void setMatrixPrecon(void);
         
         VarCert precondition(const VarCert& X, const VarCert & Xdot) const; 
         
         double getVectorEucHessian(const MatrixY &YY, const ProblemCachedMatrices<Matrix> & problem_matrices, 
                                        Eigen::VectorXd & v_min) const;
                                        
         void checkConstraintsIndependence(void);                               
         
private:
        VectorX xm_;
        Matrix costMatrix_;
        std::vector<Matrix> constMatrices_;
        VectorM init_mult_; 
        MatrixY init_Hessian_; 
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> trace_constr_;

        /** The product manifold of SO(2) X S(2) ~ E(3) that is the domain of our method */
        RankYCertManifold domain_;
        
        double precon_mult_; 
        double precon_Y_; 
        
        
        double threshold_cost_; 
        double threshold_cost_norm_; 
        double threshold_min_svd_;
        
        bool use_euc_simplification_;

}; // end of problem class

}  // end of  namespace


