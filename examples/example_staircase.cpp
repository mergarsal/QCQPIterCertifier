#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <sstream>



// include problem generation
#include "./staircase/StaircaseCert.h"
#include "./staircase/StaircaseCert.cpp"

#include <Eigen/Eigenvalues> 


using namespace std;
using namespace Eigen;
using namespace StaircaseCert; 


int main(int argc, char** argv)
{
        std::srand(std::time(nullptr));
        const int m_constr = 1; 
        const int n_size = 3; 
        
        Eigen::MatrixXd A1 = Eigen::MatrixXd::Identity(n_size, n_size); 
        
        Eigen::MatrixXd P1 = Eigen::MatrixXd::Random(3,2); 
        Eigen::MatrixXd C1 = Eigen::MatrixXd::Random(3,2); 
        Eigen::Matrix3d C = C1*C1.transpose();
        
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, n_size, n_size>> eig_qh(C);
        Eigen::VectorXd q_eig(3); 
        q_eig = eig_qh.eigenvalues(); 
            
        int id_min = 0; 
        double val_min = q_eig(0); 
        for (int i=1;i<3;i++)
        {
                if (q_eig(i) < val_min)
                        {val_min = q_eig(i); id_min = i;}
        }
        Eigen::Vector3d xm; 
        xm = eig_qh.eigenvectors().col(id_min);      
        std::cout << "eigenvalues:" << q_eig << std::endl;
        
        std::cout << "Creating constraint matrices!\n";
        std::vector<Eigen::Matrix3d> constMatrices; 
        constMatrices.push_back(A1);
        // A1.setZero(); 
        //  A1(0,0)=1;
        // A1(2,2)=-1;
        // constMatrices.push_back(A1);
        std::cout << "Creating init multipliers!\n";

        Eigen::VectorXd init_mult(1); 
        init_mult << 1; 
        
        std::cout << "Creating init Hessian!\n";
        Eigen::MatrixXd init_Hessian = C1 + P1; 
        
        std::cout << "Using Euclidean simplification!\n";
        StaircaseCertOptions options;
        options.use_euc_simplification = true;
        options.max_rank = n_size;

        std::cout << "Creating object!\n";
        
        StaircaseCertClass<Eigen::Matrix3d, Eigen::Matrix<double, 3, 2>, Eigen::Vector3d, Eigen::Matrix<double, 1, 1>> cert_iter(options);
        std::cout << "Getting results!\n";
        
        StaircaseCertResult<Eigen::Matrix3d, Eigen::Matrix<double, 3, 2>, Eigen::Matrix<double, 1, 1>> my_result = cert_iter.getResults(xm, C, constMatrices, init_mult, init_Hessian);
        std::cout << "Printing results!\n";
        cert_iter.printResult(my_result); 
        std::cout << "Matrix C:\n" << C << std::endl;
        
        // Using quotient manifold
        std::cout << "Using full geometry!\n";
        
        options.use_euc_simplification = false;

        std::cout << "Creating object!\n";
        
        StaircaseCertClass<Eigen::Matrix3d, Eigen::Matrix<double, 3, 2>, Eigen::Vector3d, Eigen::Matrix<double, 1, 1>> cert_iter_full(options);
        std::cout << "Getting results!\n";
        
        StaircaseCertResult<Eigen::Matrix3d, Eigen::Matrix<double, 3, 2>, Eigen::Matrix<double, 1, 1>> my_result_full = cert_iter_full.getResults(xm, C, constMatrices, init_mult, init_Hessian);
        std::cout << "Printing results!\n";
        cert_iter_full.printResult(my_result_full); 
        
        std::cout << "Matrix C:\n" << C << std::endl;
        
        return 0; 
}
