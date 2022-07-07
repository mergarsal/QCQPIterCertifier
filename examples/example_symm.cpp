#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <sstream>



// include problem generation
#include "./symmetric/SymmCert.h"
#include "./symmetric/SymmCert.cpp"

#include <Eigen/Eigenvalues> 


using namespace std;
using namespace Eigen;
using namespace SymmCert; 


int main(int argc, char** argv)
{
        std::srand(std::time(nullptr));
        const int m_constr = 1; 
        const int n_size = 3; 
        
        Eigen::MatrixXd A1 = Eigen::MatrixXd::Identity(n_size, n_size); 
        
        Eigen::MatrixXd C1 = Eigen::MatrixXd::Random(3,2); 
        Eigen::Matrix3d C = C1*C1.transpose();
        C << 1.4980 ,   1.3156 ,   0.1926,   1.3156 ,   1.2203,    0.1767,    0.1926   , 0.1767 ,   0.0256; 
        std::cout << "Matrix C1:\n" << C1 << std::endl; 
        std::cout << "Matrix C:\n" << C << std::endl; 
        
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, n_size, n_size>> eig_qh(C);
        Eigen::VectorXd q_eig(3); 
        q_eig = eig_qh.eigenvalues(); 
        std::cout << "REAL Values Q:\n" << q_eig << std::endl; 
            
        int id_min = 0; 
        double val_min = q_eig(0); 
        for (int i=1;i<3;i++)
        {
                if (q_eig(i) < val_min)
                        {val_min = q_eig(i); id_min = i;}
        }
        std::cout << "Min value C: " << val_min << std::endl;
        Eigen::Vector3d xm; 
        xm = -eig_qh.eigenvectors().col(id_min); 
        std::cout << "Eigenvector:\n" << xm << std::endl; 
        std::cout << "Cost optimal: " << xm.transpose() * C * xm << std::endl;
        
        std::cout << "Creating constraint matrices!\n";
        std::vector<Eigen::Matrix3d> constMatrices; 
        constMatrices.push_back(A1);
        A1.setZero(); 
        A1(0,0)=1;
        A1(2,2)=-1;
        constMatrices.push_back(A1);
        std::cout << "Creating init multipliers!\n";

        Eigen::Vector2d init_mult; 
        init_mult << 10, 0; 
        
        std::cout << "Creating init Hessian!\n";
        Eigen::Matrix3d init_Hessian = C; 
        
        std::cout << "Creating options!\n";
        SymmCertOptions options;

        std::cout << "Creating object!\n";
        SymmCertClass<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector2d> cert_iter(options);
        std::cout << "Getting results!\n";
        SymmCertResult<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector2d> my_result = cert_iter.getResults(xm, C, constMatrices, init_mult, init_Hessian);
        std::cout << "Printing results!\n";
        cert_iter.printResult(my_result); 

        return 0; 
}
