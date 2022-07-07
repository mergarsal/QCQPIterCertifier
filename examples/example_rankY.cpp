#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <sstream>



// include problem generation
#include "./rankY/RankYCert.h"
#include "./rankY/RankYCert.cpp"

#include <Eigen/Eigenvalues> 


using namespace std;
using namespace Eigen;
using namespace RankYCert; 


int main(int argc, char** argv)
{
        std::srand(std::time(nullptr));
        const int m_constr = 1; 
        const int n_size = 3; 
        
        Eigen::MatrixXd A1 = Eigen::MatrixXd::Identity(n_size, n_size); 
        
        
        Eigen::MatrixXd C(3,3);
        C << 0.7431, 0.7536, 0.5123, 0.7536, 0.7681, 0.4963, 0.5123, 0.4963, 0.4939;
    

        std::cout << "Matrix C:\n" << C << std::endl; 
        
        Eigen::MatrixXd C1(3,2); 
        C1 << 0.0893, 0.8605, 0.1391, 0.8675, -0.3129, 0.6310;
       
        Eigen::VectorXd xm(3); 
        xm << 0.7439, -0.6593, -0.1092;
     
        
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
        Eigen::MatrixXd init_Hessian = C1; 
        
        std::cout << "Creating options!\n";
        RankYCertOptions options;

        std::cout << "Creating object!\n";
        RankYCertClass<Eigen::Matrix3d, Eigen::Matrix<double, 3, 2>, Eigen::Vector3d, Eigen::Matrix<double, 1, 1>> cert_iter(options);
        std::cout << "Getting results!\n";
        RankYCertResult<Eigen::Matrix3d, Eigen::Matrix<double, 3, 2>, Eigen::Matrix<double, 1, 1>> my_result = cert_iter.getResults(xm, C, constMatrices, init_mult, init_Hessian);
        std::cout << "Printing results!\n";
        cert_iter.printResult(my_result); 
        
        std::cout << "Matrix C:\n" << C << std::endl;

        return 0; 
}
