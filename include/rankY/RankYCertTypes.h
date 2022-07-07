#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <sstream>




namespace RankYCert
{

struct VarCert{
EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Y; 
        
        Eigen::Matrix<double, Eigen::Dynamic, 1> mult; 
        
        /// DEFAULT CONSTRUCTOR with default values
        VarCert(){}; 
        
        VarCert(  const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> & Yy,
                                const Eigen::Matrix<double, Eigen::Dynamic, 1> & multh) :
                                Y( Yy ), mult( multh ){};
                        

        double norm(void)
        {
            return ((Y.transpose() * Y).trace() + (mult.dot(mult)));
        
        }
        
        VarCert& operator=(const VarCert& a)
        {
                mult=a.mult;
                Y=a.Y;
                return *this;
        }
    
        VarCert& operator-=(const VarCert& a)
        {
                mult-=a.mult;
                Y-=a.Y;
                return *this;
        }
        
        VarCert& operator+=(const VarCert& a)
        {
                mult+=a.mult;
                Y+=a.Y;
                return *this;
        }
        
         template <typename T>
         VarCert& operator*=(const T a)
        {
                mult*=a;
                Y*=a;
                return *this;
        }

        VarCert operator+(const VarCert& a) const
        {
                return VarCert(a.Y+Y, a.mult+mult);
        }
        
        VarCert operator-() const
        {
                return VarCert(-Y, -mult);
        }
         
        VarCert operator-(const VarCert& a) const
        {
                return VarCert(this->Y-a.Y, this->mult-a.mult);
        }      
        
        template <typename T>
        VarCert operator*(const T a) const
        {
                return VarCert(a*Y, a*mult);
        }                   
       
        
        /* int rows() const 
        {
                // return size H 
                return (Y.rows());        
        }
        */
        
        std::vector<int> rows() const 
        {
                // return size H 
                std::vector<int> t; 
                t.push_back(Y.rows()); 
                t.push_back(Y.cols()); 
                return (t);        
        }
        
        inline int cols() const
        {
                // return size multipliers
                return (mult.rows());
        }     
        
        static VarCert Zero(const int rows, const int cols)
        {
                Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(rows,rows);
                Eigen::VectorXd mult = Eigen::VectorXd::Zero(cols); 
                return VarCert(Y,mult);
        } 
        
        static VarCert Zero(const std::vector<int> & y_size, const int cols)
        {
                Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(y_size[0], y_size[1]);
                Eigen::VectorXd mult = Eigen::VectorXd::Zero(cols); 
                return VarCert(Y,mult);
        }   
                      
};  // end of struct

VarCert sum(const VarCert& X, const VarCert& Y);

       
}  // end of namespace



 
        
