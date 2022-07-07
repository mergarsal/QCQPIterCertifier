#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <sstream>




namespace SymmCert
{

struct VarCert{
EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H; 
        
        Eigen::Matrix<double, Eigen::Dynamic, 1> mult; 
        
        /// DEFAULT CONSTRUCTOR with default values
        VarCert(){}; 
        
        VarCert(  const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> & Hh,
                                const Eigen::Matrix<double, Eigen::Dynamic, 1> & multh) :
                                H( Hh ), mult( multh ){};
                        

        VarCert& operator=(const VarCert& a)
        {
                mult=a.mult;
                H=a.H;
                return *this;
        }
    
        VarCert& operator-=(const VarCert& a)
        {
                mult-=a.mult;
                H-=a.H;
                return *this;
        }
        
        VarCert& operator+=(const VarCert& a)
        {
                mult+=a.mult;
                H+=a.H;
                return *this;
        }
        
         template <typename T>
         VarCert& operator*=(const T a)
        {
                mult*=a;
                H*=a;
                return *this;
        }

        VarCert operator+(const VarCert& a) const
        {
                return VarCert(a.H+H, a.mult+mult);
        }
        
        VarCert operator-() const
        {
                return VarCert(-H, -mult);
        }
         
        VarCert operator-(const VarCert& a) const
        {
                return VarCert(this->H-a.H, this->mult-a.mult);
        }      
        
        
        template <typename T>
        VarCert operator*(const T a) const
        {
                return VarCert(a*H, a*mult);
        } 
                          
        VarCert operator*(const VarCert& a) const
        {
                return VarCert(a.H*H, a.mult*mult);
        } 
        
        inline int rows() const 
        {
                // return size H 
                return (H.rows());        
        }
        
        inline int cols() const
        {
                // return size multipliers
                return (mult.rows());
        }     
        
        static VarCert Zero(const int rows, const int cols)
        {
                Eigen::MatrixXd H = Eigen::MatrixXd::Zero(rows,rows);
                Eigen::VectorXd mult = Eigen::VectorXd::Zero(cols); 
                return VarCert(H,mult);
        }   
                      
};  // end of struct

VarCert sum(const VarCert& X, const VarCert& Y);

// VarCert operator*(double b, const VarCert & a);  
       
// VarCert operator*(const VarCert & a, double b) ;
       
}  // end of namespace



 
        
