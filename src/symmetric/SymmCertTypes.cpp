#include "./symmetric/SymmCertTypes.h"

#include <vector>
#include <eigen3/Eigen/Dense>





SymmCert::VarCert SymmCert::sum(const SymmCert::VarCert& X, const SymmCert::VarCert& Y)
{
        VarCert Z(X.H + X.H, X.mult + Y.mult); 
        return Z;
};


SymmCert::VarCert operator*(const double b, const SymmCert::VarCert & a) 
{
                return SymmCert::VarCert(b * a.H, b * a.mult);
} ; 
       
       
SymmCert::VarCert operator*( const SymmCert::VarCert & a, const double b) 
{
                return SymmCert::VarCert(b * a.H, b * a.mult);
} ; 
       


 
        
