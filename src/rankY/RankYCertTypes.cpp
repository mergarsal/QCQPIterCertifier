#include "./rankY/RankYCertTypes.h"

#include <vector>
#include <eigen3/Eigen/Dense>





RankYCert::VarCert RankYCert::sum(const RankYCert::VarCert& X, const RankYCert::VarCert& Y)
{
        VarCert Z(X.Y + X.Y, X.mult + Y.mult); 
        return Z;
};


RankYCert::VarCert operator*(const double b, const RankYCert::VarCert & a) 
{
                return RankYCert::VarCert(b * a.Y, b * a.mult);
} ; 
       
       
RankYCert::VarCert operator*( const RankYCert::VarCert & a, const double b) 
{
                return RankYCert::VarCert(b * a.Y, b * a.mult);
} ; 
       


 
        
