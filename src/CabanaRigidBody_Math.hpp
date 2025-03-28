#ifndef CABANARigidBodyMath_HPP
#define CABANARigidBodyMath_HPP

#include <cmath>

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

namespace CabanaRigidBody
{
  namespace Math {
    KOKKOS_INLINE_FUNCTION
    void compute_matrix_inverse(double *A, double *A_inv){
      double det = A[0]*(A[4]*A[8] - A[5]*A[7]) -
        A[1]*(A[3]*A[8] - A[5]*A[6]) +
        A[2]*(A[3]*A[7] - A[4]*A[6]);

      if (det == 0) {
        std::cerr << "Matrix is singular and cannot be inverted." << std::endl;
        return;
      }

      // Calculate the inverse using the adjugate method
      A_inv[0] = (A[4]*A[8] - A[5]*A[7]) / det;
      A_inv[1] = -(A[1]*A[8] - A[2]*A[7]) / det;
      A_inv[2] = (A[1]*A[5] - A[2]*A[4]) / det;
      A_inv[3] = -(A[3]*A[8] - A[5]*A[6]) / det;
      A_inv[4] = (A[0]*A[8] - A[2]*A[6]) / det;
      A_inv[5] = -(A[0]*A[5] - A[2]*A[3]) / det;
      A_inv[6] = (A[3]*A[7] - A[4]*A[6]) / det;
      A_inv[7] = -(A[0]*A[7] - A[1]*A[6]) / det;
      A_inv[8] = (A[0]*A[4] - A[1]*A[3]) / det;
    }
  }
}
#endif
