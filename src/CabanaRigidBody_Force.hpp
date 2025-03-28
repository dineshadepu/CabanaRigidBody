#ifndef CabanaRigidBodyForce_HPP
#define CabanaRigidBodyForce_HPP

#include <memory>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

namespace CabanaRigidBody
{
  template <class ParticlesType, class ExecSpace>
  void compute_effective_force_and_torque_on_rigid_body(ParticlesType& particles, ExecSpace& exec_space ){
    auto x_p = particles.slicePosition();
    auto u_p = particles.sliceVelocity();
    auto au_p = particles.sliceAcceleration();
    auto force_p = particles.sliceForce();
    auto m_p = particles.sliceMass();
    auto rho_p = particles.sliceDensity();
    auto p_p = particles.slicePressure();
    auto h_p = particles.sliceH();
    auto wij_p = particles.sliceWij();
    auto arho_p = particles.sliceArho();
    auto x_body_p = particles.sliceX_body();
    auto body_id_p = particles.sliceBody_id();

    auto rb_limits = particles.sliceRb_limits();
    auto m_cm = particles.sliceM_cm();
    auto x_cm = particles.sliceX_cm();
    auto u_cm = particles.sliceU_cm();
    auto w_cm = particles.sliceW_cm();
    auto rot_mat_cm = particles.sliceRot_mat_cm();
    auto moi_body_mat_cm = particles.sliceMoi_body_mat_cm();
    auto moi_global_mat_cm = particles.sliceMoi_global_mat_cm();
    auto moi_inv_body_mat_cm = particles.sliceMoi_inv_body_mat_cm();
    auto moi_inv_global_mat_cm = particles.sliceMoi_inv_global_mat_cm();
    auto force_cm = particles.sliceForce_cm();
    auto torque_cm = particles.sliceTorque_cm();
    auto ang_mom_cm = particles.sliceTorque_cm();

    auto total_force_torque_func = KOKKOS_LAMBDA( const int i )
      {
        force_cm(i, 0) = 0.;
        force_cm(i, 1) = 0.;
        force_cm(i, 2) = 0.;

        torque_cm(i, 0) = 0.;
        torque_cm(i, 1) = 0.;
        torque_cm(i, 2) = 0.;

        for ( std::size_t j = rb_limits(i, 0); j < rb_limits(i, 1); ++j )
          {
            double fx_j = force_p(j, 0);
            double fy_j = force_p(j, 1);
            double fz_j = force_p(j, 2);

            force_cm(i, 0) += fx_j;
            force_cm(i, 1) += fy_j;
            force_cm(i, 2) += fz_j;

            double dx = x_p( j, 0 ) - x_cm( i, 0 );
            double dy = x_p( j, 1 ) - x_cm( i, 1 );
            double dz = x_p( j, 2 ) - x_cm( i, 2 );

            torque_cm(i, 0) += dy * fz_j - dz * fy_j;
            torque_cm(i, 1) += dz * fx_j - dx * fz_j;
            torque_cm(i, 2) += dx * fy_j - dy * fx_j;
          }
      };

    Kokkos::RangePolicy<ExecSpace> policy( 0, rb_limits.size() );
    Kokkos::parallel_for( "CabanaRB:RB:TotalForceTorque", policy,
                          total_force_torque_func );
  }
}

#endif
