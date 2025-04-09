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
  void compute_effective_force_and_torque_on_rigid_body(ParticlesType& particles ){
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
    auto ang_mom_cm = particles.sliceAng_mom_cm();
    auto moi_body_principal_cm = particles.sliceMoi_body_principal_cm();
    auto w_body_cm = particles.sliceW_body_cm();
    auto w_body_dot_cm = particles.sliceW_body_dot_cm();
    auto quat_cm = particles.sliceQuat_cm();

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

  template <class ParticleType, class NeighListType>
  void compute_force_on_rigid_bodies(ParticleType& particles,
                                     const NeighListType& neigh_list,
                                     const Kokkos::DefaultExecutionSpace& exec_space)
  {

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
    auto rad_p = particles.sliceRadius();

    Cabana::deep_copy( force_p, 0. );

    auto force_full = KOKKOS_LAMBDA( const int i, const int j )
      {
        if ( body_id_p ( i ) != body_id_p( j ) ){
            /*
              Common to all equations in SPH.

              We compute:
              1.the vector passing from j to i
              2. Distance between the points i and j
              3. Distance square between the points i and j
              4. Velocity vector difference between i and j
              5. Kernel value
              6. Derivative of kernel value
            */
            double pos_i[3] = {x_p( i, 0 ),
                               x_p( i, 1 ),
                               x_p( i, 2 )};

            double pos_j[3] = {x_p( j, 0 ),
                               x_p( j, 1 ),
                               x_p( j, 2 )};

            double pos_ij[3] = {x_p( i, 0 ) - x_p( j, 0 ),
                                x_p( i, 1 ) - x_p( j, 1 ),
                                x_p( i, 2 ) - x_p( j, 2 )};

            // squared distance
            double r2ij = pos_ij[0] * pos_ij[0] + pos_ij[1] * pos_ij[1] + pos_ij[2] * pos_ij[2];
            // distance between i and j
            double rij = sqrt(r2ij);

            // const double mass_i = aosoa_mass( i );
            const double mass_j = m_p ( j );

            // Find the overlap amount
            double overlap =  rad_p ( i ) + rad_p ( j ) - rij;

            double a_i = rad_p ( i ) - overlap / 2.;
            double a_j = rad_p ( j ) - overlap / 2.;

            // normal vector passing from j to i
            double nij_x = pos_ij[0] / rij;
            double nij_y = pos_ij[1] / rij;
            double nij_z = pos_ij[2] / rij;

            double vel_i[3] = {0., 0., 0.};
            vel_i[0] = u_p ( i, 0 );
            vel_i[1] = u_p ( i, 1 );
            vel_i[2] = u_p ( i, 2 );

            double vel_j[3] = {0., 0., 0.};
            vel_j[0] = u_p ( j, 0 );
            vel_j[1] = u_p ( i, 1 );
            vel_j[2] = u_p ( i, 2 );

            // Now the relative velocity of particle i w.r.t j at the contact
            // point is
            double vel_ij[3] = {vel_i[0] - vel_j[0],
                                vel_i[1] - vel_j[1],
                                vel_i[2] - vel_j[2]};

            // normal velocity magnitude
            double vij_dot_nij = vel_ij[0] * nij_x + vel_ij[1] * nij_y + vel_ij[2] * nij_z;
            double vn_x = vij_dot_nij * nij_x;
            double vn_y = vij_dot_nij * nij_y;
            double vn_z = vij_dot_nij * nij_z;

            // tangential velocity
            double vt_x = vel_ij[0] - vn_x;
            double vt_y = vel_ij[1] - vn_y;
            double vt_z = vel_ij[2] - vn_z;

            /*
              ====================================
              End: common to all equations in SPH.
              ====================================
            */
            // find the force if the particles are overlapping
            double fn_x = 0.;
            double fn_y = 0.;
            double fn_z = 0.;
            if (overlap > 0.) {
              // normal force
              double fn =  1e5 * overlap;
              fn_x = fn * nij_x;
              fn_y = fn * nij_y;
              fn_z = fn * nij_z;
            }

            // Add force to the particle i due to contact with particle j
            force_p( i, 0 ) += fn_x;
            force_p( i, 1 ) += fn_y;
            force_p( i, 2 ) += fn_z;
          }
      };



    Kokkos::RangePolicy<exec_space> policy(0, u_p.size());


    Cabana::neighbor_parallel_for( policy,
                                   force_full,
                                   neigh_list,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::SerialOpTag(),
                                   "CabanaRigidBody::ForceFull" );
    Kokkos::fence();
  }

}


#endif
