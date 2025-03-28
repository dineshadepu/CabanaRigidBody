#ifndef CabanaRigidBody_TIMEINTEGRATOR_HPP
#define CabanaRigidBody_TIMEINTEGRATOR_HPP

#include <memory>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

namespace CabanaRigidBody
{
  template <class ExecutionSpace>
  class Integrator
  {
    using exec_space = ExecutionSpace;

    double _dt, _half_dt;
  public:
    Integrator ( double dt )
      : _dt (dt)
    {
      _half_dt = 0.5 * dt;

    }
    ~Integrator() {}

    template <class ParticlesType>
    void euler_stage1(ParticlesType& particles){
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

      auto dt = _dt;
      // Update the rigid body properties
      auto rb_euler_stage_1_lambda_func = KOKKOS_LAMBDA( const int i )
        {
          auto m_i = m_cm( i );
          auto m_i_1 = 1. / m_i;

          // Translation motion variables update
          x_cm( i, 0 ) += u_cm( i, 0 ) * dt;
          x_cm( i, 1 ) += u_cm( i, 1 ) * dt;
          x_cm( i, 2 ) += u_cm( i, 2 ) * dt;

          u_cm( i, 0 ) += force_cm( i, 0 ) * m_i_1 * dt;
          u_cm( i, 1 ) += force_cm( i, 1 ) * m_i_1 * dt;
          u_cm( i, 2 ) += force_cm( i, 2 ) * m_i_1 * dt;

          // Rotation motion variables update
          double omega_mat[9] = {0};
          omega_mat[0] = 0.;
          omega_mat[1] = -w_cm( i, 2 );
          omega_mat[2] = w_cm( i, 1 );
          omega_mat[3] = w_cm( i, 2 );
          omega_mat[4] = 0.;
          omega_mat[5] = -w_cm( i, 0 );
          omega_mat[6] = -w_cm( i, 1 );
          omega_mat[7] = w_cm( i, 0 );
          omega_mat[8] = 0.;

          double R_dot[9] = {0};
          R_dot[0] = omega_mat[0] * rot_mat_cm( i, 0 ) + omega_mat[1] * rot_mat_cm( i, 3 ) + omega_mat[2] * rot_mat_cm( i, 6 );
          R_dot[1] = omega_mat[0] * rot_mat_cm( i, 1 ) + omega_mat[1] * rot_mat_cm( i, 4 ) + omega_mat[2] * rot_mat_cm( i, 7 );
          R_dot[2] = omega_mat[0] * rot_mat_cm( i, 2 ) + omega_mat[1] * rot_mat_cm( i, 5 ) + omega_mat[2] * rot_mat_cm( i, 8 );

          R_dot[3] = omega_mat[3] * rot_mat_cm( i, 0 ) + omega_mat[4] * rot_mat_cm( i, 3 ) + omega_mat[5] * rot_mat_cm( i, 6 );
          R_dot[4] = omega_mat[3] * rot_mat_cm( i, 1 ) + omega_mat[4] * rot_mat_cm( i, 4 ) + omega_mat[5] * rot_mat_cm( i, 7 );
          R_dot[5] = omega_mat[3] * rot_mat_cm( i, 2 ) + omega_mat[4] * rot_mat_cm( i, 5 ) + omega_mat[5] * rot_mat_cm( i, 8 );

          R_dot[6] = omega_mat[6] * rot_mat_cm( i, 0 ) + omega_mat[7] * rot_mat_cm( i, 3 ) + omega_mat[8] * rot_mat_cm( i, 6 );
          R_dot[7] = omega_mat[6] * rot_mat_cm( i, 1 ) + omega_mat[7] * rot_mat_cm( i, 4 ) + omega_mat[8] * rot_mat_cm( i, 7 );
          R_dot[8] = omega_mat[6] * rot_mat_cm( i, 2 ) + omega_mat[7] * rot_mat_cm( i, 5 ) + omega_mat[8] * rot_mat_cm( i, 8 );

          for ( std::size_t j = 0; j < 9; ++j )
            {
              rot_mat_cm( i, j ) += R_dot[j] * dt;
            }

          for ( std::size_t j = 0; j < 3; ++j )
            {
              ang_mom_cm( i, j ) += torque_cm( i, j ) * dt;
            }

          // normalize the orientation using Gram Schmidt process
          double R[9] = {0};
          double a1[3] = {0};
          double a2[3] = {0};
          double a3[3] = {0};
          double b1[3] = {0};
          double b2[3] = {0};
          double b3[3] = {0};

          for ( std::size_t j = 0; j < 9; ++j )
            {
              R[j] = rot_mat_cm( i, j );
            }

          a1[0] = R[0];
          a1[1] = R[3];
          a1[2] = R[6];

          a2[0] = R[1];
          a2[1] = R[4];
          a2[2] = R[7];

          a3[0] = R[2];
          a3[1] = R[5];
          a3[2] = R[8];

          // norm of col0
          double na1 = sqrt(a1[0]*a1[0] + a1[1]*a1[1] + a1[2]*a1[2]);
          if (na1 > 1e-12) {
            b1[0] = a1[0] / na1;
            b1[1] = a1[1] / na1;
            b1[2] = a1[2] / na1;
          }
          else {
            b1[0] = a1[0];
            b1[1] = a1[1];
            b1[2] = a1[2];
          }
          double b1_dot_a2 = b1[0] * a2[0] + b1[1] * a2[1] + b1[2] * a2[2];
          b2[0] = a2[0] - b1_dot_a2 * b1[0];
          b2[1] = a2[1] - b1_dot_a2 * b1[1];
          b2[2] = a2[2] - b1_dot_a2 * b1[2];
          double nb2 = sqrt(b2[0]*b2[0] + b2[1]*b2[1] + b2[2]*b2[2]);
          b2[0] = b2[0] / nb2;
          b2[1] = b2[1] / nb2;
          b2[2] = b2[2] / nb2;
          if ( nb2 > 1e-12 ) {
            b2[0] = b2[0] / nb2;
            b2[1] = b2[1] / nb2;
            b2[2] = b2[2] / nb2;
          }

          double b1_dot_a3 = b1[0] * a3[0] + b1[1] * a3[1] + b1[2] * a3[2];
          double b2_dot_a3 = b2[0] * a3[0] + b2[1] * a3[1] + b2[2] * a3[2];
          b3[0] = a3[0] - b1_dot_a3 * b1[0] - b2_dot_a3 * b2[0];
          b3[1] = a3[1] - b1_dot_a3 * b1[1] - b2_dot_a3 * b2[1];
          b3[2] = a3[2] - b1_dot_a3 * b1[2] - b2_dot_a3 * b2[2];
          double nb3 = sqrt(b3[0]*b3[0] + b3[1]*b3[1] + b3[2]*b3[2]);
          if ( nb3 > 1e-12 ) {
            b3[0] = b3[0] / nb3;
            b3[1] = b3[1] / nb3;
            b3[2] = b3[2] / nb3;
          }

          R[0] = b1[0];
          R[3] = b1[1];
          R[6] = b1[2];
          R[1] = b2[0];
          R[4] = b2[1];
          R[7] = b2[2];
          R[2] = b3[0];
          R[5] = b3[1];
          R[8] = b3[2];

          for ( std::size_t j = 0; j < 9; ++j )
            {
              rot_mat_cm( i, j ) = R[j];
            }

          double R_t[9] = {0};
          R_t[0] = R[0];
          R_t[1] = R[3];
          R_t[2] = R[6];

          R_t[3] = R[1];
          R_t[4] = R[4];
          R_t[2] = R[6];

          R_t[6] = R[2];
          R_t[7] = R[5];
          R_t[8] = R[8];

          // copy moi to local matrix
          double tmp_moi_inv[9] = {0};
          for ( std::size_t j = 0; j < 9; ++j )
            {
              tmp_moi_inv[j] = moi_inv_body_mat_cm( i, j );
            }

          double R_moi[9] = {0};
          double new_moi[9] = {0};
          R_moi[0] = R[0] * tmp_moi_inv[0] + R[1] * tmp_moi_inv[3] + R[2] * tmp_moi_inv[6];
          R_moi[1] = R[0] * tmp_moi_inv[1] + R[1] * tmp_moi_inv[4] + R[2] * tmp_moi_inv[7];
          R_moi[2] = R[0] * tmp_moi_inv[2] + R[1] * tmp_moi_inv[5] + R[2] * tmp_moi_inv[8];

          R_moi[3] = R[3] * tmp_moi_inv[0] + R[4] * tmp_moi_inv[3] + R[5] * tmp_moi_inv[6];
          R_moi[4] = R[3] * tmp_moi_inv[1] + R[4] * tmp_moi_inv[4] + R[5] * tmp_moi_inv[7];
          R_moi[5] = R[3] * tmp_moi_inv[2] + R[4] * tmp_moi_inv[5] + R[5] * tmp_moi_inv[8];

          R_moi[6] = R[6] * tmp_moi_inv[0] + R[7] * tmp_moi_inv[3] + R[8] * tmp_moi_inv[6];
          R_moi[7] = R[6] * tmp_moi_inv[1] + R[7] * tmp_moi_inv[4] + R[8] * tmp_moi_inv[7];
          R_moi[8] = R[6] * tmp_moi_inv[2] + R[7] * tmp_moi_inv[5] + R[8] * tmp_moi_inv[8];

          new_moi[0] = R_moi[0] * R_t[0] + R_moi[1] * R_t[3] + R_moi[2] * R_t[6];
          new_moi[1] = R_moi[0] * R_t[1] + R_moi[1] * R_t[4] + R_moi[2] * R_t[7];
          new_moi[2] = R_moi[0] * R_t[2] + R_moi[1] * R_t[5] + R_moi[2] * R_t[8];

          new_moi[3] = R_moi[3] * R_t[0] + R_moi[4] * R_t[3] + R_moi[5] * R_t[6];
          new_moi[4] = R_moi[3] * R_t[1] + R_moi[4] * R_t[4] + R_moi[5] * R_t[7];
          new_moi[5] = R_moi[3] * R_t[2] + R_moi[4] * R_t[5] + R_moi[5] * R_t[8];

          new_moi[6] = R_moi[6] * R_t[0] + R_moi[7] * R_t[3] + R_moi[8] * R_t[6];
          new_moi[7] = R_moi[6] * R_t[1] + R_moi[7] * R_t[4] + R_moi[8] * R_t[7];
          new_moi[8] = R_moi[6] * R_t[2] + R_moi[7] * R_t[5] + R_moi[8] * R_t[8];

          // update moi to particle array
          for ( std::size_t j = 0; j < 9; ++j )
            {
              moi_inv_global_mat_cm( i, j ) = new_moi[j];
            }

          // Angular velocity update
          w_cm( i, 0 ) = new_moi[0] * ang_mom_cm( i, 0 ) + new_moi[1] * ang_mom_cm( i, 1 ) + new_moi[2] * ang_mom_cm( i, 2 );
          w_cm( i, 1 ) = new_moi[3] * ang_mom_cm( i, 0 ) + new_moi[4] * ang_mom_cm( i, 1 ) + new_moi[5] * ang_mom_cm( i, 2 );
          w_cm( i, 2 ) = new_moi[6] * ang_mom_cm( i, 0 ) + new_moi[7] * ang_mom_cm( i, 1 ) + new_moi[8] * ang_mom_cm( i, 2 );

        };
      Kokkos::RangePolicy<ExecutionSpace> policy( 0, rb_limits.size() );
      Kokkos::parallel_for( "CabanaRB:Integrator:RBEulerStage1", policy,
                            rb_euler_stage_1_lambda_func );

      // Update the particle properties
      auto rb_particles_euler_stage_1_lambda_func = KOKKOS_LAMBDA( const int i )
        {
          auto bid_i = body_id_p(i);

          // auto x_body_p = particles.sliceX_body();
          double dx = rot_mat_cm( bid_i, 0 ) * x_body_p( i, 0 ) + rot_mat_cm( bid_i, 1 ) * x_body_p( i, 1 ) + rot_mat_cm( bid_i, 2 ) * x_body_p( i, 2 );
          double dy = rot_mat_cm( bid_i, 3 ) * x_body_p( i, 0 ) + rot_mat_cm( bid_i, 4 ) * x_body_p( i, 1 ) + rot_mat_cm( bid_i, 5 ) * x_body_p( i, 2 );
          double dz = rot_mat_cm( bid_i, 6 ) * x_body_p( i, 0 ) + rot_mat_cm( bid_i, 7 ) * x_body_p( i, 1 ) + rot_mat_cm( bid_i, 8 ) * x_body_p( i, 2 );

          x_p ( i, 0 ) = x_cm ( bid_i, 0 ) + dx;
          x_p ( i, 1 ) = x_cm ( bid_i, 1 ) + dy;
          x_p ( i, 2 ) = x_cm ( bid_i, 2 ) + dz;

          // u_p ( i, 0 ) = u_cm ( bid_i, 0 );
          // u_p ( i, 1 ) = u_cm ( bid_i, 1 );
          // u_p ( i, 2 ) = u_cm ( bid_i, 2 );
        };

      Kokkos::RangePolicy<ExecutionSpace> policy1( 0, x_p.size() );
      Kokkos::parallel_for( "CabanaRB:Integrator:RBParticlesEulerStage1", policy1,
                            rb_particles_euler_stage_1_lambda_func );
    }

    // template <class ParticlesType>
    // void stage1(ParticlesType& p){
    //   auto x_p = particles.slicePosition();
    //   auto u_p = particles.sliceVelocity();
    //   auto au_p = particles.sliceAcceleration();
    //   auto force_p = particles.sliceForce();
    //   auto m_p = particles.sliceMass();
    //   auto rho_p = particles.sliceDensity();
    //   auto p_p = particles.slicePressure();
    //   auto h_p = particles.sliceH();
    //   auto wij_p = particles.sliceWij();
    //   auto arho_p = particles.sliceArho();
    //   auto x_body_p = particles.sliceX_body();
    //   auto u_body_p = particles.sliceU_body();
    //   auto body_id_p = particles.sliceBody_id();

    //   auto rb_limits = sliceRb_limits();
    //   auto m_cm = sliceM_cm();
    //   auto x_cm = sliceX_cm();

    //   auto half_dt = dt * 0.5;
    //   auto rb_gtvf_stage_1_lambda_func = KOKKOS_LAMBDA( const int i )
    //     {
    //       auto mass_i = m_cm( i );
    //       auto mass_i_1 = 1. / mass_i;

    //       rb_velocity( i, 0 ) += rb_force( i, 0 ) * mass_i_1 * rb_lin_acc( i, 0 ) * half_dt;
    //       rb_velocity( i, 1 ) += rb_force( i, 1 ) * mass_i_1 * rb_lin_acc( i, 1 ) * half_dt;
    //       rb_velocity( i, 2 ) += rb_force( i, 2 ) * mass_i_1 * rb_lin_acc( i, 2 ) * half_dt;
    //     };
    //   Kokkos::RangePolicy<ExecutionSpace> policy( 0, rb_velocity.size() );
    //   Kokkos::parallel_for( "CabanaRB:Integrator:RBGTVFStage1", policy,
    //                         rb_gtvf_stage_1_lambda_func );

    //   // update the particles now
    //   auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");
    //   auto aosoa_body_id = Cabana::slice<12>     ( aosoa,    "aosoa_body_id");
    //   auto rb_velocity = Cabana::slice<3>     ( rb,    "rb_velocity");

    //   auto half_dt = dt * 0.5;
    //   auto rb_particles_gtvf_stage_1_lambda_func = KOKKOS_LAMBDA( const int i )
    //     {
    //       auto particle_body_id = aosoa_body_id(i);
    //       aosoa_velocity( i, 0 ) = rb_velocity( particle_body_id, 0 );
    //       aosoa_velocity( i, 1 ) = rb_velocity( particle_body_id, 1 );
    //       aosoa_velocity( i, 2 ) = rb_velocity( particle_body_id, 2 );
    //     };

    //   Kokkos::RangePolicy<ExecutionSpace> policy( index_limits[0], index_limits[1] );
    //   Kokkos::parallel_for( "CabanaRB:Integrator:RBParticlesGTVFStage1", policy,
    //                         rb_particles_gtvf_stage_1_lambda_func );
    // }

    // template <class ParticlesType>
    // void stage2(ParticlesType& p){
    //   auto rb_position = Cabana::slice<2>     ( rb,    "rb_position");
    //   auto rb_velocity = Cabana::slice<3>     ( rb,    "rb_velocity");

    //   auto rb_gtvf_stage_2_lambda_func = KOKKOS_LAMBDA( const int i )
    //     {
    //       rb_position( i, 0 ) += rb_velocity( i, 0 ) * dt;
    //       rb_position( i, 1 ) += rb_velocity( i, 1 ) * dt;
    //       rb_position( i, 2 ) += rb_velocity( i, 2 ) * dt;
    //     };
    //   Kokkos::RangePolicy<ExecutionSpace> policy( 0, rb_velocity.size() );
    //   Kokkos::parallel_for( "CabanaRB:Integrator:RBGTVFStage2", policy,
    //                         rb_gtvf_stage_2_lambda_func );

    //   auto aosoa_position = Cabana::slice<1>     ( aosoa,    "aosoa_velocity");
    //   auto aosoa_body_id = Cabana::slice<12>     ( aosoa,    "aosoa_body_id");
    //   auto aosoa_dx0 = Cabana::slice<13>         ( aosoa,    "aosoa_dx0");

    //   auto rb_position = Cabana::slice<2>     ( rb,    "rb_position");

    //   auto rb_particles_gtvf_stage_2_lambda_func = KOKKOS_LAMBDA( const int i )
    //     {
    //       auto particle_body_id = aosoa_body_id(i);
    //       aosoa_position( i, 0 ) = rb_position( particle_body_id, 0 ) + aosoa_dx0(i, 0);
    //       aosoa_position( i, 1 ) = rb_position( particle_body_id, 1 ) + aosoa_dx0(i, 1);
    //       aosoa_position( i, 2 ) = rb_position( particle_body_id, 2 ) + aosoa_dx0(i, 2);
    //     };

    //   Kokkos::RangePolicy<ExecutionSpace> policy( index_limits[0], index_limits[1] );
    //   Kokkos::parallel_for( "CabanaRB:Integrator:RBParticlesGTVFStage2", policy,
    //                         rb_particles_gtvf_stage_2_lambda_func );
    // }

    // template <class ParticlesType>
    // void stage3(ParticlesType& p){

    //   auto rb_velocity = Cabana::slice<3>     ( rb,    "rb_velocity");
    //   auto rb_force = Cabana::slice<4>        ( rb,       "rb_force");
    //   auto rb_lin_acc = Cabana::slice<6>      ( rb,     "rb_lin_acc");
    //   auto rb_mass = Cabana::slice<11>        ( rb,        "rb_mass");

    //   auto half_dt = dt * 0.5;
    //   auto rb_gtvf_stage_3_lambda_func = KOKKOS_LAMBDA( const int i )
    //     {
    //       auto mass_i = rb_mass( i );
    //       auto mass_i_1 = 1. / mass_i;
    //       rb_lin_acc( i, 0 ) = rb_force( i, 0 ) * mass_i_1;
    //       rb_lin_acc( i, 1 ) = rb_force( i, 1 ) * mass_i_1;
    //       rb_lin_acc( i, 2 ) = rb_force( i, 2 ) * mass_i_1;

    //       rb_velocity( i, 0 ) += rb_lin_acc( i, 0 ) * half_dt;
    //       rb_velocity( i, 1 ) += rb_lin_acc( i, 1 ) * half_dt;
    //       rb_velocity( i, 2 ) += rb_lin_acc( i, 2 ) * half_dt;
    //     };
    //   Kokkos::RangePolicy<ExecutionSpace> policy( 0, rb_velocity.size() );
    //   Kokkos::parallel_for( "CabanaRB:Integrator:RBGTVFStage3", policy,
    //                         rb_gtvf_stage_3_lambda_func );


    //   auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");
    //   auto aosoa_body_id = Cabana::slice<12>     ( aosoa,    "aosoa_body_id");
    //   auto rb_velocity = Cabana::slice<3>     ( rb,    "rb_velocity");

    //   auto half_dt = dt * 0.5;
    //   auto rb_particles_gtvf_stage_3_lambda_func = KOKKOS_LAMBDA( const int i )
    //     {
    //       auto particle_body_id = aosoa_body_id(i);
    //       aosoa_velocity( i, 0 ) = rb_velocity( particle_body_id, 0 );
    //       aosoa_velocity( i, 1 ) = rb_velocity( particle_body_id, 1 );
    //       aosoa_velocity( i, 2 ) = rb_velocity( particle_body_id, 2 );
    //     };

    //   Kokkos::RangePolicy<ExecutionSpace> policy( index_limits[0], index_limits[1] );
    //   Kokkos::parallel_for( "CabanaRB:Integrator:RBParticlesGTVFStage1", policy,
    //                         rb_particles_gtvf_stage_3_lambda_func );
    // }

  };
}

#endif // CabanaRigidBody_TIMEINTEGRATOR_HPP
