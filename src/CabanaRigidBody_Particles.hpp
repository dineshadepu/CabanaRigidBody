#ifndef CabanaRigidBodyParticles_HPP
#define CabanaRigidBodyParticles_HPP

#include <memory>
#include <filesystem> // or #include <filesystem> for C++17 and up

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include "CabanaRigidBody_Math.hpp"

namespace fs = std::filesystem;

namespace CabanaRigidBody
{
  template <class MemorySpace, int Dimension>
  class Particles
  {
  public:
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;
    static constexpr int dim = Dimension;

    using double_type = Cabana::MemberTypes<double>;
    using int_type = Cabana::MemberTypes<int>;
    using vec_double_type = Cabana::MemberTypes<double[3]>;
    using vec_int_type = Cabana::MemberTypes<int[3]>;
    using matrix_double_type = Cabana::MemberTypes<double[9]>;
    using vec_2_double_type = Cabana::MemberTypes<double[2]>;

    // FIXME: add vector length.
    // FIXME: enable variable aosoa.
    using aosoa_double_type = Cabana::AoSoA<double_type, memory_space, 1>;
    using aosoa_int_type = Cabana::AoSoA<int_type, memory_space, 1>;
    using aosoa_vec_double_type = Cabana::AoSoA<vec_double_type, memory_space, 1>;
    using aosoa_vec_int_type = Cabana::AoSoA<vec_int_type, memory_space, 1>;
    using aosoa_mat_double_type = Cabana::AoSoA<matrix_double_type, memory_space, 1>;
    using aosoa_vec_2_double_type = Cabana::AoSoA<vec_2_double_type, memory_space, 1>;


    std::array<double, dim> mesh_lo;
    std::array<double, dim> mesh_hi;

    // Constructor which initializes particles on regular grid.
    template <class ExecSpace>
    Particles( const ExecSpace& exec_space, std::size_t no_of_particles, std::size_t no_of_rb)
    {
      _no_of_particles = no_of_particles;
      _no_of_rb = no_of_rb;

      resize( _no_of_particles, _no_of_rb );
      createParticles( exec_space );
      // Set dummy values here, reset them in particular examples
      for ( int d = 0; d < dim; d++ )
        {
          mesh_lo[d] = 0.0;
          mesh_hi[d] = 0.0;
        }
    }

    // Constructor which initializes particles on regular grid.
    template <class ExecSpace>
    void createParticles( const ExecSpace& exec_space )
    {
      auto x = slicePosition();

      auto create_particles_func = KOKKOS_LAMBDA( const int i )
        {
          for (int j=0; j < dim; j++){
            // x( i, j ) = DIM * i + j;
          }
        };
      Kokkos::RangePolicy<ExecSpace> policy( 0, x.size() );
      Kokkos::parallel_for( "create_particles_lambda", policy,
                            create_particles_func );
    }

    template <class ExecSpace, class FunctorType>
    void updateParticles( const ExecSpace, const FunctorType init_functor )
    {
      Kokkos::RangePolicy<ExecSpace> policy( 0, _no_of_particles );
      Kokkos::parallel_for(
                           "CabanaPD::Particles::update_particles", policy,
                           KOKKOS_LAMBDA( const int pid ) { init_functor( pid ); } );
    }

    auto slicePosition()
    {
      return Cabana::slice<0>( _x, "positions" );
    }
    auto slicePosition() const
    {
      return Cabana::slice<0>( _x, "positions" );
    }

    auto sliceVelocity()
    {
      return Cabana::slice<0>( _u, "velocities" );
    }
    auto sliceVelocity() const
    {
      return Cabana::slice<0>( _u, "velocities" );
    }

    auto sliceAcceleration()
    {
      return Cabana::slice<0>( _au, "accelerations" );
    }
    auto sliceAcceleration() const
    {
      return Cabana::slice<0>( _au, "accelerations" );
    }

    auto sliceForce()
    {
      return Cabana::slice<0>( _force, "forces" );
    }
    auto sliceForce() const
    {
      return Cabana::slice<0>( _force, "forces" );
    }

    auto sliceMass() {
      return Cabana::slice<0>( _m, "mass" );
    }
    auto sliceMass() const
    {
      return Cabana::slice<0>( _m, "mass" );
    }

    auto sliceDensity() {
      return Cabana::slice<0>( _rho, "density" );
    }
    auto sliceDensity() const
    {
      return Cabana::slice<0>( _rho, "density" );
    }

    auto slicePressure() {
      return Cabana::slice<0>( _p, "pressure" );
    }
    auto slicePressure() const
    {
      return Cabana::slice<0>( _p, "pressure" );
    }

    auto sliceH() {
      return Cabana::slice<0>( _h, "smoothing_length" );
    }
    auto sliceH() const
    {
      return Cabana::slice<0>( _h, "smoothing_length" );
    }

    auto sliceWij() {
      return Cabana::slice<0>( _wij, "wij" );
    }
    auto sliceWij() const
    {
      return Cabana::slice<0>( _wij, "wij" );
    }

    auto sliceArho() {
      return Cabana::slice<0>( _arho, "arho" );
    }
    auto sliceArho() const
    {
      return Cabana::slice<0>( _arho, "arho" );
    }

    auto sliceX_body() {
      return Cabana::slice<0>( _x_body, "x_body" );
    }
    auto sliceX_body() const{
      return Cabana::slice<0>( _x_body, "x_body" );
    }

    auto sliceBody_id() {
      return Cabana::slice<0>( _body_id, "body_id" );
    }
    auto sliceBody_id() const{
      return Cabana::slice<0>( _body_id, "body_id" );
    }

    // Rigid body center of mass properties
    auto sliceRb_limits() {
      return Cabana::slice<0>( _rb_limits, "rb_limits" );
    }
    auto sliceRb_limits() const {
      return Cabana::slice<0>( _rb_limits, "rb_limits" );
    }

    auto sliceM_cm() {
      return Cabana::slice<0>( _m_cm, "m_cm" );
    }
    auto sliceM_cm() const {
      return Cabana::slice<0>( _m_cm, "m_cm" );
    }

    auto sliceX_cm() {
      return Cabana::slice<0>( _x_cm, "x_cm" );
    }
    auto sliceX_cm() const {
      return Cabana::slice<0>( _x_cm, "x_cm" );
    }

    auto sliceU_cm() {
      return Cabana::slice<0>( _u_cm, "u_cm" );
    }
    auto sliceU_cm() const {
      return Cabana::slice<0>( _u_cm, "u_cm" );
    }

    auto sliceW_cm() {
      return Cabana::slice<0>( _w_cm, "w_cm" );
    }
    auto sliceW_cm() const {
      return Cabana::slice<0>( _w_cm, "w_cm" );
    }

    auto sliceRot_mat_cm() {
      return Cabana::slice<0>( _rot_mat_cm, "rot_mat_cm" );
    }
    auto sliceRot_mat_cm() const {
      return Cabana::slice<0>( _rot_mat_cm, "rot_mat_cm" );
    }

    auto sliceMoi_body_mat_cm() {
      return Cabana::slice<0>( _moi_body_mat_cm, "moi_body_mat_cm" );
    }
    auto sliceMoi_body_mat_cm() const {
      return Cabana::slice<0>( _moi_body_mat_cm, "moi_body_mat_cm" );
    }

    auto sliceMoi_global_mat_cm() {
      return Cabana::slice<0>( _moi_global_mat_cm, "moi_global_mat_cm" );
    }
    auto sliceMoi_global_mat_cm() const {
      return Cabana::slice<0>( _moi_global_mat_cm, "moi_global_mat_cm" );
    }

    auto sliceMoi_inv_body_mat_cm() {
      return Cabana::slice<0>( _moi_inv_body_mat_cm, "moi_inv_body_mat_cm" );
    }
    auto sliceMoi_inv_body_mat_cm() const {
      return Cabana::slice<0>( _moi_inv_body_mat_cm, "moi_inv_body_mat_cm" );
    }

    auto sliceMoi_inv_global_mat_cm() {
      return Cabana::slice<0>( _moi_inv_global_mat_cm, "moi_inv_global_mat_cm" );
    }
    auto sliceMoi_inv_global_mat_cm() const {
      return Cabana::slice<0>( _moi_inv_global_mat_cm, "moi_inv_global_mat_cm" );
    }

    auto sliceForce_cm() {
      return Cabana::slice<0>( _force_cm, "force_cm" );
    }
    auto sliceForce_cm() const {
      return Cabana::slice<0>( _force_cm, "force_cm" );
    }

    auto sliceTorque_cm() {
      return Cabana::slice<0>( _torque_cm, "torque_cm" );
    }
    auto sliceTorque_cm() const {
      return Cabana::slice<0>( _torque_cm, "torque_cm" );
    }

    auto sliceAng_mom_cm() {
      return Cabana::slice<0>( _ang_mom_cm, "ang_mom_cm" );
    }
    auto sliceAng_mom() const {
      return Cabana::slice<0>( _ang_mom_cm, "ang_mom_cm" );
    }

    void resize(const std::size_t n, const std::size_t p)
    {
      _no_of_particles = n;
      _x.resize( n );
      _u.resize( n );
      _au.resize( n );
      _force.resize( n );
      _m.resize( n );
      _rho.resize( n );
      _p.resize( n );
      _h.resize( n );
      _wij.resize( n );
      _arho.resize( n );
      _x_body.resize( n );
      _body_id.resize( n );

      _no_of_rb = p;
      _rb_limits.resize( p );
      _m_cm.resize( p );
      _x_cm.resize( p );
      _u_cm.resize( p );
      _w_cm.resize( p );
      _rot_mat_cm.resize( p );
      _moi_body_mat_cm.resize( p );
      _moi_global_mat_cm.resize( p );
      _moi_inv_body_mat_cm.resize( p );
      _moi_inv_global_mat_cm.resize( p );
      _force_cm.resize( p );
      _torque_cm.resize( p );
      _ang_mom_cm.resize( p );
    }

    /// Todo: Change this function to GPU
    void set_rigid_body_limits()
    {
      // auto x_p = particles->slicePosition();
      auto x_p = slicePosition();
      auto u_p = sliceVelocity();
      auto au_p = sliceAcceleration();
      auto force_p = sliceForce();
      auto m_p = sliceMass();
      auto rho_p = sliceDensity();
      auto p_p = slicePressure();
      auto h_p = sliceH();
      auto wij_p = sliceWij();
      auto arho_p = sliceArho();
      auto x_body_p = sliceX_body();
      auto body_id_p = sliceBody_id();

      auto rb_limits = sliceRb_limits();
      auto m_cm = sliceM_cm();
      auto x_cm = sliceX_cm();


      int start = 0;
      int current_value = body_id_p(0);

      int step = 0;
      for (int i = 1; i < body_id_p.size(); ++i) {
        if (body_id_p(i) != current_value) {
          rb_limits(step, 0) = start;
          rb_limits(step, 1) = i;
          start = i;
          current_value = body_id_p(i);
          step += 1;
        }
      }
      // Add the last range
      rb_limits(step, 0) = start;
      rb_limits(step, 1) = body_id_p.size();

      for ( std::size_t i = 0; i < x_cm.size(); ++i )
        {
          std::cout << "rb limits " << i << " are: "
                    << rb_limits (i, 0) << ", "
                    << rb_limits (i, 1) << ", "
                    << std::endl;
        }
    }

    void set_total_mass_and_center_of_mass()
    {

      // auto x_p = particles->slicePosition();
      auto x_p = slicePosition();
      auto u_p = sliceVelocity();
      auto au_p = sliceAcceleration();
      auto force_p = sliceForce();
      auto m_p = sliceMass();
      auto rho_p = sliceDensity();
      auto p_p = slicePressure();
      auto h_p = sliceH();
      auto wij_p = sliceWij();
      auto arho_p = sliceArho();

      auto rb_limits = sliceRb_limits();
      auto m_cm = sliceM_cm();
      auto x_cm = sliceX_cm();

      auto parallel_total_mass_func = KOKKOS_LAMBDA( const int i )
        {
          m_cm(i) = 0.;
          x_cm(i, 0) = 0.;
          x_cm(i, 1) = 0.;
          x_cm(i, 2) = 0.;

          for ( std::size_t j = rb_limits(i, 0); j < rb_limits(i, 1); ++j )
            {
              auto m_j = m_p( j );

              m_cm(i) += m_j;
              x_cm(i, 0) += m_j * x_p(j, 0);
              x_cm(i, 1) += m_j * x_p(j, 1);
              x_cm(i, 2) += m_j * x_p(j, 2);
            }

          x_cm(i, 0) /= m_cm(i);
          x_cm(i, 1) /= m_cm(i);
          x_cm(i, 2) /= m_cm(i);
        };
      Kokkos::RangePolicy<execution_space> policy_tm( 0, m_cm.size());
      Kokkos::parallel_for( "CabanaRB:RBSetup:TotalMass", policy_tm,
                            parallel_total_mass_func );

    }

    void set_moment_of_inertia_and_its_inverse(){
      auto x_p = slicePosition();
      auto u_p = sliceVelocity();
      auto au_p = sliceAcceleration();
      auto force_p = sliceForce();
      auto m_p = sliceMass();
      auto rho_p = sliceDensity();
      auto p_p = slicePressure();
      auto h_p = sliceH();
      auto wij_p = sliceWij();
      auto arho_p = sliceArho();
      auto x_body_p = sliceX_body();
      auto body_id_p = sliceBody_id();

      auto rb_limits = sliceRb_limits();
      auto m_cm = sliceM_cm();
      auto x_cm = sliceX_cm();
      auto u_cm = sliceU_cm();
      auto w_cm = sliceW_cm();
      auto rot_mat_cm = sliceRot_mat_cm();
      auto moi_body_mat_cm = sliceMoi_body_mat_cm();
      auto moi_global_mat_cm = sliceMoi_global_mat_cm();
      auto moi_inv_body_mat_cm = sliceMoi_inv_body_mat_cm();
      auto moi_inv_global_mat_cm = sliceMoi_inv_global_mat_cm();
      auto force_cm = sliceForce_cm();
      auto torque_cm = sliceTorque_cm();
      auto ang_mom_cm = sliceAng_mom_cm();

      auto moment_of_inertia_func = KOKKOS_LAMBDA( const int i )
        {
          double I[9] = {0};
          for ( std::size_t j = rb_limits(i, 0); j < rb_limits(i, 1); ++j )
            {
              auto m_j = m_p( j );

              I[0] += m_j * (pow(x_p( j, 1 ) - x_cm( i, 1 ), 2.) +
                             pow(x_p( j, 2 ) - x_cm( i, 2 ), 2));

              // Iyy
              I[4] += m_j * (pow(x_p( j, 0 ) - x_cm( i, 0 ), 2) +
                             pow(x_p( j, 2 ) - x_cm( i, 2 ), 2));

              // Izz
              I[8] += m_j * (pow(x_p( j, 0 ) - x_cm( i, 0 ), 2) +
                             pow(x_p( j, 1 ) - x_cm( i, 1 ), 2));

              // Ixy
              I[1] -= m_j * (x_p( j, 0 ) - x_cm( i, 0 )) * (x_p( j, 1 ) - x_cm( i, 1 ));

              // Ixz
              I[2] -= m_j * (x_p( j, 0 ) - x_cm( i, 0 )) * (x_p( j, 2 ) - x_cm( i, 2 ));

              // Iyz
              I[5] -= m_j * (x_p( j, 1 ) - x_cm( i, 1 )) * (x_p( j, 2 ) - x_cm( i, 2 ));
            }
          I[3] = I[1];
          I[6] = I[2];
          I[7] = I[5];

          for ( std::size_t k = 0; k < 9; ++k )
            {
              moi_body_mat_cm ( i, k ) = I[k];
            }
          // compute the inverse of the moi tensor
          double I_inv[9] = {0.};
          CabanaRigidBody::Math::compute_matrix_inverse(I, I_inv);
          for ( std::size_t k = 0; k < 9; ++k )
            {
              moi_inv_body_mat_cm ( i, k ) = I_inv[k];
            }

          for ( std::size_t k = 0; k < 9; ++k )
            {
              moi_global_mat_cm ( i, k ) = I[k];
              moi_inv_global_mat_cm ( i, k ) = I_inv[k];
            }
          // Set the rotation matrix i.e., orientation of the body frame axis
          for ( std::size_t k = 0; k < 9; ++k )
            {
              if (k == 0 or k == 4 or k == 8){
                rot_mat_cm ( i, k ) = 1.;
              }
            }
        };
      Kokkos::RangePolicy<execution_space> policy_tm( 0, m_cm.size());
      Kokkos::parallel_for( "CabanaRB:RBSetup:TotalMass", policy_tm,
                            moment_of_inertia_func );
    }

    void set_body_frame_position_vectors() {
      auto x_p = slicePosition();
      auto u_p = sliceVelocity();
      auto au_p = sliceAcceleration();
      auto force_p = sliceForce();
      auto m_p = sliceMass();
      auto rho_p = sliceDensity();
      auto p_p = slicePressure();
      auto h_p = sliceH();
      auto wij_p = sliceWij();
      auto arho_p = sliceArho();
      auto x_body_p = sliceX_body();
      auto body_id_p = sliceBody_id();

      auto rb_limits = sliceRb_limits();
      auto m_cm = sliceM_cm();
      auto x_cm = sliceX_cm();
      auto u_cm = sliceU_cm();
      auto w_cm = sliceW_cm();
      auto rot_mat_cm = sliceRot_mat_cm();
      auto moi_body_mat_cm = sliceMoi_body_mat_cm();
      auto moi_global_mat_cm = sliceMoi_global_mat_cm();
      auto moi_inv_body_mat_cm = sliceMoi_inv_body_mat_cm();
      auto moi_inv_global_mat_cm = sliceMoi_inv_global_mat_cm();
      auto force_cm = sliceForce_cm();
      auto torque_cm = sliceTorque_cm();
      auto ang_mom_cm = sliceAng_mom_cm();

      auto body_frame_position_vectors = KOKKOS_LAMBDA( const int i )
        {
          for ( std::size_t j = rb_limits(i, 0); j < rb_limits(i, 1); ++j )
            {
              x_body_p( j, 0 ) = x_p( j, 0 ) - x_cm( i, 0 );
              x_body_p( j, 1 ) = x_p( j, 1 ) - x_cm( i, 1 );
              x_body_p( j, 2 ) = x_p( j, 2 ) - x_cm( i, 2 );
            }
        };
      Kokkos::RangePolicy<execution_space> policy_tm( 0, m_cm.size());
      Kokkos::parallel_for( "CabanaRB:RBSetup:TotalMass", policy_tm,
                            body_frame_position_vectors );
    }

    void setup_rigid_body_properties(){
      set_rigid_body_limits();
      set_total_mass_and_center_of_mass();
      set_moment_of_inertia_and_its_inverse();
      set_body_frame_position_vectors();
    }

    void set_particle_velocities() {
      auto x_p = slicePosition();
      auto u_p = sliceVelocity();
      auto au_p = sliceAcceleration();
      auto force_p = sliceForce();
      auto m_p = sliceMass();
      auto rho_p = sliceDensity();
      auto p_p = slicePressure();
      auto h_p = sliceH();
      auto wij_p = sliceWij();
      auto arho_p = sliceArho();
      auto x_body_p = sliceX_body();
      auto body_id_p = sliceBody_id();

      auto rb_limits = sliceRb_limits();
      auto m_cm = sliceM_cm();
      auto x_cm = sliceX_cm();
      auto u_cm = sliceU_cm();
      auto w_cm = sliceW_cm();
      auto rot_mat_cm = sliceRot_mat_cm();
      auto moi_body_mat_cm = sliceMoi_body_mat_cm();
      auto moi_global_mat_cm = sliceMoi_global_mat_cm();
      auto moi_inv_body_mat_cm = sliceMoi_inv_body_mat_cm();
      auto moi_inv_global_mat_cm = sliceMoi_inv_global_mat_cm();
      auto force_cm = sliceForce_cm();
      auto torque_cm = sliceTorque_cm();
      auto ang_mom_cm = sliceAng_mom_cm();

      auto particle_velocities_func = KOKKOS_LAMBDA( const int i )
        {
          for ( std::size_t j = rb_limits(i, 0); j < rb_limits(i, 1); ++j )
            {
              auto dx = (rot_mat_cm( i , 0 ) * x_body_p ( j, 0  ) + rot_mat_cm( i , 1 ) * x_body_p ( j, 1 ) +
                         rot_mat_cm( i , 2 ) * x_body_p ( j, 2  ));
              auto dy = (rot_mat_cm( i , 3 ) * x_body_p ( j, 0  ) + rot_mat_cm( i , 4 ) * x_body_p ( j, 1 ) +
                         rot_mat_cm( i , 5 ) * x_body_p ( j, 2  ));
              auto dz = (rot_mat_cm( i , 6 ) * x_body_p ( j, 0  ) + rot_mat_cm( i , 7 ) * x_body_p ( j, 1 ) +
                         rot_mat_cm( i , 8 ) * x_body_p ( j, 2  ));

              auto du = w_cm ( i,  1 ) * dz - w_cm ( i,  2 ) * dy;
              auto dv = w_cm ( i,  2 ) * dx - w_cm ( i,  0 ) * dz;
              auto dw = w_cm ( i,  0 ) * dy - w_cm ( i,  1 ) * dx;

              u_p ( j, 0 ) = u_cm ( i,  1 ) + du;
              u_p ( j, 1 ) = u_cm ( i,  2 ) + dv;
              u_p ( j, 2 ) = u_cm ( i,  0 ) + dw;
            }
        };
      Kokkos::RangePolicy<execution_space> policy_tm( 0, m_cm.size());
      Kokkos::parallel_for( "CabanaRB:RBSetup:TotalMass", policy_tm,
                            particle_velocities_func );
    }

    void set_cm_linear_velocity(double *u_cm_input) {
      auto x_p = slicePosition();
      auto u_p = sliceVelocity();
      auto au_p = sliceAcceleration();
      auto force_p = sliceForce();
      auto m_p = sliceMass();
      auto rho_p = sliceDensity();
      auto p_p = slicePressure();
      auto h_p = sliceH();
      auto wij_p = sliceWij();
      auto arho_p = sliceArho();
      auto x_body_p = sliceX_body();
      auto body_id_p = sliceBody_id();

      auto rb_limits = sliceRb_limits();
      auto m_cm = sliceM_cm();
      auto x_cm = sliceX_cm();
      auto u_cm = sliceU_cm();
      auto w_cm = sliceW_cm();
      auto rot_mat_cm = sliceRot_mat_cm();
      auto moi_body_mat_cm = sliceMoi_body_mat_cm();
      auto moi_global_mat_cm = sliceMoi_global_mat_cm();
      auto moi_inv_body_mat_cm = sliceMoi_inv_body_mat_cm();
      auto moi_inv_global_mat_cm = sliceMoi_inv_global_mat_cm();
      auto force_cm = sliceForce_cm();
      auto torque_cm = sliceTorque_cm();
      auto ang_mom_cm = sliceAng_mom_cm();

      auto particle_velocities_func = KOKKOS_LAMBDA( const int i )
        {
          u_cm( i, 0 ) = u_cm_input[ 3 * i + 0 ];
          u_cm( i, 1 ) = u_cm_input[ 3 * i + 1 ];
          u_cm( i, 2 ) = u_cm_input[ 3 * i + 2 ];

        };
      Kokkos::RangePolicy<execution_space> policy_tm( 0, m_cm.size());
      Kokkos::parallel_for( "CabanaRB:RBSetup:TotalMass", policy_tm,
                            particle_velocities_func );

      set_particle_velocities();
    }

    void set_cm_angular_velocity(double *w_cm_input) {
      auto x_p = slicePosition();
      auto u_p = sliceVelocity();
      auto au_p = sliceAcceleration();
      auto force_p = sliceForce();
      auto m_p = sliceMass();
      auto rho_p = sliceDensity();
      auto p_p = slicePressure();
      auto h_p = sliceH();
      auto wij_p = sliceWij();
      auto arho_p = sliceArho();
      auto x_body_p = sliceX_body();
      auto body_id_p = sliceBody_id();

      auto rb_limits = sliceRb_limits();
      auto m_cm = sliceM_cm();
      auto x_cm = sliceX_cm();
      auto u_cm = sliceU_cm();
      auto w_cm = sliceW_cm();
      auto rot_mat_cm = sliceRot_mat_cm();
      auto moi_body_mat_cm = sliceMoi_body_mat_cm();
      auto moi_global_mat_cm = sliceMoi_global_mat_cm();
      auto moi_inv_body_mat_cm = sliceMoi_inv_body_mat_cm();
      auto moi_inv_global_mat_cm = sliceMoi_inv_global_mat_cm();
      auto force_cm = sliceForce_cm();
      auto torque_cm = sliceTorque_cm();
      auto ang_mom_cm = sliceAng_mom_cm();

      auto particle_velocities_func = KOKKOS_LAMBDA( const int i )
        {
          w_cm( i, 0 ) = w_cm_input[ 3 * i + 0 ];
          w_cm( i, 1 ) = w_cm_input[ 3 * i + 1 ];
          w_cm( i, 2 ) = w_cm_input[ 3 * i + 2 ];

          ang_mom_cm( i, 0 ) = (moi_global_mat_cm ( i, 0 ) * w_cm ( i, 0 ) +
                                moi_global_mat_cm ( i, 1 ) * w_cm ( i, 1 ) +
                                moi_global_mat_cm ( i, 2 ) * w_cm ( i, 2 ));
          ang_mom_cm( i, 1 ) = (moi_global_mat_cm ( i, 3 ) * w_cm ( i, 0 ) +
                                moi_global_mat_cm ( i, 4 ) * w_cm ( i, 1 ) +
                                moi_global_mat_cm ( i, 5 ) * w_cm ( i, 2 ));
          ang_mom_cm( i, 2 ) = (moi_global_mat_cm ( i, 6 ) * w_cm ( i, 0 ) +
                                moi_global_mat_cm ( i, 7 ) * w_cm ( i, 1 ) +
                                moi_global_mat_cm ( i, 8 ) * w_cm ( i, 2 ));
        };
      Kokkos::RangePolicy<execution_space> policy_tm( 0, m_cm.size());
      Kokkos::parallel_for( "CabanaRB:RBSetup:TotalMass", policy_tm,
                            particle_velocities_func );

      set_particle_velocities();
    }

    void output(  const int output_step,
                  const double output_time,
                  const bool use_reference = true )
    {
      // _output_timer.start();

#ifdef Cabana_ENABLE_HDF5
      Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
                                                              h5_config,
                                                              "particles",
                                                              MPI_COMM_WORLD,
                                                              output_step,
                                                              output_time,
                                                              _no_of_particles,
                                                              slicePosition(),
                                                              sliceVelocity(),
                                                              sliceAcceleration(),
                                                              sliceForce(),
                                                              sliceMass(),
                                                              sliceDensity(),
                                                              slicePressure(),
                                                              sliceH(),
                                                              sliceWij(),
                                                              sliceArho());
      // #else
      // #ifdef Cabana_ENABLE_SILO
      //       Cabana::Grid::Experimental::SiloParticleOutput::
      //        writePartialRangeTimeStep(
      //                                  "particles", output_step, output_time,
      //                                  _no_of_particles,
      //                                  slicePosition(),
      //                                  sliceVelocity(),
      //                                  sliceAcceleration(),
      //                                  sliceMass(),
      //                                  sliceDensity(),
      //                                  sliceRadius());
#else
      std::cout << "No particle output enabled.";
      // log( std::cout, "No particle output enabled." );
      // #endif
#endif

      // _output_timer.stop();
    }

    void output_rb_properties(  const int output_step,
                  const double output_time,
                  const bool use_reference = true )
    {
      // _output_timer.start();

#ifdef Cabana_ENABLE_HDF5
      Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
                                                              h5_config,
                                                              "rigid_bodies",
                                                              MPI_COMM_WORLD,
                                                              output_step,
                                                              output_time,
                                                              _no_of_rb,
                                                              sliceX_cm(),
                                                              sliceU_cm(),
                                                              sliceW_cm(),
                                                              sliceForce_cm(),
                                                              sliceTorque_cm(),
                                                              sliceAng_mom_cm(),
                                                              sliceRot_mat_cm(),
                                                              sliceMoi_body_mat_cm(),
                                                              sliceMoi_global_mat_cm(),
                                                              sliceMoi_inv_body_mat_cm(),
                                                              sliceMoi_inv_global_mat_cm(),
                                                              sliceRb_limits(),
                                                              sliceM_cm());
      // #else
      // #ifdef Cabana_ENABLE_SILO
      //       Cabana::Grid::Experimental::SiloParticleOutput::
      //        writePartialRangeTimeStep(
      //                                  "particles", output_step, output_time,
      //                                  _no_of_particles,
      //                                  slicePosition(),
      //                                  sliceVelocity(),
      //                                  sliceAcceleration(),
      //                                  sliceMass(),
      //                                  sliceDensity(),
      //                                  sliceRadius());
#else
      std::cout << "No particle output enabled.";
      // log( std::cout, "No particle output enabled." );
      // #endif
#endif

      // _output_timer.stop();
    }

  private:
    int _no_of_particles;
    aosoa_vec_double_type _x;
    aosoa_vec_double_type _u;
    aosoa_vec_double_type _au;
    aosoa_vec_double_type _force;
    aosoa_double_type _m;
    aosoa_double_type _rho;
    aosoa_double_type _p;
    aosoa_double_type _h;
    aosoa_double_type _wij;
    aosoa_double_type _arho;
    // particle properties corresponding to rigid body dynamics
    aosoa_vec_double_type _x_body; // position vector in body frame
    aosoa_double_type _body_id; // position vector in body frame

    int _no_of_rb;
    aosoa_vec_2_double_type _rb_limits; // indices limit in the main particles to differentiate bodies
    aosoa_double_type _m_cm; // position of the cm of the body
    aosoa_vec_double_type _x_cm; // position of the cm of the body
    aosoa_vec_double_type _u_cm; // Linear velocity of the body
    aosoa_vec_double_type _w_cm; // Angular velocity of the body
    aosoa_mat_double_type _rot_mat_cm; // orientation in the form of a rotation matrix
    aosoa_mat_double_type _moi_body_mat_cm; // orientation in the form of a rotation matrix
    aosoa_mat_double_type _moi_global_mat_cm; // orientation in the form of a rotation matrix
    aosoa_mat_double_type _moi_inv_body_mat_cm; // orientation in the form of a rotation matrix
    aosoa_mat_double_type _moi_inv_global_mat_cm; // orientation in the form of a rotation matrix
    aosoa_vec_double_type _force_cm; // orientation in the form of a rotation matrix
    aosoa_vec_double_type _torque_cm; // orientation in the form of a rotation matrix
    aosoa_vec_double_type _ang_mom_cm; // orientation in the form of a rotation matrix

    // auto x_p = particles.slicePosition();
    // auto u_p = particles.sliceVelocity();
    // auto au_p = particles.sliceAcceleration();
    // auto force_p = particles.sliceForce();
    // auto m_p = particles.sliceMass();
    // auto rho_p = particles.sliceDensity();
    // auto p_p = particles.slicePressure();
    // auto h_p = particles.sliceH();
    // auto wij_p = particles.sliceWij();
    // auto arho_p = particles.sliceArho();
    // auto x_body_p = particles.sliceX_body();
    // auto body_id_p = particles.sliceBody_id();

    // auto rb_limits = particles.sliceRb_limits();
    // auto m_cm = particles.sliceM_cm();
    // auto x_cm = particles.sliceX_cm();
    // auto u_cm = particles.sliceU_cm();
    // auto w_cm = particles.sliceW_cm();
    // auto rot_mat_cm = particles.sliceRot_mat_cm();
    // auto moi_body_mat_cm = sliceMoi_body_mat_cm();
    // auto moi_global_mat_cm = particles.sliceMoi_global_mat_cm();
    // auto moi_inv_body_mat_cm = particles.sliceMoi_inv_body_mat_cm();
    // auto moi_inv_global_mat_cm = particles.sliceMoi_inv_global_mat_cm();
    // auto force_cm = particles.sliceForce_cm();
    // auto torque_cm = particles.sliceTorque_cm();
    // auto ang_mom_cm = particles.sliceAng_mom_cm();

#ifdef Cabana_ENABLE_HDF5
    Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
#endif

    // Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
  };

} // namespace CabanaRigidBody

#endif
