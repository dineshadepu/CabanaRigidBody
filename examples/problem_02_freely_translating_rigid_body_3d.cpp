#include <fstream>
#include <iostream>
#include <math.h>
#include <cmath>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaRigidBody.hpp>

typedef Kokkos::View<double*>   ViewVectorType;


template<class MemorySpace, class ExecutionSpace>
auto create_particles(double body_length, double body_height, double body_depth, double body_spacing){
  std::vector<double> xf, yf, zf;
  std::tie(xf, yf, zf) = CabanaRigidBody::Geometry::create_3d_block(body_length, body_height, body_depth, body_spacing);
  // std::vector<double> xb, yb, zb;
  // std::tie(xb, yb, zb) = CabanaSPH::Geometry::create_3d_dam(dam_length, dam_height, dam_spacing, dam_layers);

  std::vector<double> x_full = xf;
  // x_full.insert(x_full.end(), xb.begin(), xb.end()); // Append vec2
  std::vector<double> y_full = yf;
  // y_full.insert(y_full.end(), yb.begin(), yb.end()); // Append vec2
  std::vector<double> z_full = zf;
  // z_full.insert(z_full.end(), zb.begin(), zb.end()); // Append vec2

  // create body id to differentiate the rigid bodies
  std::vector<int> body_id_full(x_full.size(), 0);

  auto no_particles = x_full.size();

  ViewVectorType x( "x", no_particles );
  ViewVectorType y( "y", no_particles );
  ViewVectorType z( "z", no_particles );
  ViewVectorType body_id( "body_id", no_particles );
  ViewVectorType::HostMirror host_x = Kokkos::create_mirror_view( x );
  ViewVectorType::HostMirror host_y = Kokkos::create_mirror_view( y );
  ViewVectorType::HostMirror host_z = Kokkos::create_mirror_view( z );
  ViewVectorType::HostMirror host_body_id = Kokkos::create_mirror_view( body_id );

  for ( std::size_t i = 0; i < host_x.size(); ++i )
    {
      host_x ( i ) = x_full [ i ];
      host_y ( i ) = y_full [ i ];
      host_z ( i ) = z_full [ i ];
      host_body_id ( i ) = body_id_full [ i ];
    }

  Kokkos::deep_copy( x, host_x );
  Kokkos::deep_copy( y, host_y );
  Kokkos::deep_copy( z, host_z );
  Kokkos::deep_copy( body_id, host_body_id );

  auto no_bodies = 1;
  auto particles = CabanaRigidBody::Particles<MemorySpace, 3>(ExecutionSpace(), no_particles, no_bodies);

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

  auto init_particles_positions = KOKKOS_LAMBDA( const int pid )
    {
      // Initial conditions: displacements and velocities
      double m_p_i = pow(body_spacing, 2.) * 1000.;

      x_p( pid, 0 ) = x[pid];
      x_p( pid, 1 ) = y[pid];
      x_p( pid, 2 ) = z[pid];

      m_p( pid ) = m_p_i;
      wij_p( pid ) = 0.;
      rho_p( pid ) = 1000.;
      // property where we will approximate the sin
      p_p( pid ) = 0.;

      h_p( pid ) = body_spacing;

      x_body_p ( pid, 0 ) = 0.;
      x_body_p ( pid, 1 ) = 0.;
      x_body_p ( pid, 2 ) = 0.;

      body_id_p ( pid )= body_id[pid];
    };
  Kokkos::RangePolicy<ExecutionSpace> policy(0, no_particles);
  Kokkos::parallel_for("init_particles_positions", policy, init_particles_positions );
  return particles;
}


/*

 */
void Problem02FreelyTranslatingRigidBody3D(double body_length_,
                                           double body_height_,
                                           double body_depth_,
                                           double body_spacing_,
                                           double use_quaternion_)
{
  int comm_rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

  if ( comm_rank == 0 )
    std::cout << "A freely translating rigid body in 2D example\n" << std::endl;

  // ====================================================
  //             Use default Kokkos spaces
  // ====================================================
  using exec_space = Kokkos::DefaultExecutionSpace;
  using memory_space = typename exec_space::memory_space;
  using ExecutionSpace = exec_space;
  using MemorySpace = memory_space;

  // =================================================================
  //                   1. Read inputs and initialize the variables
  // =================================================================
  double body_length = body_length_;
  double body_height = body_height_;
  double body_depth = body_depth_;
  double body_spacing = body_spacing_;
  double hdx = 1.;
  double h = hdx * body_spacing;
  double use_quaternion = use_quaternion_;
  // =================================================================
  //                   2. create the particles
  // =================================================================
  auto particles = create_particles<MemorySpace, ExecutionSpace>(body_length,
                                                                 body_height,
                                                                 body_depth,
                                                                 body_spacing);
  particles.setup_rigid_body_properties();
  CabanaRigidBody::print_rigid_body_properties(particles);
  double lin_vel[3] = {0.0, 0.0, 0.0};
  double ang_vel[3] = {0.0, 0.0, 2. * M_PI};
  // double ang_vel[3] = {0.0, 2. * M_PI, 0.0};
  // double ang_vel[3] = {2. * M_PI, 0.0, 0.0};
  // double ang_vel[3] = {0.0, 0.0, 1.};
  particles.set_cm_linear_velocity(lin_vel);
  particles.set_cm_angular_velocity(ang_vel);
  // CabanaRigidBody::print_particle_properties(particles);
  CabanaRigidBody::print_rigid_body_properties(particles);

  // =================================================================
  // //                   3. create the neighbours
  // // =================================================================
  // // TODO: move all this to function
  // double scale_factor = 3.;
  // double neighborhood_radius = scale_factor * h;
  // double grid_min[3] = { -length, -neighborhood_radius -  spacing, - neighborhood_radius -  spacing};
  // double grid_max[3] = { length + 2. * neighborhood_radius , neighborhood_radius + spacing, neighborhood_radius + spacing};

  // double cell_ratio = 1.0;
  // using ListAlgorithm = Cabana::FullNeighborTag;
  // using ListType = Cabana::VerletList<memory_space, ListAlgorithm,
  //                                     Cabana::VerletLayoutCSR,
  //                                     Cabana::TeamOpTag>;

  // auto positions = particles->slicePosition();
  // auto verlet_list = std::make_shared<ListType>( positions, 0, positions.size(), neighborhood_radius,
  //                       cell_ratio, grid_min, grid_max );


  // ====================================================
  //                   4. The time loop
  // ====================================================
  auto dt = 1e-4;
  auto final_time = 1.;
  // auto final_time = 2. * M_PI;
  // auto final_time = 100. * dt;
  auto time = 0.;
  int num_steps = final_time / dt;
  int output_frequency = 100;

  using integrator_type = CabanaRigidBody::Integrator<exec_space>;
  integrator_type integrator ( dt );

  // Main timestep loop.
  for ( int step = 0; step <= num_steps; step++ )
    {
      if (use_quaternion == 1.) {
        integrator.euler_stage1_quaternion( particles );
      }
      else {
        integrator.euler_stage1( particles );
      }
      // integrator.euler_stage1_quaternion_body_frame_eqs( particles );

      // integrator->stage2( *particles );

      // integrator->stage3( *particles );


      // ====================================================
      //                   5. output
      // ====================================================
      if ( step % output_frequency == 0 )
        {
          // std::cout << "We are at " << step << " " << "/ " << num_steps << " ";
          // std::cout << std::endl;
          particles.output( step / output_frequency, time );
          particles.output_rb_properties( step / output_frequency, time );
        }
      time += dt;
      // CabanaRigidBody::print_rigid_body_properties(particles);
      CabanaRigidBody::printProgressBar(float(step) / num_steps);
    }
  CabanaRigidBody::print_rigid_body_properties(particles);
}


int main( int argc, char* argv[] )
{
  // Initialize MPI+Kokkos.
  MPI_Init( &argc, &argv );
  Kokkos::initialize( argc, argv );

  // check inputs and write usage
  if ( argc < 4 )
    {
      std::cerr << "Usage: ./Problem02FreelyTranslatingRigidBody3D body-length "
        "body-height body-depth  body-spacing use-quaternion \n";
      std::cerr << "\nwhere body-length       length of the body block"
        "\n";
      std::cerr
        << "      body-height       height of the body block\n";
      std::cerr
        << "      body-depth       depth of the body block\n";
      std::cerr
        << "      body-spacing       spacing of the body particles\n";
      std::cerr
        << "      use-quaternion       Use quaternions for rotation\n";
      std::cerr << "\nfor example: ./Problem02FreelyTranslatingRigidBody3D 1. 1. 1. 0.1 1.\n";
      Kokkos::finalize();
      MPI_Finalize();
      return 0;
    }

  // get the command line options
  double body_length = std::atof( argv[1] );
  double body_height = std::atof( argv[2] );
  double body_depth = std::atof( argv[3] );
  double body_spacing = std::atof( argv[4] );
  double use_quaternion = std::atof( argv[5] );
  // run the problem
  Problem02FreelyTranslatingRigidBody3D(
                      body_length,
                      body_height,
                      body_depth,
                      body_spacing,
                      use_quaternion);

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
