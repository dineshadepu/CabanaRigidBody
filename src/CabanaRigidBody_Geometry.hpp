#ifndef CABANARigidBodyGeometry_HPP
#define CABANARigidBodyGeometry_HPP

#include <cmath>

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

namespace CabanaRigidBody
{
  namespace Geometry
  {
    std::tuple<std::vector<double>, std::vector<double>> create_2d_block(double length, double height, double spacing) {
      std::vector<double> x_points;
      std::vector<double> y_points;

      for (double x = 0; x <= length; x += spacing) {
        for (double y = 0; y <= height; y += spacing) {
          x_points.push_back(x);
          y_points.push_back(y);
        }
      }

      return {x_points, y_points};
    }


    std::tuple<std::vector<double>, std::vector<double>> create_2d_dam(double length, double height, double spacing,
                                                                       int no_layers) {

      std::vector<double> x_points;
      std::vector<double> y_points;

      /*
        There will be three blocks in a dam. Left, bottom and right
       */

      std::vector<double> x_left;
      std::vector<double> y_left;

      auto left_length = (no_layers + 1) * spacing;
      for (double x = 0; x <= left_length; x += spacing) {
        for (double y = 0; y <= height; y += spacing) {
          x_left.push_back(x);
          y_left.push_back(y);
        }
      }

      std::vector<double> x_right;
      std::vector<double> y_right;

      for (double x = 0; x <= left_length; x += spacing) {
        for (double y = 0; y <= height; y += spacing) {
          x_right.push_back(x);
          y_right.push_back(y);
        }
      }

      std::vector<double> x_bottom;
      std::vector<double> y_bottom;

      auto bottom_length = length + 2. * left_length;
      auto bottom_height = left_length;
      for (double x = 0; x <= bottom_length; x += spacing) {
        for (double y = 0; y <= bottom_height; y += spacing) {
          x_bottom.push_back(x);
          y_bottom.push_back(y);
        }
      }

      return {x_points, y_points};
    }
    // todo: move the left and right to their respective positions


    /// Lenght is in x direction
    /// height is in y direction
    /// depth is in z direction
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> create_3d_block(double length, double height, double depth, double spacing) {
      std::vector<double> x_points;
      std::vector<double> y_points;
      std::vector<double> z_points;

      for (double x = 0; x <= length; x += spacing) {
        for (double y = 0; y <= height; y += spacing) {
          for (double z = 0; z <= depth; z += spacing) {
            x_points.push_back(x);
            y_points.push_back(y);
            z_points.push_back(z);
          }
        }
      }

      return {x_points, y_points, z_points};
    }
  }
}

#endif
