set(HEADERS
  CabanaRigidBody_Particles.hpp
  CabanaRigidBody_Integrator.hpp
  CabanaRigidBody_Force.hpp
  CabanaRigidBody_Math.hpp
  CabanaRigidBody.hpp
  )

set(SOURCES
  CabanaRigidBody_Particles.cpp
  )

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(cabanarigidbody ${SOURCES})

target_link_libraries(cabanarigidbody
  Cabana::Core
  Cabana::Grid
  nlohmann_json::nlohmann_json
  )

target_include_directories(cabanarigidbody
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

install(TARGETS cabanarigidbody LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
