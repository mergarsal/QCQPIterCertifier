project(IterCertAlg-Examples CXX)


# Find Eigen library
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/")
find_package(Eigen3 3.3.3 REQUIRED)
if(EIGEN3_FOUND)
message(STATUS "Found Eigen3 library (version ${EIGEN3_VERSION})")
message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}\n")
else()
message(STATUS "Eigen library not found!")
endif()

# Add the Eigen include directories
include_directories(${EIGEN3_INCLUDE_DIR})

# Fast certifier (symmetric)
add_executable(example_symm ${CMAKE_CURRENT_SOURCE_DIR}/example_symm.cpp)
target_link_libraries(example_symm IterCertAlg)

# Fast certifier (rank)
add_executable(example_rankY ${CMAKE_CURRENT_SOURCE_DIR}/example_rankY.cpp)
target_link_libraries(example_rankY IterCertAlg)

# Fast certifier (staircase)
add_executable(example_staircase ${CMAKE_CURRENT_SOURCE_DIR}/example_staircase.cpp)
target_link_libraries(example_staircase IterCertAlg)
                 
