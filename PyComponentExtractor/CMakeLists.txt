cmake_minimum_required(VERSION 2.8)                         
project( ComponentExtractor )                                                               
find_package( OpenCV REQUIRED )     
        
file(GLOB SRC
    "*.h"
    "*.cpp"
)
include_directories($ENV{AC_DIR})
include_directories($ENV{EIGEN_DIR})

add_subdirectory(pybind11)
pybind11_add_module(ComponentExtractor ${SRC})            
target_link_libraries( ComponentExtractor ${OpenCV_LIBS} )  
target_link_libraries( ComponentExtractor ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
