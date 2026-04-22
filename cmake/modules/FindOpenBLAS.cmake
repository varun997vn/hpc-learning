# Fallback FindOpenBLAS module for systems where find_package(OpenBLAS) doesn't work.
# Sets: OpenBLAS_FOUND, OpenBLAS_INCLUDE_DIRS, OpenBLAS_LIBRARIES

find_path(OpenBLAS_INCLUDE_DIR cblas.h
    HINTS
        /usr/include/openblas
        /usr/local/include/openblas
        /opt/OpenBLAS/include
    PATH_SUFFIXES openblas
)

find_library(OpenBLAS_LIBRARY
    NAMES openblas
    HINTS
        /usr/lib
        /usr/local/lib
        /opt/OpenBLAS/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS
    REQUIRED_VARS OpenBLAS_LIBRARY OpenBLAS_INCLUDE_DIR
)

if(OpenBLAS_FOUND)
    set(OpenBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIR})
    set(OpenBLAS_LIBRARIES    ${OpenBLAS_LIBRARY})

    if(NOT TARGET OpenBLAS::OpenBLAS)
        add_library(OpenBLAS::OpenBLAS UNKNOWN IMPORTED)
        set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
            IMPORTED_LOCATION             "${OpenBLAS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(OpenBLAS_INCLUDE_DIR OpenBLAS_LIBRARY)
