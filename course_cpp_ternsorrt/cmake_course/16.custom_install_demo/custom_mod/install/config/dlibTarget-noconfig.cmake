#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "dlib" for configuration ""
set_property(TARGET dlib APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(dlib PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libdlib.so"
  IMPORTED_SONAME_NOCONFIG "libdlib.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS dlib )
list(APPEND _IMPORT_CHECK_FILES_FOR_dlib "${_IMPORT_PREFIX}/lib/libdlib.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
