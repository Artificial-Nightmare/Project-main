cmake_minimum_required(VERSION 3.15)
project(Project_main)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Génère la bibliothèque partagée classLin.dll à partir de main.cpp
add_library(CLASSIFICATION SHARED C:/PA/Project-main/software/main.cpp)

# Lie l'exécutable à la bibliothèque partagée
add_executable(Project_main software/main.cpp)
target_link_libraries(Project_main classLin)

# Place la bibliothèque partagée directement dans le dossier "C:/PA/Project-main"
set_target_properties(classLin PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
