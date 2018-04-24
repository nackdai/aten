# Compiler flags...
CPP_COMPILER = g++
C_COMPILER = gcc

# Include paths...
Debug_Include_Path=-I"../src/libaten" -I"../src/libatenscene" -I"../src/libidaten" -I"../src/svgftest" -I"../3rdparty/glm" -I"../3rdparty/imgui" 
Release_Include_Path=-I"../src/libaten" -I"../src/libatenscene" -I"../src/libidaten" -I"../src/svgftest" -I"../3rdparty/glm" -I"../3rdparty/imgui" 

# Library paths...
Debug_Library_Path=-L"x64/Debug" 
Release_Library_Path=-L"x64/Release" 

# Additional libraries...
Debug_Libraries=-Wl,--no-as-needed -Wl,--start-group -laten -latenscene -lidaten -lcudart -fopenmp -lGL -lglfw -lGLEW  -Wl,--end-group
Release_Libraries=-Wl,--no-as-needed -Wl,--start-group -laten -latenscene -lidaten -lcudart -fopenmp -lGL -lglfw -lGLEW  -Wl,--end-group

# Preprocessor definitions...
Debug_Preprocessor_Definitions=-D GCC_BUILD -D __AT_DEBUG__ 
Release_Preprocessor_Definitions=-D GCC_BUILD -D NDEBUG 

# Implictly linked object files...
Debug_Implicitly_Linked_Objects=
Release_Implicitly_Linked_Objects=

# Compiler flags...
Debug_Compiler_Flags=-O0 -g -std=c++11 -fopenmp 
Release_Compiler_Flags=-O2 -std=c++11 -fopenmp 

# Location of the CUDA Toolkit
CUDA_PATH?="/usr/local/cuda"
NVCC:= $(CUDA_PATH)/bin/nvcc -ccbin $(CPP_COMPILER)

Debug_Include_Path+=-I"/usr/local/cuda/include"
Release_Include_Path+=-I"/usr/local/cuda/include"

Debug_Library_Path+=-L"/usr/local/cuda/lib64"
Release_Library_Path+=-L"/usr/local/cuda/lib64"

# Builds all configurations for this project...
.PHONY: build_all_configurations
build_all_configurations: Debug Release 

# Builds the Debug configuration...
.PHONY: Debug
Debug: create_folders x64/Debug/svgftest/src/svgftest/main.o x64/Debug/svgftest/src/svgftest/scenedefs.o 
	g++ x64/Debug/svgftest/src/svgftest/main.o x64/Debug/svgftest/src/svgftest/scenedefs.o  $(Debug_Library_Path) $(Debug_Libraries) -Wl,-rpath,./ -o ../src/svgftest/svgftest.exe

# Compiles file ../src/svgftest/main.cpp for the Debug configuration...
-include x64/Debug/svgftest/src/svgftest/main.d
x64/Debug/svgftest/src/svgftest/main.o: ../src/svgftest/main.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/svgftest/main.cpp $(Debug_Include_Path) -o x64/Debug/svgftest/src/svgftest/main.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/svgftest/main.cpp $(Debug_Include_Path) > x64/Debug/svgftest/src/svgftest/main.d

# Compiles file ../src/svgftest/scenedefs.cpp for the Debug configuration...
-include x64/Debug/svgftest/src/svgftest/scenedefs.d
x64/Debug/svgftest/src/svgftest/scenedefs.o: ../src/svgftest/scenedefs.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/svgftest/scenedefs.cpp $(Debug_Include_Path) -o x64/Debug/svgftest/src/svgftest/scenedefs.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/svgftest/scenedefs.cpp $(Debug_Include_Path) > x64/Debug/svgftest/src/svgftest/scenedefs.d

# Builds the Release configuration...
.PHONY: Release
Release: create_folders x64/Release/svgftest/src/svgftest/main.o x64/Release/svgftest/src/svgftest/scenedefs.o 
	g++ x64/Release/svgftest/src/svgftest/main.o x64/Release/svgftest/src/svgftest/scenedefs.o  $(Release_Library_Path) $(Release_Libraries) -Wl,-rpath,./ -o ../src/svgftest/svgftest.exe

# Compiles file ../src/svgftest/main.cpp for the Release configuration...
-include x64/Release/svgftest/src/svgftest/main.d
x64/Release/svgftest/src/svgftest/main.o: ../src/svgftest/main.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/svgftest/main.cpp $(Release_Include_Path) -o x64/Release/svgftest/src/svgftest/main.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/svgftest/main.cpp $(Release_Include_Path) > x64/Release/svgftest/src/svgftest/main.d

# Compiles file ../src/svgftest/scenedefs.cpp for the Release configuration...
-include x64/Release/svgftest/src/svgftest/scenedefs.d
x64/Release/svgftest/src/svgftest/scenedefs.o: ../src/svgftest/scenedefs.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/svgftest/scenedefs.cpp $(Release_Include_Path) -o x64/Release/svgftest/src/svgftest/scenedefs.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/svgftest/scenedefs.cpp $(Release_Include_Path) > x64/Release/svgftest/src/svgftest/scenedefs.d

# Creates the intermediate and output folders for each configuration...
.PHONY: create_folders
create_folders:
	mkdir -p x64/Debug/svgftest/src/svgftest
	mkdir -p ../src/svgftest
	mkdir -p x64/Release/svgftest/src/svgftest
	mkdir -p ../src/svgftest

# Cleans intermediate and output files (objects, libraries, executables)...
.PHONY: clean
clean:
	rm -f x64/Debug/svgftest/src/svgftest/*.o
	rm -f x64/Debug/svgftest/src/svgftest/*.d
	rm -f ../src/svgftest/svgftest.exe
	rm -f x64/Release/svgftest/src/svgftest/*.o
	rm -f x64/Release/svgftest/src/svgftest/*.d
	rm -f ../src/svgftest/svgftest.exe

