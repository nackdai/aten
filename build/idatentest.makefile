# Compiler flags...
CPP_COMPILER = g++
C_COMPILER = gcc

# Include paths...
Debug_Include_Path=-I"../src/libaten" -I"../src/libatenscene" -I"../src/libidaten" -I"../src/idatentest" -I"../3rdparty/glm" -I"../3rdparty/imgui" 
Release_Include_Path=-I"../src/libaten" -I"../src/libatenscene" -I"../src/libidaten" -I"../src/idatentest" -I"../3rdparty/glm" -I"../3rdparty/imgui" 

# Library paths...
Debug_Library_Path=-L"x64/Debug" 
Release_Library_Path=-L"x64/Release" 

# Additional libraries...
Debug_Libraries=-Wl,--no-as-needed -Wl,--start-group -laten -latenscene -lidaten -lcudart -lcudadevrt -fopenmp -lGL -lglfw -lGLEW  -Wl,--end-group
Release_Libraries=-Wl,--no-as-needed -Wl,--start-group -laten -latenscene -lidaten -lcudart -lcudadevrt -fopenmp -lGL -lglfw -lGLEW  -Wl,--end-group

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
Debug: create_folders x64/Debug/idatentest/src/common/scenedefs.o x64/Debug/idatentest/src/idatentest/main.o 
	g++ x64/Debug/idatentest/src/common/scenedefs.o x64/Debug/idatentest/src/idatentest/main.o  $(Debug_Library_Path) $(Debug_Libraries) -Wl,-rpath,./ -o ../src/idatentest/idatentest.exe

# Compiles file ../src/common/scenedefs.cpp for the Debug configuration...
-include x64/Debug/idatentest/src/common/scenedefs.d
x64/Debug/idatentest/src/common/scenedefs.o: ../src/common/scenedefs.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/common/scenedefs.cpp $(Debug_Include_Path) -o x64/Debug/idatentest/src/common/scenedefs.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/common/scenedefs.cpp $(Debug_Include_Path) > x64/Debug/idatentest/src/common/scenedefs.d

# Compiles file ../src/idatentest/main.cpp for the Debug configuration...
-include x64/Debug/idatentest/src/idatentest/main.d
x64/Debug/idatentest/src/idatentest/main.o: ../src/idatentest/main.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/idatentest/main.cpp $(Debug_Include_Path) -o x64/Debug/idatentest/src/idatentest/main.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/idatentest/main.cpp $(Debug_Include_Path) > x64/Debug/idatentest/src/idatentest/main.d

# Builds the Release configuration...
.PHONY: Release
Release: create_folders x64/Release/idatentest/src/common/scenedefs.o x64/Release/idatentest/src/idatentest/main.o 
	g++ x64/Release/idatentest/src/common/scenedefs.o x64/Release/idatentest/src/idatentest/main.o  $(Release_Library_Path) $(Release_Libraries) -Wl,-rpath,./ -o ../src/idatentest/idatentest.exe

# Compiles file ../src/common/scenedefs.cpp for the Release configuration...
-include x64/Release/idatentest/src/common/scenedefs.d
x64/Release/idatentest/src/common/scenedefs.o: ../src/common/scenedefs.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/common/scenedefs.cpp $(Release_Include_Path) -o x64/Release/idatentest/src/common/scenedefs.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/common/scenedefs.cpp $(Release_Include_Path) > x64/Release/idatentest/src/common/scenedefs.d

# Compiles file ../src/idatentest/main.cpp for the Release configuration...
-include x64/Release/idatentest/src/idatentest/main.d
x64/Release/idatentest/src/idatentest/main.o: ../src/idatentest/main.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/idatentest/main.cpp $(Release_Include_Path) -o x64/Release/idatentest/src/idatentest/main.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/idatentest/main.cpp $(Release_Include_Path) > x64/Release/idatentest/src/idatentest/main.d

# Creates the intermediate and output folders for each configuration...
.PHONY: create_folders
create_folders:
	mkdir -p x64/Debug/idatentest/src/common
	mkdir -p x64/Debug/idatentest/src/idatentest
	mkdir -p ../src/idatentest
	mkdir -p x64/Release/idatentest/src/common
	mkdir -p x64/Release/idatentest/src/idatentest
	mkdir -p ../src/idatentest

# Cleans intermediate and output files (objects, libraries, executables)...
.PHONY: clean
clean:
	rm -f x64/Debug/idatentest/src/common/*.o
	rm -f x64/Debug/idatentest/src/common/*.d
	rm -f x64/Debug/idatentest/src/idatentest/*.o
	rm -f x64/Debug/idatentest/src/idatentest/*.d
	rm -f x64/Debug/idatentest/*.o
	rm -f x64/Debug/idatentest/*.d
	rm -f ../src/idatentest/idatentest.exe
	rm -f x64/Release/idatentest/src/common/*.o
	rm -f x64/Release/idatentest/src/common/*.d
	rm -f x64/Release/idatentest/src/idatentest/*.o
	rm -f x64/Release/idatentest/src/idatentest/*.d
	rm -f x64/Release/idatentest/*.o
	rm -f x64/Release/idatentest/*.d
	rm -f ../src/idatentest/idatentest.exe

