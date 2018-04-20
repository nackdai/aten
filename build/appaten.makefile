# Compiler flags...
CPP_COMPILER = g++
C_COMPILER = gcc

# Include paths...
Debug_Include_Path=-I"../src/libaten" -I"../src/libatenscene" -I"../3rdparty/glew/include" -I"../3rdparty/glm" 
Release_Include_Path=-I"../src/libaten" -I"../src/libatenscene" -I"../3rdparty/glew/include" -I"../3rdparty/glm" 

# Library paths...
Debug_Library_Path=-L"../build/x64/Debug" 
Release_Library_Path=-L"../build/x64/Release" 

# Additional libraries...
Debug_Libraries=-Wl,--no-as-needed -Wl,--start-group -laten -latenscene -lGLEW -lglfw -lGL -fopenmp  -Wl,--end-group
Release_Libraries=-Wl,--no-as-needed -Wl,--start-group -laten -latenscene -lGLEW -lglfw -lGL -fopenmp  -Wl,--end-group

# Preprocessor definitions...
Debug_Preprocessor_Definitions=-D __AT_DEBUG__ -D GCC_BUILD 
Release_Preprocessor_Definitions=-D GCC_BUILD 

# Implictly linked object files...
Debug_Implicitly_Linked_Objects=
Release_Implicitly_Linked_Objects=

# Compiler flags...
Debug_Compiler_Flags=-O0 -g -std=c++11 -fopenmp 
Release_Compiler_Flags=-O2 -g -std=c++11 -fopenmp 

# Builds all configurations for this project...
.PHONY: build_all_configurations
build_all_configurations: Debug Release 

# Builds the Debug configuration...
.PHONY: Debug
Debug: create_folders x64/Debug/appaten/src/appaten/main_denoise.o x64/Debug/appaten/src/appaten/main_test.o x64/Debug/appaten/src/appaten/scenedefs.o 
	g++ x64/Debug/appaten/src/appaten/main_denoise.o x64/Debug/appaten/src/appaten/main_test.o x64/Debug/appaten/src/appaten/scenedefs.o  $(Debug_Library_Path) $(Debug_Libraries) -Wl,-rpath,./ -o ../src/appaten/appaten.exe

# Compiles file ../src/appaten/main_denoise.cpp for the Debug configuration...
-include x64/Debug/appaten/src/appaten/main_denoise.d
x64/Debug/appaten/src/appaten/main_denoise.o: ../src/appaten/main_denoise.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/appaten/main_denoise.cpp $(Debug_Include_Path) -o x64/Debug/appaten/src/appaten/main_denoise.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/appaten/main_denoise.cpp $(Debug_Include_Path) > x64/Debug/appaten/src/appaten/main_denoise.d

# Compiles file ../src/appaten/main_test.cpp for the Debug configuration...
-include x64/Debug/appaten/src/appaten/main_test.d
x64/Debug/appaten/src/appaten/main_test.o: ../src/appaten/main_test.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/appaten/main_test.cpp $(Debug_Include_Path) -o x64/Debug/appaten/src/appaten/main_test.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/appaten/main_test.cpp $(Debug_Include_Path) > x64/Debug/appaten/src/appaten/main_test.d

# Compiles file ../src/appaten/scenedefs.cpp for the Debug configuration...
-include x64/Debug/appaten/src/appaten/scenedefs.d
x64/Debug/appaten/src/appaten/scenedefs.o: ../src/appaten/scenedefs.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/appaten/scenedefs.cpp $(Debug_Include_Path) -o x64/Debug/appaten/src/appaten/scenedefs.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/appaten/scenedefs.cpp $(Debug_Include_Path) > x64/Debug/appaten/src/appaten/scenedefs.d

# Builds the Release configuration...
.PHONY: Release
Release: create_folders x64/Release/appaten/src/appaten/main_denoise.o x64/Release/appaten/src/appaten/main_test.o x64/Release/appaten/src/appaten/scenedefs.o 
	g++ x64/Release/appaten/src/appaten/main_denoise.o x64/Release/appaten/src/appaten/main_test.o x64/Release/appaten/src/appaten/scenedefs.o  $(Release_Library_Path) $(Release_Libraries) -Wl,-rpath,./ -o ../src/appaten/appaten.exe

# Compiles file ../src/appaten/main_denoise.cpp for the Release configuration...
-include x64/Release/appaten/src/appaten/main_denoise.d
x64/Release/appaten/src/appaten/main_denoise.o: ../src/appaten/main_denoise.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/appaten/main_denoise.cpp $(Release_Include_Path) -o x64/Release/appaten/src/appaten/main_denoise.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/appaten/main_denoise.cpp $(Release_Include_Path) > x64/Release/appaten/src/appaten/main_denoise.d

# Compiles file ../src/appaten/main_test.cpp for the Release configuration...
-include x64/Release/appaten/src/appaten/main_test.d
x64/Release/appaten/src/appaten/main_test.o: ../src/appaten/main_test.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/appaten/main_test.cpp $(Release_Include_Path) -o x64/Release/appaten/src/appaten/main_test.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/appaten/main_test.cpp $(Release_Include_Path) > x64/Release/appaten/src/appaten/main_test.d

# Compiles file ../src/appaten/scenedefs.cpp for the Release configuration...
-include x64/Release/appaten/src/appaten/scenedefs.d
x64/Release/appaten/src/appaten/scenedefs.o: ../src/appaten/scenedefs.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/appaten/scenedefs.cpp $(Release_Include_Path) -o x64/Release/appaten/src/appaten/scenedefs.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/appaten/scenedefs.cpp $(Release_Include_Path) > x64/Release/appaten/src/appaten/scenedefs.d

# Creates the intermediate and output folders for each configuration...
.PHONY: create_folders
create_folders:
	mkdir -p x64/Debug/appaten/src/appaten
	mkdir -p ../src/appaten
	mkdir -p x64/Release/appaten/src/appaten
	mkdir -p ../src/appaten

# Cleans intermediate and output files (objects, libraries, executables)...
.PHONY: clean
clean:
	rm -f x64/Debug/appaten/src/appaten/*.o
	rm -f x64/Debug/appaten/src/appaten/*.d
	rm -f ../src/appaten/appaten.exe
	rm -f x64/Release/appaten/src/appaten/*.o
	rm -f x64/Release/appaten/src/appaten/*.d
	rm -f ../src/appaten/appaten.exe

