# Compiler flags...
CPP_COMPILER = g++
C_COMPILER = gcc

# Include paths...
Debug_Include_Path=-I"../src/libatenscene" -I"../src/libaten" -I"../3rdparty/tinyobjloader_v09" -I"../3rdparty/picojson" -I"../3rdparty/stb" -I"../3rdparty/tinyxml2" -I"../3rdparty/glm" 
Release_Include_Path=-I"../src/libatenscene" -I"../src/libaten" -I"../3rdparty/tinyobjloader_v09" -I"../3rdparty/picojson" -I"../3rdparty/stb" -I"../3rdparty/tinyxml2" -I"../3rdparty/glm" 

# Library paths...
Debug_Library_Path=
Release_Library_Path=

# Additional libraries...
Debug_Libraries=-Wl,--no-as-needed -Wl,--start-group -l$(NOINHERIT)  -Wl,--end-group
Release_Libraries=-Wl,--no-as-needed -Wl,--start-group -l$(NOINHERIT)  -Wl,--end-group

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
Debug: create_folders x64/Debug/libatenscene/src/libatenscene/AssetManager.o x64/Debug/libatenscene/src/libatenscene/ImageLoader.o x64/Debug/libatenscene/src/libatenscene/MaterialExporter.o x64/Debug/libatenscene/src/libatenscene/MaterialLoader.o x64/Debug/libatenscene/src/libatenscene/ObjLoader.o x64/Debug/libatenscene/src/libatenscene/ObjWriter.o x64/Debug/libatenscene/src/libatenscene/SceneLoader.o x64/Debug/libatenscene/3rdparty/tinyxml2/tinyxml2.o x64/Debug/libatenscene/3rdparty/tinyobjloader_v09/tiny_obj_loader.o 
	ar rcs x64/Debug/libatenscene.a x64/Debug/libatenscene/src/libatenscene/AssetManager.o x64/Debug/libatenscene/src/libatenscene/ImageLoader.o x64/Debug/libatenscene/src/libatenscene/MaterialExporter.o x64/Debug/libatenscene/src/libatenscene/MaterialLoader.o x64/Debug/libatenscene/src/libatenscene/ObjLoader.o x64/Debug/libatenscene/src/libatenscene/ObjWriter.o x64/Debug/libatenscene/src/libatenscene/SceneLoader.o x64/Debug/libatenscene/3rdparty/tinyxml2/tinyxml2.o x64/Debug/libatenscene/3rdparty/tinyobjloader_v09/tiny_obj_loader.o  $(Debug_Implicitly_Linked_Objects)

# Compiles file ../src/libatenscene/AssetManager.cpp for the Debug configuration...
-include x64/Debug/libatenscene/src/libatenscene/AssetManager.d
x64/Debug/libatenscene/src/libatenscene/AssetManager.o: ../src/libatenscene/AssetManager.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libatenscene/AssetManager.cpp $(Debug_Include_Path) -o x64/Debug/libatenscene/src/libatenscene/AssetManager.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libatenscene/AssetManager.cpp $(Debug_Include_Path) > x64/Debug/libatenscene/src/libatenscene/AssetManager.d

# Compiles file ../src/libatenscene/ImageLoader.cpp for the Debug configuration...
-include x64/Debug/libatenscene/src/libatenscene/ImageLoader.d
x64/Debug/libatenscene/src/libatenscene/ImageLoader.o: ../src/libatenscene/ImageLoader.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libatenscene/ImageLoader.cpp $(Debug_Include_Path) -o x64/Debug/libatenscene/src/libatenscene/ImageLoader.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libatenscene/ImageLoader.cpp $(Debug_Include_Path) > x64/Debug/libatenscene/src/libatenscene/ImageLoader.d

# Compiles file ../src/libatenscene/MaterialExporter.cpp for the Debug configuration...
-include x64/Debug/libatenscene/src/libatenscene/MaterialExporter.d
x64/Debug/libatenscene/src/libatenscene/MaterialExporter.o: ../src/libatenscene/MaterialExporter.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libatenscene/MaterialExporter.cpp $(Debug_Include_Path) -o x64/Debug/libatenscene/src/libatenscene/MaterialExporter.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libatenscene/MaterialExporter.cpp $(Debug_Include_Path) > x64/Debug/libatenscene/src/libatenscene/MaterialExporter.d

# Compiles file ../src/libatenscene/MaterialLoader.cpp for the Debug configuration...
-include x64/Debug/libatenscene/src/libatenscene/MaterialLoader.d
x64/Debug/libatenscene/src/libatenscene/MaterialLoader.o: ../src/libatenscene/MaterialLoader.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libatenscene/MaterialLoader.cpp $(Debug_Include_Path) -o x64/Debug/libatenscene/src/libatenscene/MaterialLoader.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libatenscene/MaterialLoader.cpp $(Debug_Include_Path) > x64/Debug/libatenscene/src/libatenscene/MaterialLoader.d

# Compiles file ../src/libatenscene/ObjLoader.cpp for the Debug configuration...
-include x64/Debug/libatenscene/src/libatenscene/ObjLoader.d
x64/Debug/libatenscene/src/libatenscene/ObjLoader.o: ../src/libatenscene/ObjLoader.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libatenscene/ObjLoader.cpp $(Debug_Include_Path) -o x64/Debug/libatenscene/src/libatenscene/ObjLoader.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libatenscene/ObjLoader.cpp $(Debug_Include_Path) > x64/Debug/libatenscene/src/libatenscene/ObjLoader.d

# Compiles file ../src/libatenscene/ObjWriter.cpp for the Debug configuration...
-include x64/Debug/libatenscene/src/libatenscene/ObjWriter.d
x64/Debug/libatenscene/src/libatenscene/ObjWriter.o: ../src/libatenscene/ObjWriter.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libatenscene/ObjWriter.cpp $(Debug_Include_Path) -o x64/Debug/libatenscene/src/libatenscene/ObjWriter.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libatenscene/ObjWriter.cpp $(Debug_Include_Path) > x64/Debug/libatenscene/src/libatenscene/ObjWriter.d

# Compiles file ../src/libatenscene/SceneLoader.cpp for the Debug configuration...
-include x64/Debug/libatenscene/src/libatenscene/SceneLoader.d
x64/Debug/libatenscene/src/libatenscene/SceneLoader.o: ../src/libatenscene/SceneLoader.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libatenscene/SceneLoader.cpp $(Debug_Include_Path) -o x64/Debug/libatenscene/src/libatenscene/SceneLoader.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libatenscene/SceneLoader.cpp $(Debug_Include_Path) > x64/Debug/libatenscene/src/libatenscene/SceneLoader.d

# Compiles file ../3rdparty/tinyxml2/tinyxml2.cpp for the Debug configuration...
-include x64/Debug/libatenscene/3rdparty/tinyxml2/tinyxml2.d
x64/Debug/libatenscene/3rdparty/tinyxml2/tinyxml2.o: ../3rdparty/tinyxml2/tinyxml2.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../3rdparty/tinyxml2/tinyxml2.cpp $(Debug_Include_Path) -o x64/Debug/libatenscene/3rdparty/tinyxml2/tinyxml2.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../3rdparty/tinyxml2/tinyxml2.cpp $(Debug_Include_Path) > x64/Debug/libatenscene/3rdparty/tinyxml2/tinyxml2.d

# Compiles file ../3rdparty/tinyobjloader_v09/tiny_obj_loader.cc for the Debug configuration...
-include x64/Debug/libatenscene/3rdparty/tinyobjloader_v09/tiny_obj_loader.d
x64/Debug/libatenscene/3rdparty/tinyobjloader_v09/tiny_obj_loader.o: ../3rdparty/tinyobjloader_v09/tiny_obj_loader.cc
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../3rdparty/tinyobjloader_v09/tiny_obj_loader.cc $(Debug_Include_Path) -o x64/Debug/libatenscene/3rdparty/tinyobjloader_v09/tiny_obj_loader.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../3rdparty/tinyobjloader_v09/tiny_obj_loader.cc $(Debug_Include_Path) > x64/Debug/libatenscene/3rdparty/tinyobjloader_v09/tiny_obj_loader.d

# Builds the Release configuration...
.PHONY: Release
Release: create_folders x64/Release/libatenscene/src/libatenscene/AssetManager.o x64/Release/libatenscene/src/libatenscene/ImageLoader.o x64/Release/libatenscene/src/libatenscene/MaterialExporter.o x64/Release/libatenscene/src/libatenscene/MaterialLoader.o x64/Release/libatenscene/src/libatenscene/ObjLoader.o x64/Release/libatenscene/src/libatenscene/ObjWriter.o x64/Release/libatenscene/src/libatenscene/SceneLoader.o x64/Release/libatenscene/3rdparty/tinyxml2/tinyxml2.o x64/Release/libatenscene/3rdparty/tinyobjloader_v09/tiny_obj_loader.o 
	ar rcs x64/Release/libatenscene.a x64/Release/libatenscene/src/libatenscene/AssetManager.o x64/Release/libatenscene/src/libatenscene/ImageLoader.o x64/Release/libatenscene/src/libatenscene/MaterialExporter.o x64/Release/libatenscene/src/libatenscene/MaterialLoader.o x64/Release/libatenscene/src/libatenscene/ObjLoader.o x64/Release/libatenscene/src/libatenscene/ObjWriter.o x64/Release/libatenscene/src/libatenscene/SceneLoader.o x64/Release/libatenscene/3rdparty/tinyxml2/tinyxml2.o x64/Release/libatenscene/3rdparty/tinyobjloader_v09/tiny_obj_loader.o  $(Release_Implicitly_Linked_Objects)

# Compiles file ../src/libatenscene/AssetManager.cpp for the Release configuration...
-include x64/Release/libatenscene/src/libatenscene/AssetManager.d
x64/Release/libatenscene/src/libatenscene/AssetManager.o: ../src/libatenscene/AssetManager.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libatenscene/AssetManager.cpp $(Release_Include_Path) -o x64/Release/libatenscene/src/libatenscene/AssetManager.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libatenscene/AssetManager.cpp $(Release_Include_Path) > x64/Release/libatenscene/src/libatenscene/AssetManager.d

# Compiles file ../src/libatenscene/ImageLoader.cpp for the Release configuration...
-include x64/Release/libatenscene/src/libatenscene/ImageLoader.d
x64/Release/libatenscene/src/libatenscene/ImageLoader.o: ../src/libatenscene/ImageLoader.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libatenscene/ImageLoader.cpp $(Release_Include_Path) -o x64/Release/libatenscene/src/libatenscene/ImageLoader.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libatenscene/ImageLoader.cpp $(Release_Include_Path) > x64/Release/libatenscene/src/libatenscene/ImageLoader.d

# Compiles file ../src/libatenscene/MaterialExporter.cpp for the Release configuration...
-include x64/Release/libatenscene/src/libatenscene/MaterialExporter.d
x64/Release/libatenscene/src/libatenscene/MaterialExporter.o: ../src/libatenscene/MaterialExporter.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libatenscene/MaterialExporter.cpp $(Release_Include_Path) -o x64/Release/libatenscene/src/libatenscene/MaterialExporter.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libatenscene/MaterialExporter.cpp $(Release_Include_Path) > x64/Release/libatenscene/src/libatenscene/MaterialExporter.d

# Compiles file ../src/libatenscene/MaterialLoader.cpp for the Release configuration...
-include x64/Release/libatenscene/src/libatenscene/MaterialLoader.d
x64/Release/libatenscene/src/libatenscene/MaterialLoader.o: ../src/libatenscene/MaterialLoader.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libatenscene/MaterialLoader.cpp $(Release_Include_Path) -o x64/Release/libatenscene/src/libatenscene/MaterialLoader.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libatenscene/MaterialLoader.cpp $(Release_Include_Path) > x64/Release/libatenscene/src/libatenscene/MaterialLoader.d

# Compiles file ../src/libatenscene/ObjLoader.cpp for the Release configuration...
-include x64/Release/libatenscene/src/libatenscene/ObjLoader.d
x64/Release/libatenscene/src/libatenscene/ObjLoader.o: ../src/libatenscene/ObjLoader.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libatenscene/ObjLoader.cpp $(Release_Include_Path) -o x64/Release/libatenscene/src/libatenscene/ObjLoader.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libatenscene/ObjLoader.cpp $(Release_Include_Path) > x64/Release/libatenscene/src/libatenscene/ObjLoader.d

# Compiles file ../src/libatenscene/ObjWriter.cpp for the Release configuration...
-include x64/Release/libatenscene/src/libatenscene/ObjWriter.d
x64/Release/libatenscene/src/libatenscene/ObjWriter.o: ../src/libatenscene/ObjWriter.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libatenscene/ObjWriter.cpp $(Release_Include_Path) -o x64/Release/libatenscene/src/libatenscene/ObjWriter.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libatenscene/ObjWriter.cpp $(Release_Include_Path) > x64/Release/libatenscene/src/libatenscene/ObjWriter.d

# Compiles file ../src/libatenscene/SceneLoader.cpp for the Release configuration...
-include x64/Release/libatenscene/src/libatenscene/SceneLoader.d
x64/Release/libatenscene/src/libatenscene/SceneLoader.o: ../src/libatenscene/SceneLoader.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libatenscene/SceneLoader.cpp $(Release_Include_Path) -o x64/Release/libatenscene/src/libatenscene/SceneLoader.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libatenscene/SceneLoader.cpp $(Release_Include_Path) > x64/Release/libatenscene/src/libatenscene/SceneLoader.d

# Compiles file ../3rdparty/tinyxml2/tinyxml2.cpp for the Release configuration...
-include x64/Release/libatenscene/3rdparty/tinyxml2/tinyxml2.d
x64/Release/libatenscene/3rdparty/tinyxml2/tinyxml2.o: ../3rdparty/tinyxml2/tinyxml2.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../3rdparty/tinyxml2/tinyxml2.cpp $(Release_Include_Path) -o x64/Release/libatenscene/3rdparty/tinyxml2/tinyxml2.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../3rdparty/tinyxml2/tinyxml2.cpp $(Release_Include_Path) > x64/Release/libatenscene/3rdparty/tinyxml2/tinyxml2.d

# Compiles file ../3rdparty/tinyobjloader_v09/tiny_obj_loader.cc for the Release configuration...
-include x64/Release/libatenscene/3rdparty/tinyobjloader_v09/tiny_obj_loader.d
x64/Release/libatenscene/3rdparty/tinyobjloader_v09/tiny_obj_loader.o: ../3rdparty/tinyobjloader_v09/tiny_obj_loader.cc
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../3rdparty/tinyobjloader_v09/tiny_obj_loader.cc $(Release_Include_Path) -o x64/Release/libatenscene/3rdparty/tinyobjloader_v09/tiny_obj_loader.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../3rdparty/tinyobjloader_v09/tiny_obj_loader.cc $(Release_Include_Path) > x64/Release/libatenscene/3rdparty/tinyobjloader_v09/tiny_obj_loader.d

# Creates the intermediate and output folders for each configuration...
.PHONY: create_folders
create_folders:
	mkdir -p x64/Debug/libatenscene/src/libatenscene
	mkdir -p x64/Debug/libatenscene/3rdparty/tinyxml2
	mkdir -p x64/Debug/libatenscene/3rdparty/tinyobjloader_v09
	mkdir -p x64/Debug
	mkdir -p x64/Release/libatenscene/src/libatenscene
	mkdir -p x64/Release/libatenscene/3rdparty/tinyxml2
	mkdir -p x64/Release/libatenscene/3rdparty/tinyobjloader_v09
	mkdir -p x64/Release

# Cleans intermediate and output files (objects, libraries, executables)...
.PHONY: clean
clean:
	rm -f x64/Debug/libatenscene/src/libatenscene/*.o
	rm -f x64/Debug/libatenscene/src/libatenscene/*.d
	rm -f x64/Debug/libatenscene/3rdparty/tinyxml2/*.o
	rm -f x64/Debug/libatenscene/3rdparty/tinyxml2/*.d
	rm -f x64/Debug/libatenscene/3rdparty/tinyobjloader_v09/*.o
	rm -f x64/Debug/libatenscene/3rdparty/tinyobjloader_v09/*.d
	rm -f x64/Debug/libatenscene/*.o
	rm -f x64/Debug/libatenscene/*.d
	rm -f x64/Debug/libatenscene.a
	rm -f x64/Release/libatenscene/src/libatenscene/*.o
	rm -f x64/Release/libatenscene/src/libatenscene/*.d
	rm -f x64/Release/libatenscene/3rdparty/tinyxml2/*.o
	rm -f x64/Release/libatenscene/3rdparty/tinyxml2/*.d
	rm -f x64/Release/libatenscene/3rdparty/tinyobjloader_v09/*.o
	rm -f x64/Release/libatenscene/3rdparty/tinyobjloader_v09/*.d
	rm -f x64/Release/libatenscene/*.o
	rm -f x64/Release/libatenscene/*.d
	rm -f x64/Release/libatenscene.a

