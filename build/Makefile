# Builds all the projects in the solution...
.PHONY: all_projects
all_projects: Debug Release 

.PHONY: Debug
Debug: libaten_Debug 

.PHONY: Release
Release: libaten_Release 

# Builds project 'libaten'...
.PHONY: libaten_Debug
libaten_Debug: 
	make --directory="../vs2015/" --file=libaten.makefile Debug

.PHONY: libaten_Release
libaten_Release: 
	make --directory="../vs2015/" --file=libaten.makefile Release

# Cleans all projects...
.PHONY: clean
clean:
	make --directory="../vs2015/" --file=libaten.makefile clean

