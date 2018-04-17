# Builds all the projects in the solution...
.PHONY: all_projects
all_projects: Debug Release 

.PHONY: Debug
Debug: libaten_Debug libatenscene_Debug 

.PHONY: Release
Release: libaten_Release libatenscene_Release 

# Builds project 'libaten'...
.PHONY: libaten_Debug
libaten_Debug: 
	make --directory="../vs2015/" --file=libaten.makefile Debug

.PHONY: libaten_Release
libaten_Release: 
	make --directory="../vs2015/" --file=libaten.makefile Release

# Builds project 'libatenscene'...
.PHONY: libatenscene_Debug
libatenscene_Debug: 
	make --directory="../vs2015/" --file=libatenscene.makefile Debug

.PHONY: libatenscene_Release
libatenscene_Release: 
	make --directory="../vs2015/" --file=libatenscene.makefile Release

# Cleans all projects...
.PHONY: clean
clean:
	make --directory="../vs2015/" --file=libaten.makefile clean
	make --directory="../vs2015/" --file=libatenscene.makefile clean

