# Builds all the projects in the solution...
.PHONY: all_projects
all_projects: Debug Release 

.PHONY: Debug
Debug: libatenscene_Debug libidaten_Debug libaten_Debug appaten_Debug idatentest_Debug svgftest_Debug mansion_Debug multigputest_Debug 

.PHONY: Release
Release: libatenscene_Release libidaten_Release libaten_Release appaten_Release idatentest_Release svgftest_Release mansion_Release multigputest_Release 

# Builds project 'appaten'...
.PHONY: appaten_Debug
appaten_Debug: libatenscene libaten 
	make --directory=./ --file=appaten.makefile Debug

.PHONY: appaten_Release
appaten_Release: libatenscene libaten 
	make --directory=./ --file=appaten.makefile Release

# Builds project 'idatentest'...
.PHONY: idatentest_Debug
idatentest_Debug: libatenscene libidaten libaten 
	make --directory=./ --file=idatentest.makefile Debug

.PHONY: idatentest_Release
idatentest_Release: libatenscene libidaten libaten 
	make --directory=./ --file=idatentest.makefile Release

# Builds project 'svgftest'...
.PHONY: svgftest_Debug
svgftest_Debug: libatenscene libidaten libaten 
	make --directory=./ --file=svgftest.makefile Debug

.PHONY: svgftest_Release
svgftest_Release: libatenscene libidaten libaten 
	make --directory=./ --file=svgftest.makefile Release

# Builds project 'libatenscene'...
.PHONY: libatenscene_Debug
libatenscene_Debug: 
	make --directory=./ --file=libatenscene.makefile Debug

.PHONY: libatenscene_Release
libatenscene_Release: 
	make --directory=./ --file=libatenscene.makefile Release

# Builds project 'libidaten'...
.PHONY: libidaten_Debug
libidaten_Debug: 
	make --directory=./ --file=libidaten.makefile Debug

.PHONY: libidaten_Release
libidaten_Release: 
	make --directory=./ --file=libidaten.makefile Release

# Builds project 'libaten'...
.PHONY: libaten_Debug
libaten_Debug: 
	make --directory=./ --file=libaten.makefile Debug

.PHONY: libaten_Release
libaten_Release: 
	make --directory=./ --file=libaten.makefile Release

# Builds project 'mansion'...
.PHONY: mansion_Debug
mansion_Debug: libatenscene libidaten libaten 
	make --directory=./ --file=mansion.makefile Debug

.PHONY: mansion_Release
mansion_Release: libatenscene libidaten libaten 
	make --directory=./ --file=mansion.makefile Release

# Builds project 'multigputest'...
.PHONY: multigputest_Debug
multigputest_Debug: libatenscene libidaten libaten 
	make --directory=./ --file=multigputest.makefile Debug

.PHONY: multigputest_Release
multigputest_Release: libatenscene libidaten libaten 
	make --directory=./ --file=multigputest.makefile Release

# Cleans all projects...
.PHONY: clean
clean:
	make --directory=./ --file=appaten.makefile clean
	make --directory=./ --file=idatentest.makefile clean
	make --directory=./ --file=svgftest.makefile clean
	make --directory=./ --file=libatenscene.makefile clean
	make --directory=./ --file=libidaten.makefile clean
	make --directory=./ --file=libaten.makefile clean
	make --directory=./ --file=mansion.makefile clean
	make --directory=./ --file=multigputest.makefile clean

