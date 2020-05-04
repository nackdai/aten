#!/bin/bash

# e option : Exit immediately if an error happen.
#            i.e. exit immeddiately if any command return 'Exit 0'
# u option : Treat unset variables as an error when substituting.
set -eu

# $* : Treat all arguments as one argument
# Run input command
result=$(sh -c "$*")

# Set outout variable
# In this case, set outcome of specified command to 'result'
echo "::set-output name=result::$result"
