#!/usr/bin/env bash
# renames all the output of the outomated file collection from the PA
rename -v 's/run001/run002/' $1*VG20VDS0.1*
rename -v 's/run001/run003/' $1*VG10VDS0.01*
rename -v 's/run001/run004/' $1*VG20VDS0.01*
rename -v 's/run001/run005/' $1*VG10VDS0.1+*
rename -v 's/run001/run005/' $1*VG10VDS1+*
rename -v 's/run001/run006/' $1*VDS5VG*
