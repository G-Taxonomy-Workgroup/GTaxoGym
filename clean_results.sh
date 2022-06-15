#!/bin/bash --login
# This script cleans up the result direcotry by 1) removing checkpoint files
# residing in any ckpt/ directory under results, and 2) removing individual
# runlog files.
#
# To prevent accidently runing the script, it prompts for whether the user
# really want to execute the cleaning script. Answer yes ('y', 'Y', 'yes', ...)
# to confirm, and no ('n', 'N', 'no', ...) to abort.

while true; do
    read -p "Do you wish to clean up the result directory by removing all logs and ckpts? (yn)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

homedir=$(dirname $(realpath $0))
echo homedir=$homedir

if [[ -d results ]]; then
    cd results

    echo Removing checkpoint files
    find -type d -name "ckpt" -exec rm -rf {} +

    for i in $(seq 0 9); do
        echo Removing individual run log for seed $i
        find -type d -name $i -exec rm -rf {} +
    done

    echo Done!
else
    echo results/ directory does not exist
fi
