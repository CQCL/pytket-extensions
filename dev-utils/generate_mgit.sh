#!/bin/bash

# file to generate bash files, clone, commit, push, and open PR on a selected subset of all pytket extensions

# run `bash mgitclone.sh` to clone all the repos
# run `bash mgit.sh status` to run `git status` on all repos
# run `bash mgit.sh add .` to run `git add .` on all repos
# run `bash mgitnewbranch.sh new/branch` to generate a new branch from develop with the name `new/branch` on all repos

echo "#!/bin/bash" > mgit.sh
echo "#!/bin/bash" > mgitclone.sh
echo "#!/bin/bash" > mgitcommit.sh
echo "#!/bin/bash" > mgitnewbranch.sh
echo "#!/bin/bash" > mgitopenpr.sh
echo "#!/bin/bash" > mgitrename.sh
echo "#!/bin/bash" > mgitcopy.sh

# choose the list of the extensions you want to use
# list of all extensions: "aqt" "braket" "cirq" "ionq" "iqm" "pennylane" "projectq" "pyquil" "pysimplex" "pyzx" "qir" "qiskit" "qsharp" "quantinuum" "qulacs" "qujax" "stim"
for ext in "aqt" "braket" "cirq" "ionq" "iqm" "pennylane" "projectq" "pyquil" "pysimplex" "pyzx" "qir" "qiskit" "qsharp" "quantinuum" "qulacs" "qujax" "stim"
do
  
  echo "git clone git@github.com:CQCL/pytket-$ext.git" >> mgitclone.sh

  echo "" >> mgit.sh
  echo "cd pytket-$ext" >> mgit.sh
  echo "pwd" >> mgit.sh
  echo "git \$1 \$2 \$3 \$4 " >> mgit.sh
  echo "cd .." >> mgit.sh
  echo "" >> mgit.sh

  echo "" >> mgitcommit.sh
  echo "cd pytket-$ext" >> mgitcommit.sh
  echo "pwd" >> mgitcommit.sh
  echo "git commit -m\"\$1\"" >> mgitcommit.sh
  echo "cd .." >> mgitcommit.sh
  echo "" >> mgitcommit.sh

  echo "" >> mgitnewbranch.sh
  echo "cd pytket-$ext" >> mgitnewbranch.sh
  echo "pwd" >> mgitnewbranch.sh
  echo "git checkout develop" >> mgitnewbranch.sh
  echo "git pull" >> mgitnewbranch.sh
  echo "git checkout -b \$1" >> mgitnewbranch.sh
  echo "cd .." >> mgitnewbranch.sh
  echo "" >> mgitnewbranch.sh

  echo "" >> mgitopenpr.sh
  echo "cd pytket-$ext" >> mgitopenpr.sh
  echo "pwd" >> mgitopenpr.sh
  echo "git checkout \$2" >> mgitopenpr.sh
  echo "git push" >> mgitopenpr.sh
  echo "cd .." >> mgitopenpr.sh
  echo "firefox https://github.com/CQCL/pytket-$ext/compare/\$1...\$2 &"  >> mgitopenpr.sh
  echo "" >> mgitopenpr.sh

  echo "" >> mgitrename.sh
  echo "cd pytket-$ext" >> mgitrename.sh
  echo "pwd" >> mgitrename.sh
  echo "mv \$1 \$2" >> mgitrename.sh
  echo "cd .." >> mgitrename.sh
  echo "" >> mgitrename.sh

  echo "" >> mgitcopy.sh
  echo "cd pytket-$ext" >> mgitcopy.sh
  echo "pwd" >> mgitcopy.sh
  echo "cp \$1 \$2" >> mgitcopy.sh
  echo "cd .." >> mgitcopy.sh
  echo "" >> mgitcopy.sh

done

echo ""
echo "file mgit.sh:"

cat mgit.sh

echo ""
echo "file mgitclone.sh:"

cat mgitclone.sh

echo ""
echo "file mgitcommit.sh:"

cat mgitcommit.sh

echo ""
echo "file mgitnewbranch.sh:"

cat mgitnewbranch.sh

echo ""
echo "file mgitopenpr.sh:"

cat mgitopenpr.sh




