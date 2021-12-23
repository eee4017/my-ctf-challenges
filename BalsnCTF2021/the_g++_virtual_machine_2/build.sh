set -eux
argument=`python -c "flag=input();print(' '.join([f\"-D__IN_{i}={ ( ord(c)-ord('a') ) % 31 }\" for i, c in enumerate(flag)]))"`
python compiler.py --debug --input vm.cc --output output.cpp --obfuscate
g++ ${argument} -ftemplate-depth=2147483647 -std=c++17 -I. -o output output.cpp

if [ `./output` = "25" ]
then
  echo "yes"
else
  echo "no"
fi

rm output
