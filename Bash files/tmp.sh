cd APES2/output
for dir in */
do
    dir=${dir%*/}
    echo ${dir##*/}
    tail -n 5 ../logs/${dir##*/}.out
done

