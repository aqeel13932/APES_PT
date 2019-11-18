cd output
for dir in */
do
	cd $dir
	rm -r PNG VID
	cd ../
done
