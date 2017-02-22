:<<'END'
t='_Test.avi'
a='_TestAG.avi'
o='_output.avi'
for j in {3,10,101,1001,1777,2549,4009,4385,4802,4858};
do
	echo $j
	ffmpeg -y -ss 0.04 -i $j$t Done/$j$t
done
mv Done/* .

for j in {3,10,101,1001,1777,2549,4009,4385,4802,4858};
do
	echo $j
	
	ffmpeg -y -i $j$t -i $j$a -filter_complex \
	'[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
	-map [vid] -c:v libx264 -crf 23 -preset veryfast $j$o
done
mv *_output.avi Done/

cd Done
o='_output.avi'
oo='_output_speed.avi'
for i in {3,10,101,1001,1777,2549};
do
	ffmpeg -y -i $i$o -filter:v "setpts=0.075*PTS" $i$oo
done
ffmpeg -y -i 4009_output.avi -filter:v "setpts=5*PTS" 4009_output_speed.avi
ffmpeg -y -i 4385_output.avi -filter:v "setpts=7*PTS" 4385_output_speed.avi
ffmpeg -y -i 4802_output.avi -filter:v "setpts=7*PTS" 4802_output_speed.avi
ffmpeg -y -i 4858_output.avi -filter:v "setpts=6*PTS" 4858_output_speed.avi
rm *output.avi


cd Done

n='_output_speed.avi'
a='After '
o=' training episode'
oo='_output.avi'
for j in {3,10,101,1001,1777,2549,4009,4385,4802,4858};
do
	jj=$(($j*10))
	#ffmpeg -y -i $j$n -vf drawtext="fontfile=/path/to/font.ttf:text='After "$jj" Episode': fontcolor=white: fontsize=24: box=1: boxcolor=black@0.5:boxborderw=5: x=(w-text_w)/2: y=(h-text_h)/2" -codec:a copy $j$oo
	ffmpeg -y -i $j$n -vf drawtext="text='After "$jj" Episode': fontcolor=white: fontsize=24: box=1: boxcolor=black@0.5:boxborderw=5: x=(w-text_w)/2: y=(h-text_h)/2" -vcodec huffyuv -acodec copy $j$oo
done
rm *$n
END
cd Done
ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.avi
