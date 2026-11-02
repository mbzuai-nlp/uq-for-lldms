# usage: bash r3.sh ds id pct   -> slow + realtime renders into out4/
mkdir -p out4
ds=$1; id=$2; pct=$3
case $ds in
 gsm8k) N="--lo 0.0012249873 --hi 0.0341487682 --thr 0.0182288414"; Q="";;
 triviaqa) N="--lo 0.0001 --hi 2.0309843950 --thr 0.1989868046"; Q="--slowstep 500 --slowpause 700";;
 coqa) N="--lo 0.0001765364 --hi 0.1228451040 --thr 0.0457071396"; Q="--slowstep 500 --slowpause 700";;
esac
python3 make_gif.py va_$ds.json $id out4/${ds}_$id --pct $pct --scale 2 $N $Q
python3 make_gif.py va_$ds.json $id out4/${ds}_${id}_rt --pct $pct --scale 2 $N --realtime
echo DONE $ds $id
