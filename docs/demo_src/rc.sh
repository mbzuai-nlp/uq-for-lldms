# comparison renders (real time + standard CoCoA with its own normalised score)
mkdir -p out4
r(){ ds=$1; id=$2; pct=$3; cs=$4
case $ds in
 gsm8k) N="--lo 0.0012249873 --hi 0.0341487682 --thr 0.0182288414"; C="--cmp 55 --dover 2 --dlabel ≈2_s --clo 0.8306928537 --chi 61.2381986 --cthr 10.51205684";;
 triviaqa) N="--lo 0.0001 --hi 2.0309843950 --thr 0.1989868046"; C="--cmp 7.5 --dover 0.1 --dlabel ≈0.1_s --clo 0.2409777502 --chi 43.0874418 --cthr 19.2364001";;
 coqa) N="--lo 0.0001765364 --hi 0.1228451040 --thr 0.0457071396"; C="--cmp 10.2 --dover 0.05 --dlabel <0.1_s --clo 0.5254086035 --chi 24.07954665 --cthr 10.7770877";;
esac
python3 make_gif.py va_$ds.json $id out4/${ds}_${id}_cmp --pct $pct --scale 2 $N --realtime $C --cscore $cs --nogif
echo DONE cmp $ds $id; }
if [ "$1" = a ]; then
r triviaqa 92 2 40.92210547; r triviaqa 235 3 0.3918116608; r triviaqa 99 42 0.4534620732; r gsm8k 182 4 68.96511105; r gsm8k 279 20 1.662904742
else
r coqa 149 10 24.11180996; r coqa 146 2 0.8599687359; r coqa 12 16 3.345169619; r gsm8k 599 3 9.846853728
fi
