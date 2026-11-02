# all renders with timings MEASURED by bench.py (one A100-SXM4-40GB on CIAI, median of 3 runs; traces re-recorded on the same GPU). usage: bash r5.sh ds id pct stepms cocoa_s dcocoa_s cscore
mkdir -p out7
ds=$1; id=$2; pct=$3; ms=$4; cs=$5; dov=$6; csc=$7
case $ds in
 gsm8k) N="--lo 0.001449353013 --hi 0.03450019798 --thr 0.01708634944"; Q=""; CC="--clo 0.6769024452 --chi 23.86356478 --cthr 11.38401679";;
 triviaqa) N="--lo 0.0001 --hi 2.015086394 --thr 0.200773274"; Q="--slowstep 500 --slowpause 700"; CC="--clo 0.2403102097 --chi 35.41549844 --cthr 20.6198946";;
 coqa) N="--lo 0.000175909764 --hi 0.1179505322 --thr 0.04258194379"; Q="--slowstep 500 --slowpause 700"; CC="--clo 0.5004416028 --chi 24.16632694 --cthr 11.45975322";;
esac
lab=$(python3 -c "v=$dov; print('<1_ms' if v<0.0015 else (f'{v*1000:.0f}_ms' if v<1 else f'{v:.1f}_s'))")
python3 make_gif.py va_$ds.json $id out7/${ds}_$id --pct $pct --scale 2 $N $Q --steptimes steptimes/${ds}_${id}.json
python3 make_gif.py va_$ds.json $id out7/${ds}_${id}_rt --pct $pct --scale 2 $N --realtime --steptimes steptimes/${ds}_${id}.json
python3 make_gif.py va_$ds.json $id out7/${ds}_${id}_cmp --pct $pct --scale 2 $N --realtime --steptimes steptimes/${ds}_${id}.json --cmp $cs --dover $dov --dlabel $lab $CC --cscore $csc --nogif
echo DONE $ds $id
