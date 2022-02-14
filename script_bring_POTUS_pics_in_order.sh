#!/bin/bash

# key past problems here:
# float multiplication
# float rounding

# could do here: 
# consolidation of paths
# rounding and multiplication in function


copy_function() {
    # Anzahl an files in ordner:
    anzahl_pics=$(ls | wc -l)
    k=0
    train_split=0.7
    val_split=0.15
    eval_split=0.15
    echo $anzahl_pics


    #anzahl_train_pics=$(( $anzahl_pics * $train_split))
    anzahl_train_pics=$(echo "$anzahl_pics*$train_split" | bc)

    #echo $anzahl_train_pics
    anzahl_train_pics=$(printf "%.0f\n" $anzahl_train_pics)
    #echo $anzahl_train_pics

    anzahl_val_pics=$(echo "$anzahl_pics*$val_split" | bc)
    anzahl_val_pics=$(printf "%.0f\n" $anzahl_val_pics)

    anzahl_eval_pics=$(echo "$anzahl_pics*$eval_split" | bc)
    anzahl_eval_pics=$(printf "%.0f\n" $anzahl_eval_pics)

    for f in *
    do 
        if [[ k -lt anzahl_train_pics ]]
        then
            cp "$f" "/Users/br/Documents/Semester 3 Beuth/learning from images/project/POTUS pics well ordered/training/${i}_$k"
        elif [[ $(( k - anzahl_train_pics )) -lt  anzahl_val_pics ]]
        then
            cp "$f" "/Users/br/Documents/Semester 3 Beuth/learning from images/project/POTUS pics well ordered/validation/${i}_$k"
        elif [[ $(($(( k - anzahl_train_pics )) - anzahl_val_pics)) -lt  anzahl_eval_pics ]]
        then
            cp "$f" "/Users/br/Documents/Semester 3 Beuth/learning from images/project/POTUS pics well ordered/evaluation/${i}_$k"
        fi
        k=$((k + 1))
    done 
    
} 

the_for_loop(){
    #echo 'function 1 entry'
    #pwd
    #echo 'abgeschlossen'
    # i= Laufvariable: der wievielte Ordner?
    i=0
    for d in *
    do 
        cd "$d" && pwd && copy_function "$1" && i=$((i + 1)) && echo $i && cd .. 
    done
}

path_goal="/Users/br/Documents/Semester 3 Beuth/learning from images/project/POTUS pics well ordered/"
rm -r "${path_goal}"
mkdir "${path_goal}"

text="training"
mkdir "${path_goal}$text"

text="evaluation"
mkdir "${path_goal}$text"

text="validation"
mkdir "${path_goal}$text"

cd "/Users/br/Documents/Semester 3 Beuth/learning from images/project/POTUS pics/"
the_for_loop "${text}"
