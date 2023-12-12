#!/bin/bash

VENDOR=$1

function arr_sub1()
{
    local cards_num=("$@")
    local new_cards_num=()

    for elem in "${cards_num[@]}"; do
        new_elem=$((elem -1))
        new_cards_num+=("$new_elem")
    done
    echo "${new_cards_num[@]}"
}


function get_available_cards_on_supa()
{
    # for example available_cards:1 2 3 4 ...
    available_cards=$(brsmi gpu query -d memory | grep Used | awk '{print $3}' | grep -nw 0 |  awk -F : '{print $1}')
    available_cards=$(arr_sub1 $available_cards)
    echo "$available_cards"
}

function get_available_cards_on_ascend()
{
    # for example available_cards:1 2 3 4 ...
    available_cards=$(npu-smi info | grep  -E "/.*/" | awk -F " " '{print $10}' | grep -nw 1 | awk -F : '{print $1}') # grep -nw 1 for docker, otherwise grep -nw 0
    available_cards=$(arr_sub1 $available_cards)
    echo "$available_cards"
}

function run()
{
    vendor=$1
    env_var_need_export=$2

    available_cards=$(get_available_cards_on_"${vendor}")
    # no available card, not export the valiable_devices env var
    if [ "$available_cards" = "" ];then
        echo "no available card to pick, run on the default card"
        return 0
    fi
    echo "available cards are" "$available_cards"
    IFS=' ' read -ra array <<< "$available_cards" # convert to the type array
    visiable_device="${array[-1]}"  #get the last one
    echo "run on the picked card $visiable_device"

    # export the visiable_devices env var
    export "${env_var_need_export}"="$visiable_device"
}

if [ "$VENDOR" = "supa" ];then
    run supa SUPA_VISIABLE_DEVICES
elif [ "$VENDOR" = "ascend" ];then
    run ascend ASCEND_VISIABLE_DEVICES
else
    echo "Invalid VENDOR parameter: $VENDOR" 1>&2
fi