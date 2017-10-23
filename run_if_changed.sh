#!/usr/bin/env bash
# src: https://superuser.com/a/634313
set -ue -o pipefail
# set -vx

dbecho=""

if [[ ! -z ${1:-} ]];then
  file=$1
else
  echo "test with:"
  echo 'sleep 5 && touch server/res/gps_input_route.json & '
  echo "$0  server/res/gps_input_route.json "
  echo "sample usage:"
  echo "./run_if_changed.sh server/res/gps_input_route.json"
  exit
fi
if [[ ! -r $file ]]; then
  echo "INVALID"
fi

# doesn't work
# runhook="python3 ./code/model.py"
# ` ${runhook}

get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

target=$(readlink -e $file || exit 0);
target=$(get_abs_filename $file || exit 0);
if [[ ! -z ${target} ]]; then
  file=$target
fi

mlast=$(stat -c %Z $file)

while true; do
  mcur=$(stat -c %Z $file)
  if [[ ${mcur} != ${mlast} ]]; then
    if [[ ! -z $dbecho ]]; then
      echo "CHANGED - ${target}"
    fi
    echo "CHANGED - ${target}"
    mlast=${mcur}
    # doesn't work # runhook="python3 ./code/model.py"
    python3 ./code/model.py

    echo "WAITING - ${target}"
  else
    if [[ ! -z $dbecho ]]; then
      echo "SAME - ${target}"
    fi
  fi
  sleep 1
done
