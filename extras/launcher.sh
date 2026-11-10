#!/bin/bash -i
# Запуск launcher.py с тем же окружением ROS, что и в SSH (bash -i читает ~/.bashrc).
# Флаги миссии — в строке ниже (сейчас: без объезда сугробов, с распознаванием).
cd /home/pi/hack
sleep 15                      # дать ROS (microros, turtlebro) подняться после включения
exec python3 launcher.py --nosnow >> /home/pi/hack/launcher.log 2>&1
