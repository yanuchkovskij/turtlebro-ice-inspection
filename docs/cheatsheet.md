# Шпаргалка команд на соревнованиях

Всё выполняется на роботе по SSH (MobaXterm), рабочая папка — `~/hack`.

## Подключение

```bash
ssh pi@<IP-робота>            # пароль — с наклейки / от организаторов
cd ~/hack
```

Если MobaXterm портит вставку многострочных команд:

```bash
echo 'set enable-bracketed-paste off' >> ~/.inputrc   # переподключиться
```

## Перед попыткой

```bash
ros2 topic list                       # видны /odom /scan /imu /cmd_vel /front_camera/...
python3 nav.py pose                   # координаты и наклон
python3 nav.py scan                   # лидар: СПЕРЕДИ / СЛЕВА / СПРАВА / СЗАДИ
python3 calib.py show                 # калибровка в local_settings.json
ls route.json best_v2.onnx            # маршрут и модель на месте
```

## Маршрут

```bash
python3 teach.py                      # записать (w a s d, пробел, p — проезд, i — осмотр, q — сохранить)
cat route.json                        # посмотреть
python3 mission.py --dry              # прогон без распознавания
```

## Попытка

```bash
cd ~/hack && nohup python3 mission.py --nosnow > mission_out.log 2>&1 &
tail -f mission_out.log
```

Без `--nosnow` — с объездом сугробов по камере. `--direct` — только точки осмотра.

## Аварийный стоп

```bash
pkill -f mission.py
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"
```

## Результаты

```bash
ls -t logs/ | head -3                 # последние заезды
cat logs/run_*/mission.txt | tail -40 # лог последнего заезда
```

Скачать папку `logs/run_…` — через левую панель MobaXterm (summary.jpg, armature_N.jpg).

## Проверка распознавания на месте

```bash
python3 inspect_vision.py --ros       # табличка + манометр, кадр в last_vision.jpg
python3 aruco_detect.py --ros DICT_6X6_50
python3 animals_detect.py best_v2.onnx --ros
python3 snow.py --ros                 # сугроб впереди / свободно, кадр в last_snow.jpg
python3 inspect_vision.py logs/run_*/raw/*.jpg   # пересчитать по сохранённым кадрам
```

## Калибровка (если меняли колёса или робот «промахивается»)

```bash
python3 calib.py dist                 # путь → ODOM_SCALE
python3 calib.py rot                  # поворот → YAW_SCALE
python3 nav.py fwd 0.8                # проверка: должно выйти ~80 см
python3 nav.py rot 90
```

## Сбор кадров для датасета

```bash
python3 collect.py 0.5 bear           # кадр каждые 0.5 с в ~/hack/frames
```
