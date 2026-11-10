# extras — запуск без SSH (экспериментально)

На соревнованиях **не использовалось**: заезд запускали по SSH через `nohup`. Оставлено как задел.

- `launcher.py` — ждёт старта и запускает миссию: ладонь перед носом робота (лидар, 5–15 см) на ~2 с
  или кнопка D22–D25 Arduino-части платы. Светодиоды D26–D29: готов / едет / финиш / ошибка.
- `arduino_buttons.py` — чтение кнопок и управление светодиодами по последовательному порту (без pyserial).
- `buttons/buttons.ino` — скетч для Arduino (на соревнованиях не прошит).
- `launcher.sh`, `hack-mission.service` — автозапуск `launcher.py` как сервиса systemd.

```bash
cp extras/launcher.py extras/arduino_buttons.py ~/hack/
cd ~/hack && python3 launcher.py --nosnow
```
