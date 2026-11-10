// Кнопки TurtleBro V1.3A (часть Arduino Mega 2560) -> сообщения в USB-порт для Raspberry Pi.
// Кнопки: D22..D25. Светодиоды: D26..D29 (Pi включает: '0'..'3' — вкл, 'a'..'d' — выкл).
// Сообщение при любом изменении кнопки: "BTN 22 0" / "BTN 22 1" (полярность не важна).

const int BTN[] = {22, 23, 24, 25};
const int LED[] = {26, 27, 28, 29};
const int N = 4;
int state[N];

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < N; i++) {
    pinMode(BTN[i], INPUT_PULLUP);
    state[i] = digitalRead(BTN[i]);
    pinMode(LED[i], OUTPUT);
    digitalWrite(LED[i], LOW);
  }
  // приветствие: пробежать огоньками
  for (int i = 0; i < N; i++) { digitalWrite(LED[i], HIGH); delay(120); digitalWrite(LED[i], LOW); }
  Serial.println("READY");
}

void loop() {
  for (int i = 0; i < N; i++) {
    int v = digitalRead(BTN[i]);
    if (v != state[i]) {
      delay(20);                              // дребезг
      v = digitalRead(BTN[i]);
      if (v != state[i]) {
        state[i] = v;
        Serial.print("BTN ");
        Serial.print(BTN[i]);
        Serial.print(" ");
        Serial.println(v);
      }
    }
  }
  while (Serial.available()) {
    char c = Serial.read();
    if (c >= '0' && c <= '3') digitalWrite(LED[c - '0'], HIGH);
    else if (c >= 'a' && c <= 'd') digitalWrite(LED[c - 'a'], LOW);
    else if (c == 'x') for (int i = 0; i < N; i++) digitalWrite(LED[i], LOW);
    else if (c == '?') Serial.println("READY");
  }
}
