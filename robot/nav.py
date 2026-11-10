#!/usr/bin/env python3
"""Навигация TurtleBro: повороты, езда в точку с подруливанием,
объезд препятствий (лидар — стенки/стойки, камера — сугробы)
и выход из застревания (колёса крутятся, а робот стоит / наклонился).

Проверки (из ~/hack, робот на полу, вокруг свободно):
    python3 nav.py pose            # положение (x, y, курс) + наклон по IMU
    python3 nav.py scan            # расстояния вперёд/влево/вправо/назад + ближайшая точка
    python3 nav.py rot 90          # повернуться на 90° влево (минус — вправо)
    python3 nav.py fwd 0.5         # проехать 0.5 м вперёд (минус — назад)
    python3 nav.py goto 1.0 0      # в точку (1.0, 0) с объездом сугробов — поставь сугроб на пути
    python3 nav.py square 0.5      # квадрат 0.5 м и вывести ошибку возврата

В своём коде:
    from nav import Nav
    nav = Nav(camera=True)         # camera=True — видеть сугробы
    nav.go_to(1.2, 0.8)            # True — доехал, False — не смог (точку можно пропустить)
"""
import math
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Imu, LaserScan

# ---- Настройки (подбираются на полу) --------------------------------------
MAX_LIN = 0.26            # м/с (потолок робота ~0.30 на родных колёсах)
ODOM_SCALE = 1.0          # реальный путь / путь по одометрии (другие колёса). Мерить: calib.py dist
YAW_SCALE = 1.0           # реальный поворот / поворот по одометрии. Мерить: calib.py rot
MIN_LIN = 0.04            # ниже робот может не тронуться
MAX_ANG = 1.5             # рад/с
MIN_ANG = 0.25
ACCEL = 0.6               # м/с²
LIN_TOL = 0.02            # м — точность остановки
ANG_TOL = math.radians(2)  # точность поворота
K_LIN = 1.2               # замедление перед целью
K_ANG = 2.0               # П-регулятор поворота на месте
K_HEAD = 2.5              # П-регулятор удержания курса при езде
LIDAR_FRONT_DEG = 0.0     # «сырой» угол лидара, который смотрит вперёд робота (nav.py scan)
LIDAR_DIR = 1             # -1, если у лидара перепутаны лево и право
STOP_DIST = 0.25          # м — стенка/стойка в полосе движения ближе (от центра) -> объезд
SLOW_DIST = 0.45          # м — ближе этого к препятствию едем медленно
SLOW_LIN = 0.12           # м/с — скорость рядом с препятствием
ROBOT_FRONT = 0.10        # м — от центра робота до переднего края
ROBOT_HALF_W = 0.14       # м — половина ширины робота + запас (полоса, которую проверяем)
SIDE_CLEAR = 0.30         # м — сбоку ближе этого стенка -> в эту сторону не объезжаем
REVERSE_MAX = 0.35        # м — если точка сзади и ближе этого, отъезжаем задом без разворота
RATE = 20                 # Гц

# объезд и застревание
AVOID = True              # False — старое поведение: только стоп перед препятствием
DETOUR_TURN = 45          # ° — на сколько отвернуть от препятствия
DETOUR_STEP = 0.30        # м — сколько проехать вбок перед новой попыткой
MAX_DETOURS = 10          # попыток объезда на одну точку, потом точка пропускается
SIDE_SWITCH = 3           # столько неудачных объездов с одной стороны — пробуем другую
GOAL_CLOSE = 0.12         # м — «почти доехал»: при помехе считаем точку достигнутой
SNOW_IGNORE_NEAR = 0.08   # м — совсем у цели камеру не слушаем
STUCK_WIN = 1.2           # с — окно проверки застревания
STUCK_MIN_MOVE = 0.02     # м — за окно сдвинулись меньше, хотя едем -> упёрлись
SLIP_FRAC = 0.3           # колёса «проехали» d, а лидар изменился меньше 0.3·d -> буксуем
TILT_DEG = 5.0            # ° — наклон по IMU больше этого -> наехали на сугроб
BACKUP = 0.15             # м — отъезд назад при застревании

# защита координат от пробуксовки и привязка к стенкам полигона
SLIP_GUARD = True         # колёса насчитали путь, а лидар «стоит на месте» -> путь не засчитываем
WALL_FIX = True           # перед каждой точкой поправлять координаты по расстояниям до стенок
WALL_GATE = 0.35          # м — расхождение больше этого = луч упёрся не в стенку (стойка, человек)
WALL_MAX_FIX = 0.40       # м — больше за раз не поправляем
WALL_CONE = 2.0           # ° — ширина «луча» в сторону стенки


# ---- местные настройки робота: файл local_settings.json рядом со скриптами ----
# Обновления .py-файлов его не трогают, поэтому калибровка не слетает.
def _load_local(names):
    import json, os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_settings.json")
    try:
        data = json.load(open(p, encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:
        print("! local_settings.json не прочитан:", e, flush=True)
        return {}
    used = {k: v for k, v in data.items() if k in names}
    names.update(used)
    return used


LOCAL = _load_local(globals())

def norm(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_q(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def tilt_from_q(q):
    """Угол отклонения оси Z робота от вертикали, градусы."""
    if abs(q.w) + abs(q.x) + abs(q.y) + abs(q.z) < 1e-6:
        return None                                   # IMU не даёт ориентацию
    zz = 1 - 2 * (q.x * q.x + q.y * q.y)
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class StuckWatch:
    """Раз в STUCK_WIN секунд сравнивает, сколько реально проехали."""

    def __init__(self, nav):
        self.nav = nav
        self.reset()

    def reset(self):
        self.t, self.p, self.scan = time.time(), self.nav.pose()[:2], self.nav._scan_array()

    def check(self, v_cmd):
        """None или 'stuck' / 'slip'."""
        if time.time() - self.t < STUCK_WIN:
            return None
        px, py, _ = self.nav.pose()
        moved = math.hypot(px - self.p[0], py - self.p[1])
        scan = self.nav._scan_array()
        res = None
        if abs(v_cmd) > 0.06 and moved < STUCK_MIN_MOVE:
            res = "stuck"                              # колёса стоят — упёрлись
        elif moved > 0.08 and scan is not None and self.scan is not None \
                and len(scan) == len(self.scan):
            both = ~np.isnan(scan) & ~np.isnan(self.scan)
            # лучи вперёд/назад должны измениться почти на пройденное расстояние
            if both.sum() > 20 and \
                    np.percentile(np.abs(scan[both] - self.scan[both]), 80) < SLIP_FRAC * moved:
                res = "slip"                           # одометрия едет, а мир стоит
        self.t, self.p, self.scan = time.time(), (px, py), scan
        return res


class Nav(Node):
    def __init__(self, camera=False, snow_avoid=True):
        """camera — смотреть сугробы камерой; snow_avoid=False — сугробы не объезжаем
        вообще (большие колёса их переезжают): нет проверок наклона/застревания."""
        self.snow_avoid = snow_avoid
        self._last_raw = None      # для своей одометрии с поправками колёс
        if not rclpy.ok():
            rclpy.init()
        super().__init__("nav")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        # sensor_data (best effort) совместим и с reliable, и с best effort издателем
        self.create_subscription(Odometry, "/odom", self._on_odom, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Imu, "/imu", self._on_imu, qos_profile_sensor_data)
        self._odom = None          # (x, y, yaw) в системе одометрии
        self._scan = None
        self._tilt = None
        self._tilt0 = None
        self._tf = None            # перевод одометрии в координаты полигона
        self._v = 0.0
        self.obstacles = []        # где уже упирались: [(x, y), ...] — для выбора точек обзора
        self._acc = [0.0, 0.0, 0.0, 0.0]   # с прошлого скана: dx, dy, |путь|, |поворот|
        self._prev_scan = None
        self.slips = 0             # сколько раз ловили пробуксовку
        self.walls = None          # {"x_max":…, "x_min":…, "y_max":…, "y_min":…} от старта
        self._snow = None          # последний результат детектора сугробов
        self._snow_t = 0.0
        self.snow = None
        if camera:
            try:
                from snow import DriftDetector
                self.snow = DriftDetector()
                self.create_subscription(CompressedImage, "/front_camera/image_raw/compressed",
                                         self._on_cam, qos_profile_sensor_data)
            except Exception as e:      # без камеры тоже едем, просто без объезда сугробов
                print("! детектор сугробов не запущен:", e, flush=True)
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self)
        self._running = True
        self._thr = threading.Thread(target=self._spin, daemon=True)
        self._thr.start()
        try:
            self._wait_ready()
        except Exception:
            self.close()                       # без 'core dumped' при ошибке
            raise
        self._tilt0 = self._tilt
        self.set_start(0.0, 0.0, 0.0)  # по умолчанию: (0,0,0) = место запуска

    def _spin(self):
        while self._running and rclpy.ok():
            try:
                self._exec.spin_once(timeout_sec=0.1)
            except Exception:
                break

    def close(self):
        """Аккуратно остановить ROS (без 'terminate called... core dumped' на выходе)."""
        try:
            self.stop()
        except Exception:
            pass
        self._running = False
        self._thr.join(timeout=1.0)
        try:
            self._exec.shutdown()
            self.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()

    # ---- данные -----------------------------------------------------------
    def _on_odom(self, m):
        """Своя одометрия поверх платы: шаги пути умножаем на ODOM_SCALE,
        повороты — на YAW_SCALE (если колёса другого размера, плата врёт)."""
        p = m.pose.pose
        rx, ry, ryaw = p.position.x, p.position.y, yaw_from_q(p.orientation)
        if self._last_raw is None:
            self._last_raw = (rx, ry, ryaw)
            self._odom = (rx, ry, ryaw)
            return
        lx, ly, lyaw = self._last_raw
        self._last_raw = (rx, ry, ryaw)
        x, y, yaw = self._odom
        # шаг вдоль старого курса платы (знак — вперёд/назад)
        ds = (rx - lx) * math.cos(lyaw) + (ry - ly) * math.sin(lyaw)
        dyaw = norm(ryaw - lyaw) * YAW_SCALE
        mid = yaw + dyaw / 2
        sx, sy = ds * ODOM_SCALE * math.cos(mid), ds * ODOM_SCALE * math.sin(mid)
        self._odom = (x + sx, y + sy, yaw + dyaw)
        a = self._acc
        a[0] += sx
        a[1] += sy
        a[2] += abs(ds * ODOM_SCALE)
        a[3] += abs(dyaw)

    def _on_scan(self, m):
        """Пробуксовка: колёса проехали >= 5 см, а картинка лидара почти не изменилась
        относительно опорного скана -> этот путь не засчитываем."""
        r = np.asarray(m.ranges, np.float32)
        r = np.where(np.isfinite(r) & (r > m.range_min) & (r < m.range_max), r, np.nan)
        self._scan = m
        ref = self._prev_scan
        dx, dy, dist, turn = self._acc
        if not SLIP_GUARD or ref is None or len(ref) != len(r) or self._odom is None:
            self._prev_scan, self._acc = r, [0.0, 0.0, 0.0, 0.0]
            return
        if turn > math.radians(3):                    # повернулись — сравнивать нечего
            self._prev_scan, self._acc = r, [0.0, 0.0, 0.0, 0.0]
            return
        if dist < 0.05:
            return                                    # копим путь, опорный скан тот же
        self._prev_scan, self._acc = r, [0.0, 0.0, 0.0, 0.0]
        t = self.tilt()
        if t is not None and t > TILT_DEG:
            return                                    # на сугробе лидар видит пол — не судим
        both = ~np.isnan(r) & ~np.isnan(ref)
        if both.sum() < 30:
            return
        change = float(np.percentile(np.abs(r[both] - ref[both]), 80))
        if change < 0.3 * dist:                       # мир не сдвинулся — колёса буксуют
            x, y, yaw = self._odom
            self._odom = (x - dx, y - dy, yaw)
            self.slips += 1

    def _on_imu(self, m):
        self._tilt = tilt_from_q(m.orientation)

    def _on_cam(self, msg):
        now = time.time()
        if now - self._snow_t < 0.12:          # ~8 раз в секунду хватает
            return
        import cv2
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        r = self.snow.analyze(img)
        r.pop("small", None)
        r.pop("mask", None)
        self._snow, self._snow_t = r, now

    def _wait_ready(self, timeout=5.0):
        t0 = time.time()
        while self._odom is None or self._scan is None:
            if time.time() - t0 > timeout:
                miss = [n for n, v in (("/odom", self._odom), ("/scan", self._scan)) if v is None]
                raise RuntimeError(f"Нет данных из {miss}. Запущен ли ROS? "
                                   "sudo systemctl restart microros turtlebro")
            time.sleep(0.05)
        time.sleep(0.3)                        # дать прийти IMU

    def set_start(self, x, y, yaw_deg):
        """Сообщить, где робот стоит сейчас в координатах полигона."""
        ox, oy, oyaw = self._odom
        dyaw = math.radians(yaw_deg) - oyaw
        self._tf = (ox, oy, oyaw, x, y, dyaw)

    def pose(self):
        """(x, y, yaw) в координатах полигона; yaw в радианах."""
        ox0, oy0, _, x0, y0, dyaw = self._tf
        ox, oy, oyaw = self._odom
        dx, dy = ox - ox0, oy - oy0            # уже в реальных метрах (см. _on_odom)
        c, s = math.cos(dyaw), math.sin(dyaw)
        return (x0 + c * dx - s * dy, y0 + s * dx + c * dy, norm(oyaw + dyaw))

    def tilt(self):
        """Наклон относительно положения на старте, градусы (None — нет данных IMU)."""
        if self._tilt is None or self._tilt0 is None:
            return None
        return abs(self._tilt - self._tilt0)

    # ---- лидар ------------------------------------------------------------
    def _angles(self, m):
        """Углы лучей в системе робота: 0 — вперёд, +90 — влево, рад."""
        raw = m.angle_min + np.arange(len(m.ranges)) * m.angle_increment
        a = LIDAR_DIR * (raw - math.radians(LIDAR_FRONT_DEG))
        return np.arctan2(np.sin(a), np.cos(a))

    def sector(self, center_deg, width_deg=30):
        """Минимальное расстояние в секторе (0 = вперёд, 90 = влево). inf, если пусто."""
        m = self._scan
        if m is None:
            return float("inf")
        r = np.asarray(m.ranges, np.float32)
        ok = np.isfinite(r) & (r > m.range_min) & (r < m.range_max)
        d = np.abs(np.arctan2(np.sin(self._angles(m) - math.radians(center_deg)),
                              np.cos(self._angles(m) - math.radians(center_deg))))
        sel = ok & (d <= math.radians(width_deg) / 2)
        return float(r[sel].min()) if sel.any() else float("inf")

    def front(self, width_deg=30):
        return self.sector(0, width_deg)

    def _ray(self, world_deg, yaw):
        """Расстояние лидара в сторону мира world_deg (0 = +X старта), медиана узкого конуса."""
        m = self._scan
        if m is None:
            return None
        r = np.asarray(m.ranges, np.float32)
        ok = np.isfinite(r) & (r > m.range_min) & (r < m.range_max)
        want = math.radians(world_deg) - yaw
        d = np.abs(np.arctan2(np.sin(self._angles(m) - want), np.cos(self._angles(m) - want)))
        sel = ok & (d <= math.radians(WALL_CONE))
        return float(np.median(r[sel])) if sel.sum() >= 2 else None

    def init_walls(self):
        """На старте: запомнить, где стенки (вдоль осей старта). Робот стоит ровно."""
        time.sleep(0.4)
        x, y, yaw = self.pose()
        w = {}
        for key, deg, sign, base in (("x_max", 0, 1, x), ("y_max", 90, 1, y),
                                     ("x_min", 180, -1, x), ("y_min", 270, -1, y)):
            d = self._ray(deg, yaw)
            if d is not None and 0.12 < d < 4.0:
                w[key] = base + sign * d
        self.walls = w or None
        print("  стенки от старта:", {k: round(v, 2) for k, v in w.items()} or "не видны",
              flush=True)
        return self.walls

    def relocalize(self, verbose=True):
        """Поправить x, y по расстояниям до стенок. Робот должен стоять. -> сдвиг, м."""
        if not (WALL_FIX and self.walls):
            return 0.0
        t = self.tilt()
        if t is not None and t > TILT_DEG:
            return 0.0
        time.sleep(0.25)                               # свежий скан после остановки
        x, y, yaw = self.pose()
        est = {"x": [], "y": []}                       # (оценка координаты, прошла ли ворота)
        for key, deg in (("x_max", 0), ("y_max", 90), ("x_min", 180), ("y_min", 270)):
            if key not in self.walls:
                continue
            d = self._ray(deg, yaw)
            if d is None:
                continue
            wall = self.walls[key]
            pred = abs(wall - (x if key[0] == "x" else y))
            e = wall - d if key.endswith("max") else wall + d
            est[key[0]].append((e, abs(d - pred) <= WALL_GATE))
        limit = WALL_MAX_FIX
        out = {}
        for ax, cur in (("x", x), ("y", y)):
            c = est[ax]
            if len(c) == 2 and abs(c[0][0] - c[1][0]) < 0.08:
                out[ax] = (c[0][0] + c[1][0]) / 2      # обе стенки согласны — верим даже большой
                limit = 1.0
            else:
                ok = [e for e, g in c if g]
                if ok:
                    out[ax] = float(np.mean(ok))
        xs = [out["x"]] if "x" in out else []
        ys = [out["y"]] if "y" in out else []
        fx = (np.mean(xs) - x) if xs else 0.0
        fy = (np.mean(ys) - y) if ys else 0.0
        n = math.hypot(fx, fy)
        if n > limit:
            fx, fy = fx * limit / n, fy * limit / n
        if n < 0.01:
            return 0.0
        # сдвиг в мировых координатах -> в нашу одометрию
        dyaw = self._tf[5]
        c, s_ = math.cos(-dyaw), math.sin(-dyaw)
        ox, oy, oyaw = self._odom
        self._odom = (ox + c * fx - s_ * fy, oy + s_ * fx + c * fy, oyaw)
        if verbose and n > 0.03:
            print(f"  поправка по стенкам: {fx * 100:+.0f} / {fy * 100:+.0f} см", flush=True)
        return math.hypot(fx, fy)

    def corridor(self, backwards=False):
        """Ближайшее препятствие в полосе шириной с робота прямо по ходу (или сзади), м.
        Лидар RPLIDAR не видит ближе ~15 см от себя — поэтому тормозим заранее."""
        m = self._scan
        if m is None:
            return float("inf")
        r = np.asarray(m.ranges, np.float32)
        a = self._angles(m)
        ok = np.isfinite(r) & (r > m.range_min) & (r < m.range_max)
        fwd, lat = r * np.cos(a), r * np.sin(a)
        if backwards:
            fwd = -fwd
        sel = ok & (fwd > 0) & (np.abs(lat) < ROBOT_HALF_W)
        return float(fwd[sel].min()) if sel.any() else float("inf")

    def _scan_array(self):
        m = self._scan
        if m is None:
            return None
        r = np.asarray(m.ranges, np.float32)
        r[~np.isfinite(r) | (r <= m.range_min) | (r >= m.range_max)] = np.nan
        return r

    # ---- что мешает впереди ----------------------------------------------
    def snow_ahead(self):
        """Свежий результат камеры: сугроб в коридоре движения?"""
        r = self._snow
        if r is None or time.time() - self._snow_t > 0.6:
            return False
        return r["blocked"]

    def blocked(self, dist_goal):
        """Причина остановки или None. dist_goal — сколько осталось до цели."""
        t = self.tilt()
        on_drift = t is not None and t > TILT_DEG
        f = self.corridor()
        # препятствие дальше цели — не мешает; на сугробе робот наклонён и лидар
        # может «увидеть» пол — тогда лидару не верим (в режиме без объезда сугробов)
        if f < STOP_DIST and f - ROBOT_FRONT < dist_goal and \
                not (on_drift and not self.snow_avoid):
            return "lidar"
        if self.snow_avoid:
            if dist_goal > SNOW_IGNORE_NEAR and self.snow_ahead():
                return "snow"
            if on_drift:
                return "tilt"
        return None

    # ---- движение ---------------------------------------------------------
    def _send(self, v, w):
        t = Twist()
        t.linear.x = float(v) / ODOM_SCALE        # плата считает скорость по старому радиусу
        t.angular.z = float(w)
        self.pub.publish(t)

    def stop(self):
        self._v = 0.0
        for _ in range(3):
            self._send(0, 0)
            time.sleep(0.02)

    def rotate_to(self, yaw, timeout=15.0):
        """Повернуться на месте к курсу yaw (рад, координаты полигона)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            err = norm(yaw - self.pose()[2])
            if abs(err) < ANG_TOL:
                break
            w = clamp(K_ANG * err, -MAX_ANG, MAX_ANG)
            if abs(w) < MIN_ANG:
                w = math.copysign(MIN_ANG, w)
            self._send(0, w)
            time.sleep(1 / RATE)
        self.stop()
        return abs(norm(yaw - self.pose()[2])) < ANG_TOL * 2

    def rotate_to_deg(self, deg):
        return self.rotate_to(math.radians(deg))

    def rotate_by(self, deg):
        return self.rotate_to(norm(self.pose()[2] + math.radians(deg)))

    def forward(self, dist, speed=MAX_LIN, timeout=30.0, check_obstacles=True):
        """Проехать dist метров (минус — назад), удерживая текущий курс.
        Назад едем, только если сзади свободно (лидар). False — остановились раньше."""
        x0, y0, head = self.pose()
        sgn = 1 if dist >= 0 else -1
        tx, ty = x0 + dist * math.cos(head), y0 + dist * math.sin(head)
        self._v = 0.0
        t0 = time.time()
        watch = StuckWatch(self)
        while time.time() - t0 < timeout:
            x, y, yaw = self.pose()
            rem = sgn * ((tx - x) * math.cos(head) + (ty - y) * math.sin(head))
            if rem < LIN_TOL:
                self.stop()
                return True
            if self.snow_avoid and watch.check(sgn * self._v):
                self.stop()
                return False
            if check_obstacles:
                if sgn > 0 and self.blocked(rem):
                    self.stop()
                    return False
                if sgn < 0 and self.corridor(backwards=True) < STOP_DIST:
                    self.stop()
                    return False
            v_target = clamp(K_LIN * rem, MIN_LIN, speed)
            if self.corridor(backwards=sgn < 0) < SLOW_DIST:
                v_target = min(v_target, SLOW_LIN)
            self._v = min(v_target, self._v + ACCEL / RATE)
            w = clamp(K_HEAD * norm(head - yaw), -MAX_ANG, MAX_ANG)
            self._send(sgn * self._v, w)
            time.sleep(1 / RATE)
        self.stop()
        return False

    def go_to(self, x, y, speed=MAX_LIN, timeout=60.0, check_obstacles=True,
              max_detours=None, near_ok=GOAL_CLOSE):
        """Доехать до точки (x, y). Если мешает сугроб/стойка или застряли —
        объезжаем (до MAX_DETOURS раз). True — доехали, False — не смогли."""
        t_end = time.time() + timeout
        detours, side, side_tries = 0, None, 0
        fixes = 0
        self.relocalize()
        while time.time() < t_end:
            px, py, yaw = self.pose()
            d = math.hypot(x - px, y - py)
            if d < LIN_TOL:
                # доехали по одометрии — сверяемся со стенками, при заметной ошибке доезжаем
                if fixes < 2 and self.relocalize() > 0.05:
                    fixes += 1
                    continue
                return True
            bearing = math.atan2(y - py, x - px)
            if d < REVERSE_MAX and abs(norm(bearing - yaw)) > math.radians(120):
                self.rotate_to(norm(bearing + math.pi))
                return self.forward(-d, speed, check_obstacles=check_obstacles)
            self.rotate_to(bearing)
            res = self._drive_to(x, y, speed, max(1.0, t_end - time.time()), check_obstacles)
            if res is True:
                if fixes < 2 and self.relocalize() > 0.05:
                    fixes += 1
                    continue
                return True
            self._remember(res)
            px, py, _ = self.pose()
            if math.hypot(x - px, y - py) < near_ok:
                return True                       # почти на месте — хватит
            if res == "timeout" or not (AVOID and check_obstacles):
                return False
            detours += 1
            print(f"  ! {self._reason_ru(res)} — объезд {detours}/{MAX_DETOURS}", flush=True)
            if detours > (MAX_DETOURS if max_detours is None else max_detours):
                return False
            if res in ("stuck", "slip", "tilt"):
                self.forward(-BACKUP, speed=0.15)       # сначала слезаем назад
            if side is None:
                side, side_tries = self._choose_side(), 0
            moved = self._detour(side)
            side_tries += 1
            if not moved or side_tries >= SIDE_SWITCH:      # с этой стороны тупик
                side, side_tries = -side, 0
                print(f"  ! пробую объезд с {'левой' if side > 0 else 'правой'} стороны",
                      flush=True)
        return False

    def _remember(self, res):
        """Запомнить, где впереди помеха."""
        ahead = {"snow": 0.30, "lidar": 0.25, "stuck": 0.15, "slip": 0.15, "tilt": 0.10}.get(res)
        if ahead is None:
            return
        px, py, yaw = self.pose()
        for lat in (-0.12, 0.0, 0.12):
            self.obstacles.append((px + ahead * math.cos(yaw) - lat * math.sin(yaw),
                                   py + ahead * math.sin(yaw) + lat * math.cos(yaw)))

    def near_obstacle(self, x, y, r=0.25):
        return any(math.hypot(x - ox, y - oy) < r for ox, oy in self.obstacles)

    @staticmethod
    def _reason_ru(res):
        return {"lidar": "впереди препятствие (лидар)", "snow": "впереди сугроб (камера)",
                "tilt": "наклон — наехали на сугроб", "stuck": "упёрлись, не едем",
                "slip": "колёса буксуют"}.get(res, res)

    def _choose_side(self):
        """+1 — объезжать слева, -1 — справа. Где свободнее по лидару и меньше сугроба."""
        left = min(self.sector(60, 50), 1.5)
        right = min(self.sector(-60, 50), 1.5)
        r = self._snow if (self._snow and time.time() - self._snow_t < 0.6) else None
        if r:
            left -= 2.0 * r["left"]
            right -= 2.0 * r["right"]
        # сторона, где вплотную стенка/стойка, проигрывает в любом случае
        if self.sector(70, 70) < SIDE_CLEAR:
            left -= 10
        if self.sector(-70, 70) < SIDE_CLEAR:
            right -= 10
        return 1 if left >= right else -1

    def _detour(self, side):
        """Отвернуть от препятствия и проехать немного в сторону."""
        for turn in (DETOUR_TURN, 30, 30):
            self.rotate_by(side * turn)
            if self.corridor() < STOP_DIST + 0.05:   # после поворота прямо стенка — дальше
                continue
            if self.forward(DETOUR_STEP):
                return True
        return False

    def _drive_to(self, x, y, speed, timeout, check_obstacles):
        """Езда к точке с подруливанием. True или причина остановки:
        'lidar' / 'snow' / 'tilt' / 'stuck' / 'slip' / 'timeout'."""
        px0, py0, _ = self.pose()
        L = math.hypot(x - px0, y - py0) or 1e-6
        ux, uy = (x - px0) / L, (y - py0) / L      # исходное направление на цель
        self._v = 0.0
        t0 = time.time()
        watch = StuckWatch(self)
        while time.time() - t0 < timeout:
            px, py, yaw = self.pose()
            dx, dy = x - px, y - py
            rem = dx * ux + dy * uy                 # остаток пути вдоль исходного направления
            if rem < LIN_TOL:
                self.stop()
                return True
            if check_obstacles:
                why = self.blocked(math.hypot(dx, dy))
                if why:
                    self.stop()
                    return why
            stuck = watch.check(self._v) if self.snow_avoid else None
            if stuck:
                self.stop()
                return stuck
            # рулим на саму точку, а у самой цели — по исходному курсу (чтобы не крутиться)
            target = math.atan2(dy, dx) if math.hypot(dx, dy) > 0.10 else math.atan2(uy, ux)
            err = norm(target - yaw)
            if abs(err) > math.radians(30):         # сильно сбились — довернуть на месте
                self.stop()
                self.rotate_to(target)
                watch.reset()
                continue
            v_target = clamp(K_LIN * rem, MIN_LIN, speed) * max(0.4, math.cos(err))
            if self.corridor() < SLOW_DIST:
                v_target = min(v_target, SLOW_LIN)
            self._v = min(v_target, self._v + ACCEL / RATE)
            w = clamp(K_HEAD * err, -MAX_ANG, MAX_ANG)
            self._send(self._v, w)
            time.sleep(1 / RATE)
        self.stop()
        return "timeout"


# ---- проверки из командной строки -------------------------------------------
def _fmt(p):
    return f"x={p[0]:+.3f} y={p[1]:+.3f} курс={math.degrees(p[2]):+.1f}°"


if __name__ == "__main__":
    args = sys.argv[1:] or ["pose"]
    cmd = args[0]
    nav = Nav(camera=(cmd == "goto"))
    try:
        if cmd == "pose":
            while True:
                t = nav.tilt()
                print(_fmt(nav.pose()), "| наклон", "нет IMU" if t is None else f"{t:.1f}°",
                      flush=True)
                time.sleep(0.3)
        elif cmd == "scan":
            print("Поставь бутылку в 20 см спереди, потом слева — смотри, где она.\n"
                  f"Сейчас LIDAR_FRONT_DEG={LIDAR_FRONT_DEG}, LIDAR_DIR={LIDAR_DIR} "
                  f"({'из local_settings.json' if 'LIDAR_FRONT_DEG' in LOCAL else 'по умолчанию'})",
                  flush=True)
            while True:
                m = nav._scan
                r = np.asarray(m.ranges, np.float32)
                ok = np.isfinite(r) & (r > m.range_min) & (r < m.range_max)
                i = int(np.argmin(np.where(ok, r, np.inf)))
                raw = math.degrees(m.angle_min + i * m.angle_increment)
                rob = math.degrees(nav._angles(m)[i])
                where = ("СПЕРЕДИ" if abs(rob) <= 45 else "СЗАДИ" if abs(rob) >= 135
                         else "СЛЕВА" if rob > 0 else "СПРАВА")
                print(f"ближайшее {r[i]:.2f} м -> {where:7s} (угол робота {rob:+4.0f}°, "
                      f"сырой {raw:+4.0f}°) | вперёд {nav.sector(0):.2f} влево {nav.sector(90):.2f} "
                      f"вправо {nav.sector(-90):.2f} назад {nav.sector(180):.2f}", flush=True)
                time.sleep(0.5)
        elif cmd == "rot":
            nav.rotate_by(float(args[1]))
            print(_fmt(nav.pose()))
        elif cmd == "fwd":
            ok = nav.forward(float(args[1]))
            print("готово" if ok else "остановлен", _fmt(nav.pose()))
        elif cmd == "square":
            side = float(args[1]) if len(args) > 1 else 0.5
            for k in range(4):
                nav.forward(side)
                nav.rotate_to_deg(90 * (k + 1))
            p = nav.pose()
            print("Квадрат готов.", _fmt(p),
                  f"| ошибка возврата {math.hypot(p[0], p[1]) * 100:.1f} см")
        elif cmd == "goto":
            ok = nav.go_to(float(args[1]), float(args[2]))
            print("готово" if ok else "не доехал", _fmt(nav.pose()))
        else:
            print(__doc__)
    except KeyboardInterrupt:
        pass
    finally:
        nav.close()
