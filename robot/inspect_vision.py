#!/usr/bin/env python3
"""Табличка (символьный код) и манометр (цвет сектора под стрелкой) — OpenCV, без нейросетей.

Проверка на фото (ноутбук или робот):
    python3 inspect_vision.py logs/armature_*.jpg
    -> печатает результат, рядом сохраняет *_vision.jpg с рамками

Живая камера (робот):
    python3 inspect_vision.py --ros          # пишет результат и кадр last_vision.jpg

В своём коде:
    from inspect_vision import read_plate, read_gauge, draw_results
    plate = read_plate(img, exclude=[(x1,y1,x2,y2), ...])   # exclude — рамки ArUco
    gauge = read_gauge(img)
    plate["text"] -> "А";  gauge["color"] -> "зеленый"
"""
import base64
import glob
import math
import sys

import cv2
import numpy as np

# ---- настройки ---------------------------------------------------------------
PLATE_MIN_SCORE = 0.75     # ниже — буква не считается распознанной
PLATE_ALPHABET = "АБВ"     # по регламенту на табличках только эти буквы
GAUGE_MID_DEG = 7          # стрелка ближе этого к границе цветов (в градусах) -> "посередине"
COLOR_NAMES = {"green": "зеленый", "red": "красный", "mid": "посередине"}

try:                       # местные настройки (PLATE_MIN_SCORE, GAUGE_MID_DEG) из local_settings.json
    import json as _json
    import os as _os
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "local_settings.json")
    _d = _json.load(open(_p, encoding="utf-8"))
    PLATE_MIN_SCORE = float(_d.get("PLATE_MIN_SCORE", PLATE_MIN_SCORE))
    GAUGE_MID_DEG = float(_d.get("GAUGE_MID_DEG", GAUGE_MID_DEG))
except Exception:
    pass

_TPL_CHARS = "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ0123456789АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ0123456789АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ0123456789"
_TPL_N = 114
_TPL_B64 = (
    "0tR5P7A9ZD+wPWQ/PuQyP1aURj8lvFE/eKOpP83MTD+CEl4/95BLP6uqaj/MioE/6vxaPwAAgD/q/Fo/JbxRP/HwcD+CEl4/ep5n"
    "Pw8Pjz8MlHA/ZAI9P+r8Wj9+7aE/s1+IP+vqaj8nJ6c/dX5tP6+uLj+nzi8/DMMwP7i3Nz/3kEs/nuc5P7KxMT+nzi8/uLc3P7Kx"
    "MT/d04g/+s1WP/rNVj8aYTk/kZpgPxphOT9VVdU/fLxHP7N1aD+2Dn0/Iyx3PxUvkD+zdWg/AACAP7N1aD/6zVY/Xc1VP5BJcT+2"
    "Dn0/4GydP0rxgj/VBGM/IJNiP7/Zuj+fQ6o/b3pTP+dqrj+NsFw/1lxNPxphOT9f9zo/m6s5P/rNVj8XuEA/fLxHP/c0Qj9P7EQ/"
    "T+xEPxdddD8AAEA/AABAP3TRJT/T+Eg/dNElP0YXvT/qTS8/XXRRP7roYj9GF10/XXSBP110UT+GLGQ/XXRRPwAAQD+nNz0/0UVX"
    "PwAAYD9GF40/L7poP09dSz/poks/0UWnPxofmT+nNz0/etObP3TRRT+c3jQ/uugiP1D6JD/f9CY/uuhCP6uqKj9BCzQ/o4suP+pN"
    "Lz9BCzQ/AAAAAAAP4AAAH/AAAB/wAAAf8AAAP/gAAD/4AAB++AAAfvwAAHx8AAD8fAAA/H4AAPg+AAH4PwAB+D8AAfAfAAPwH4AD"
    "8B+AB/AfgAf//8AH///AD///wA///+APwAfgH4AH4B+AA/AfgAPwPwAD+D8AAfg/AAH4fgAA/AAAAAAAAAAAP///gD///4A///+A"
    "P///gD///4A/AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/AAAAP//+AD///4A////gP///4D////A/AAfwPwAD+D8AA/g/AAH4PwAB"
    "+D8AA/g/AAP4PwAP8D////A////gP///wD///wA///wAAAAAAAAAAAA///gAP//+AD///4A////AP///wD8AH+A/AA/gPwAH4D8A"
    "B+A/AA/gPwAPwD8AP8A///+AP//+AD///gA////AP///4D8AB/A/AAPwPwAD+D8AA/g/AAP4PwAD+D8AA/A/AAfwP///8D///+A/"
    "///AP///AD///AAAAAAAAAAAAAf//8AH///AB///wAf//8AH///AB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAA"
    "B+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAB+AAAAAAAAAAAAAAAB//"
    "AAAf/wAAH/8AAB//AAA+HwAAPh8AAD4fAAA8HwAAPB8AADwfAAB8HwAAfB8AAHwfAAB4HwAA+B8AAPgfAAD4HwAB8B8AAfAfAA//"
    "/+AP///gD///4A///+APAAHgDwAB4A8AAeAPAAHgDwAB4A8AAeAPAAHgAAAAAAAAAAAf///gH///4B///+Af///gH///4B+AAAAf"
    "gAAAH4AAAB+AAAAfgAAAH4AAAB+AAAAf//+AH///wB///8Af///AH///wB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAB+AAAAfgAAA"
    "H///8B////Af///wH///8B////AAAAAAAAAAAAAAAAAAAAAAAAAAAD4DwHwfA8B4HwPA+A+DwfAHg8HwB8PD4APjw8AD48fAAfPP"
    "gAD7/wAA//8AAP//AAH//4AB48fAA+PHwAfDw+AHw8HgD4PB8A+DwPAfA8D4HwPAfD4DwHx8A8A+AAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAf4AAA//gAAf/+AAP//wAH//8AD+A/gA/AH4AfgB+AD4AfgAAAH4AAAB+AAAA/AAAP/gAAD/wAAA/wAAAP/AAAD/8AA"
    "AB/gAAAH4AAAA/AAQAPwA8AD8A/gA/AH8AfwB/gP4AP//+AB///AAP//gAB//gAAD/gAAAAAAAAAAAAfgAf4H4AH+B+AD/gfgA/4"
    "H4Af+B+AH/gfgD/4H4B/+B+AffgfgP34H4D5+B+B+fgfgfH4H4Px+B+D4fgfh+H4H4/B+B+Pgfgfn4H4H58B+B+/AfgfvgH4H/4B"
    "+B/8Afgf/AH4H/gB+B/wAfgf8AH4H+AB+B/gAfgAAAAAAAAAAA/AD8APwB/AD8AfgA/APwAPwH8AD8B+AA/A/AAPwPwAD8H4AA/D"
    "8AAPx/AAD8/gAA//wAAP/8AAD//AAA//4AAP9+AAD8PwAA/D+AAPwfgAD8H8AA/A/gAPwH4AD8B/AA/APwAPwD+AD8AfwA/AD8AP"
    "wA/gD8AH4AAAAAAAAAAAAH///AB///wAf//8AH///AB///wAfgD8AP4A/AD+APwA/AD8APwA/AD8APwA/AD8APwA/AH8APwB/AD8"
    "AfgA/AH4APwB+AD8AfgA/AP4APwD+AD8A/AA/AfwAPwH8AD8D+AA/D/gAPw/4AD8P8AA/D+AAPw+AAD4AAAAAAAAAAB/gAH+f8AD"
    "/n/AA/5/wAP+f+AH/n/gB/5/4Af+f/AP/n3wD7598A++fPgPPnz4H358+B9+fHwefnx8Pn58fD5+fDw8fnw+fH58Pnx+fD58fnwf"
    "eH58H/h+fB/4fnwP8H58D/B+fA/wfnwH4H58B+B+fAfgfnwDwH4AAAAAAAAAAB+AAfgfgAH4H4AB+B+AAfgfgAH4H4AB+B+AAfgf"
    "gAH4H4AB+B+AAfgfgAH4H4AD+B////gf///4H///+B////gf///4H4AD+B+AAfgfgAH4H4AB+B+AAfgfgAH4H4AB+B+AAfgfgAH4"
    "H4AB+B+AAfgfgAH4H4AB+AAAAAAAAAAAAA/wAAB//gAB//+AA///wAf//+AP8A/wH+AH+B/AA/g/gAH8PwAA/D8AAPw/AAD8fwAA"
    "/n8AAP5+AAB+fgAAfn8AAP5/AAD+PwAA/D8AAPw/gAD8P4AB/B/AA/gf4Af4D/AP8Af//+AD///AAf//gAB//gAAD/AAAAAAAAAA"
    "AAAf///4H///+B////gf///4H///+B+AAfgfgAH4H4AB+B+AAfgfgAH4H4AB+B+AAfgfgAH4H4AB+B+AAfgfgAH4H4AB+B+AAfgf"
    "gAH4H4AB+B+AAfgfgAH4H4AB+B+AAfgfgAH4H4AB+B+AAfgfgAH4H4AB+B+AAfgAAAAAAAAAAB//+AAf//8AH///gB///8Af///g"
    "H4Af4B+AB/AfgAfwH4AH8B+AA/AfgAPwH4AH8B+AB/AfgA/gH8A/4B///8Af//+AH///AB///AAf/4AAH4AAAB+AAAAfgAAAH4AA"
    "AB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAAAAAAAAAAAAAA/4AAB//wAA///AA///4Af///AH+A/wD+AD+A/AAfgfwADwH4AAAB+A"
    "AAAfgAAAP4AAAD8AAAA/AAAAPwAAAD8AAAA/gAAAH4AAAB+AAAAfgADAH8AA+A/gAfgP8AP4B/gP8AP///AB///gAP//gAA//wAA"
    "D/gAAAAAAAAAAAAf///4H///+B////gf///4H///+AAH4AAAB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAB+AA"
    "AAfgAAAH4AAAB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAAAAAAAAAAD+AA/gfgAPwH8AD"
    "8A/AB+AH4AfgB/APwAPwD8AD+B+AAfgfgAD8HwAA/D8AAH4+AAB+fgAAP3wAAB/8AAAf/AAAD/gAAA/4AAAH8AAAB/AAAAfgAAAH"
    "4AAAD8AAAB/AAAf/gAAH/4AAB/8AAA/+AAAP/AAAA/AAAAAAAAAAAAAAAAAAAAAHwAAAB8AAAAfgAAH//wAH///gD///8B////g/"
    "h8H4PwfA/H4HwHx+B8B+fgfAfn4HwH5+B8B+fgfAfn4HwHw+B8B8PwfA/B+Hwfgf///4D///8Af//8AA//8AAAfgAAAHwAAAB8AA"
    "AAfAAAAAAAAAAAAAAAAAAAAAAAAfwAP4D8AD8AfgB/AH8A/gA/APwAH4H8AB+B+AAPw/AAB+fwAAfn4AAD/8AAA//AAAH/gAAA/4"
    "AAAP8AAAH/gAAB/8AAA//AAAf/4AAH5+AAD+PwAA/D+AAfgfgAP4H8AD8A/gB+AH4A/gB/APwAPwH8AB+D+AAfwAAAAAAAAAAAfA"
    "HwAHwB8AB8AfAAfAHwAHwB8AB8AfAAfAHwAHwB8AB8AfAAfAHwAHwB8AB8AfAAfAHwAHwB8AB8AfAAfAHwAHwB8AB8AfAAfAHwAH"
    "///gB///4Af//+AH///gAAAB4AAAAeAAAAHgAAAB4AAAAeAAAAHgAAAB4AAAAAAAAAAAH4AD+B+AA/gfgAP4H4AD+B+AA/gfgAP4"
    "H4AD+B+AA/gfgAP4H4AD+B+AA/gfgAP4H8AD+B/AA/gP4AP4D/x/+Af///gH///4A///+AD/4/gAAAP4AAAD+AAAA/gAAAP4AAAD"
    "+AAAA/gAAAP4AAAD+AAAA/gAAAP4AAAAAAAAAAAAAAAAAAAAAAAAAAB8A8A+fAPAPnwDwD58A8A+fAPAPnwDwD58A8A+fAPAPnwD"
    "wD58A8A+fAPAPnwDwD58A8A+fAPAPnwDwD58A8A+fAPAPnwDwD58A8A+fAfAPn////5////+f////n////4AAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAB4D4HweA+B8HgPgfB4D4HweA+B8HgPgfB4D4HweA+B8HgPgfB4D4HweA+B8HgPgfB4D4HweA+B8HgPgfB4D4Hw"
    "eA+B8HgPgfB////+f////n////5////+AAAAHgAAAB4AAAAeAAAAHgAAAB4AAAAeAAAAAAAAAAAAAAAAAB/wAAD//gAD//+AB///"
    "wA///+AP8B/gH8AH8B+AA/APAAP4AAAB+AAAAfgAAAH8AA///AAP//wAD//8AA///AAP//wAAAH8AAAB/AAAAfgHAAH4HwAD+D+A"
    "A/AfwAfwD/Af4A///8AH//+AA///AAD//AAAH/AAAAAAAAAAAAAAAAAAAAAAAAAAAAB4AH8AfAH/wHwH//B8D//4fA/B+HwfAHx8"
    "HwB8fD4APHw+AD5//gA+f/4APn/+AD5//gA+fD4APnw+AD58HgA8fB8AfHwfAPx8D8H4fA//8HwH/+B8Af/AeAB/AAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAA///wA///8A////Af///wH///8D/AB/A/gAfwPwAH8D8AB/A/AAfwPwAH8D8AB/A/gAfwH+AH8B///"
    "/AP///wA///8AD///AB///wAfwH8AP4B/AH8AfwB/AH8A/gB/APwAfwH8AH8D+AB/A/gAfwfwAH8P4AB/AAAAAAAAAAAAAfgAAAf"
    "+AAAP/4AAH/+AAD//wAA/B8AAfgfgAH4D4AB8A+AAfAPwAPwD8AD8A/AA/APwAPwD8AD8A/AA/APwAPwD8AD8A/AA/APwAHwD8AB"
    "8A+AAfAPgAH4D4AB+B+AAPw/AAD//wAAf/4AAD/8AAAf+AAAB+AAAAAAAAAAAAAAD+AAAB/gAAB/4AAB/+AAA//gAAf34AAHx+AA"
    "B4fgAAYH4AAAB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAB+AAAAfgAAAH4AAAB+AAAA/g"
    "AAf//8AH///AB///wAf//8AAAAAAAAAAAAAP8AAAf/wAAP/+AAH//wAD//+AA/g/gAPwH4AH8B+AACAfgAAAH4AAAB+AAAA/gAAA"
    "PwAAAH8AAAD+AAAD/AAAB/gAAA/wAAAf4AAAP8AAAH8AAAD+AAAB/AAAA/gAAAPwAAAH///AB///wAf//8AH///AB///wAAAAAAA"
    "AAAAAA/wAAA//gAAf/8AAP//gAH//4AD+B/AA/APwAPwD8AAAA/AAAAPwAAAH4AAAD+AAAf/AAAH/gAAB/gAAAf+AAAH/4AAAD/A"
    "AAAPwAAAD+AAAAfgAOAH4AfwB+AD8A/gA/gfwAH//8AB//+AAP//AAA//gAAD/AAAAAAAAAAAAAAAH8AAAD/AAAB/wAAAf8AAAP/"
    "AAAH/wAAB/8AAA+/AAAfPwAAHz8AAD4/AAB8PwAAfD8AAPg/AADwPwAB8D8AA+A/AAPAPwAHwD8AD8B/AA////AP///wD///8A//"
    "//AAAD8AAAA/AAAAPwAAAD8AAAA/AAAAPwAAAAAAAAAAAAH//4AB//+AAf//gAH//4AB+AAAAfAAAAHwAAAB8AAAAfAAAAHwAAAD"
    "8/gAA//+AAP//wAD//+AA///wAP4H8AD8A/AAAAP4AAAB+AAAAfgAAAH4AAAB+AH4A/AA/APwAP4H8AD//+AAf//AAD//gAAf/wA"
    "AA/wAAAAAAAAAAAAAAfwAAAf/AAAf/4AAP//AAD//wAB+D+AAfgfgAPwAAAD8AAAA+AAAAPgAAAH4/gAB+/8AAf//wAH//8AB/x/"
    "gAfwH4AH8B/AB+APwAfgD8AD4A/AA/APwAPwD8AD8B+AAfg/gAH//4AA//8AAH/+AAA//AAAD/AAAAAAAAAAAAAH///AB///wAf/"
    "/8AH///AB///wAAAD4AAAB+AAAA/AAAAPgAAAH4AAAD8AAAA/AAAAfgAAAHwAAAD8AAAB+AAAAfgAAAP4AAAD8AAAA/AAAAfgAAA"
    "H4AAAB+AAAA/gAAAPwAAAD8AAAA/AAAAPwAAAD8AAAB/AAAAAAAAAAAAAAAP8AAAf/4AAP//AAH//4AB/D+AA/gfwAPwD8AD8A/A"
    "A/APwAPwD8AB+B+AAfgfgAD//wAAf/4AAD/4AAD//wAB/n+AA/gfwAPwD8AD8A/AB+AP4AfgD+AH8A/gA/APwAPwD8AD/D/AAf//"
    "gAD//wAAf/4AAA/wAAAAAAAAAAAAAB/gAAB/+AAA//wAAf/+AAP//wAD+D8AB/AfgAfgH4AH4B+AB+AfgAfgD4AH4B/AB+AfwAPw"
    "H8AD+H/AA///wAH//8AA/+/AAD+PwAAAD4AAAA+AAAAfgABgH4AD8D8AA/B/AAP//gAB//4AAP/8AAB/8AAAH8AAAAAAAAAAAAAA"
    "AAAAAB/4AAAf+AAAH/gAAD/8AAA//AAAP/wAAH/+AAB//gAA/n4AAP5/AAD+fwAB/D+AAfw/gAH8P4AD+B/AA/gfwAf4H8AH8A/g"
    "B///4A////AP///wD///8B////gfwAP4P8AD+D+AAfw/gAH8f4AB/gAAAAAAAAAAAAAAAB///8Af///AH///wB///8Af///AH+AA"
    "AB/gAAAf4AAAH+AAAB/gAAAf4AAAH+AAAB//8AAf//4AH///gB///8Af///gH+B/4B/gH+Af4A/wH+AP8B/gD/Af4A/wH+Af4B/g"
    "P+Af///gH///wB///4Af//4AH//4AAAAAAAAAAAAH//wAB///gAf//8AH///gB///8Af4H/AH+A/wB/gH8Af4B/AH+AfwB/gP8Af"
    "4H/AH///gB///gAf//8AH///gB///8Af4D/gH+Af4B/gD+Af4A/wH+AP8B/gD/Af4B/gH+B/4B///+Af///AH///gB///wAf//gA"
    "AAAAAAAAAAAH///gB///4Af//+AH///gB///4Af//+AH+AAAB/gAAAf4AAAH+AAAB/gAAAf4AAAH+AAAB/gAAAf4AAAH+AAAB/gA"
    "AAf4AAAH+AAAB/gAAAf4AAAH+AAAB/gAAAf4AAAH+AAAB/gAAAf4AAAH+AAAB/gAAAf4AAAAAAAAAAAAAAD//4AA//+AAP//gAD/"
    "/4AA//+AAPwfgAD8H4AA/B+AAPwfgAD8H4AB/B+AAfwfgAH8H4AB/B+AAfwfgAH4H4AB+B+AAfgfgAP4H4AH+B/AH///+B////gf"
    "///4H///+B////gfAAD4HwAA+B8AAPgfAAD4HwAA+AAAAAAAAAAAB///wAf//8AH///AB///wAf//8AH///AB/gAAAf4AAAH+AAA"
    "B/gAAAf4AAAH//+AB///gAf//4AH//+AB///gAf//4AH+AAAB/gAAAf4AAAH+AAAB/gAAAf4AAAH+AAAB///4Af//+AH///gB///"
    "4Af//+AH///gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+A8B8HwPA+A+DwfAHw8PgA+PHwAHzz4AA+98AAP//AAD/"
    "/wAB//+AA///wAPn58AHx+PgD4PB8A+DwfAfA8D4PgPAfD4DwHwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/4AAH"
    "//wAB///AAf//4AH///ABwB/wAAAP8AAAB/AAAAfwAAAH8AAAD+AAAD/gAB//wAAf/wAAH/+AAB//4AAf/+AAAB/wAAAH8AAAB/g"
    "AAAf4AAAH+AAAB/gCAA/wA8Af8AP///AD///gA///wAP//wAAP/gAAAAAAAAAAAAP4AP+D+AH/g/gB/4P4A/+D+AP/g/gH/4P4B/"
    "+D+A//g/gP/4P4H/+D+B//g/g/f4P4P3+D+H5/g/h+f4P4/H+D+Px/g/n4f4P7+H+D+/B/g//wf4P/4H+D/+B/g//Af4P/wH+D/4"
    "B/g/8Af4P/AH+D/gB/g/4Af4AAAAAAAAAAB/gAf8f4AP+H+AH/B/gD/gf4B/wH+A/4B/gf8Af4P+AH+H/AB/j/gAf5/wAH+/4AB/"
    "/8AAf//gAH//8AB///AAf//4AH///AB/8/4Af+H+AH/B/wB/gP+Af4B/gH+Af8B/gD/gf4Af8H+AD/B/gA/4f4AH/H+AA/wAAAAA"
    "AAAAAAD///wA///8AP///AD///wA///8AP///AD/A/wA/wP8AP8D/AD/A/wA/wP8AP8D/AD/A/wA/wP8AP8D/AD/A/wA/wP8AP8D"
    "/AD+A/wA/gP8Af4D/AH+A/wD/gP8D/wD/H/8A/x/+AP8f/AD/H/gA/x/AAP8eAAD/AAAAAAAAAAAAAAAAH/AA/5/wAf+f+AH/n/g"
    "B/5/8A/+f/AP/n/4H/5/+B/+f/g//n98Pv5/fD7+f358/n8+fP5/P/z+fx/4/n8f+P5/H/D+fw/w/n8P4P5/B+D+fwfg/n8DwP5/"
    "AAD+fwAA/n8AAP5/AAD+fwAA/gAAAAAAAAAAAAAAAAAAAAA/wAf4P8AH+D/AB/g/wAf4P8AH+D/AB/g/wAf4P8AH+D/AB/g/wAf4"
    "P8AH+D////g////4P///+D////g////4P///+D/AB/g/wAf4P8AH+D/AB/g/wAf4P8AH+D/AB/g/wAf4P8AH+D/AB/g/wAf4P8AH"
    "+D/AB/gAAAAAAAAAAAAP+AAAf/8AAf//gAP//+AH///gD/w/8B/wD/gf4Af4P8AD/D/AA/w/wAH+P4AB/n+AAf5/gAH+f4AB/n+A"
    "Af5/gAH+f4AB/j+AAf4/wAH+P8AD/D/AA/wf4Af4H/AP+A/8P/AH///wA///4AH//4AAf/8AAA/4AAAAAAAAAAAAP///+D////g/"
    "///4P///+D////g////4P8AH+D/AB/g/wAf4P8AH+D/AB/g/wAf4P8AH+D/AB/g/wAf4P8AH+D/AB/g/wAf4P8AH+D/AB/g/wAf4"
    "P8AH+D/AB/g/wAf4P8AH+D/AB/g/wAf4P8AH+D/AB/g/wAf4AAAAAAAAAAAf//AAH//+AB///4Af///AH///wB///+Af4D/gH+Af"
    "4B/gD/Af4A/wH+AP8B/gH/Af4B/gH+A/4B///+Af///AH///gB///wAf//wAH//AAB/gAAAf4AAAH+AAAB/gAAAf4AAAH+AAAB/g"
    "AAAf4AAAH+AAAB/gAAAAAAAAAAAAAAAD/gAAH//AAH//8AD///AB///wA/+D8AP+AHAH/AAQB/gAAA/wAAAP8AAAD+AAAB/gAAAf"
    "4AAAH+AAAB/gAAAf4AAAH+AAAA/gAAAP8AAAD/AAAAf4AAAH/AAQA/4AcAP/g/AB///wAP//8AB///AAH//AAAP+AAAAAAAAAAAA"
    "P////D////w////8P////D////w////8AA/wAAAP8AAAD/AAAA/wAAAP8AAAD/AAAA/wAAAP8AAAD/AAAA/wAAAP8AAAD/AAAA/w"
    "AAAP8AAAD/AAAA/wAAAP8AAAD/AAAA/wAAAP8AAAD/AAAA/wAAAP8AAAD/AAAAAAAAAAAAB/wAP+P8AD/D/gA/wf4Af4H/AH+A/w"
    "D/AP+A/wB/gf4Af8H+AD/D/AA/4/wAH+f4AB/3+AAP//gAD//wAAf/8AAH/+AAA//gAAP/wAAB/8AAAf+AAAD/gAAA/wAAAf8AAD"
    "/+AAA//gAAP/wAAD/4AAA/8AAAP4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH4AAAB+AAAA/wAAH//4AH///gD///8B////g/x+P8"
    "P4fh/H8H4P5/B+D+fwfg/n8H4P5/B+D+fwfg/j+H4fw/5+f8H///+A////AH///gAP//AAAH4AAAB+AAAAfgAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAP8AD/B/AB/gf4Af4D/AP8Af4H+AH+B/gA/w/wAH+f4AB/v8AAP//AAB//gAAf/wAAD/8AAAf+AAAH/gAAD/4AAA/"
    "/AAAf/4AAP/+AAD//wAB/n+AA/w/wAP8P8AH+B/gD/AP8A/gD/Af4Af4P8AD/H+AA/wAAAAAAAAAAAAAAAA/AD+APwA/gD8AP4A/"
    "AD+APwA/gD8AP4A/AD+APwA/gD8AP4A/AD+APwA/gD8AP4A/AD+APwA/gD8AP4A/AD+APwA/gD8AP4A/AD+AP4A/gD////g////4"
    "P///+D////g////4AAAA+AAAAPgAAAD4AAAA+AAAAPgAAAAAAAAAAD/AB/g/wAf4P8AH+D/AB/g/wAf4P8AH+D/AB/g/wAf4P8AH"
    "+D/AB/g/4Af4P+AH+B/wB/gf///4H///+A////gH///4A///+AB///gAAAf4AAAH+AAAB/gAAAf4AAAH+AAAB/gAAAf4AAAH+AAA"
    "B/gAAAf4AAAH+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHwH4H58B+B+fAfgfnwH4H58B+B+fAfgfnwH4H58B+B+fAfgfnwH4H58"
    "B+B+fAfgfnwH4H58B+B+fAfgfnwH4H58B+B+f////n////5////+f////gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAB8D4HwfA+B8HwPgfB8D4HwfA+B8HwPgfB8D4HwfA+B8HwPgfB8D4HwfA+B8HwPgfB8D4HwfA+B8HwPgfB////+f///"
    "/n////5////+AAAAHgAAAB4AAAAeAAAAHgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB+AAAH/+AAH//4AB///gAf//8AH///gB4A"
    "/4AYAH/AAAA/wAAAH+AAAB/gAAAP4AH///AB///wAf//8AH///AB///wAf//8AAAD+AAAB/gAAAf4AAAP+AQAH/AHAD/gB/H/4Af"
    "//8AH//+AB//+AAH//AAAP8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHwA/wB8A//AfA//4Hwf//B8H4P4fD8A/Hw/APx8fgB8"
    "fH4Afn/+AH5//gB+f/4Afn/+AH58fgB+fH4AfHw/APx8PwD8fB+D+Hwf//B8D//gfAP/wHwA/wAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAH//4AH//+AH///gD///4A///+Af///gH+Af4B/gH+Af4B/gH+Af4B/gH+Af4B/gD/Af4A///+AH///gA///4AD//+AAf"
    "//gAP//4AH/H+AB/h/gA/4f4AP8H+AH+B/gD/gf4A/wH+Af8B/gH+Af4D/AH+A/wB/gAAAAAAAAAAAAP8AAAP/wAAH/+AAD//wAB"
    "//+AA/w/wAP4H8AH+A/gB/AP4AfwD+AH8A/wD/AP8A/wD/AP8A/wD/AP8A/wD/AP8A/wD/AP8A/wD/AH8A/wB/AP4AfwD+AH+A/g"
    "A/gfwAP8P8AB//+AAP//AAB//gAAP/wAAA/wAAAAAAAAAAAAAD/wAAf/8AAH//AAB//wAAf/8AAH//AABg/wAAAP8AAAD/AAAA/w"
    "AAAP8AAAD/AAAA/wAAAP8AAAD/AAAA/wAAAP8AAAD/AAAA/wAAAP8AAAD/AAAA/wAAAP8AAAD/AAA///wAf//+AH///gB///4Af/"
    "/+AH///gAAAAAAAAAAAAP/AAA//8AAf//wAH//+AB///gAfg/8AHAD/ABAA/wAAAH8AAAB/AAAAfwAAAP8AAAD/AAAB/gAAA/wAA"
    "Af4AAAP8AAAH+AAAD/AAAB/gAAB/wAAA/4AAAf8AAAP+AAAH///AB///4Af//+AH///gB///4Af//+AAAAAAAAAAAAB/8AAD//4A"
    "A///AAP//4AD///AA4B/wAAAP8AAAB/AAAAfwAAAH8AAAD/AAAD/gAA//wAAP/wAAD/+AAA//4AAP//AAAB/wAAAH+AAAB/gAAAf"
    "4AAAH+AAAB/gBAA/4AeAf8AH///AB///gAf//wAH//wAAH/gAAAAAAAAAAAAAAH/AAAD/wAAB/8AAAf/AAAP/wAAH/8AAB//AAA/"
    "/wAAfv8AAHz/AAD8/wAB+P8AAfD/AAPw/wAH4P8AB8D/AA/A/wAfgP8AHwD/AB////Af///wH///8B////Af///wAAD/AAAA/wAA"
    "AP8AAAD/AAAA/wAAAP8AAAAAAAAAAAAH//+AB///gAf//4AH//+AB///gAf//wAH4AAAB+AAAAfgAAAH84AAB//4AAf//gAH//8A"
    "B///gAf//4AGAH/AAAA/wAAAH+AAAB/gAAAf4AAAH+AAAB/gAAA/wA4Af8APwP/AD///gA///wAP//4AB//4AAB/4AAAAAAAAAAA"
    "AAAH/AAAH/+AAH//gAD//4AB//+AAf4DgAP8AAAD+AAAB/AAAAfwAAAH4AAAD+f8AA///gAP//8AD///gA///8AP+D/AD/AfwA/w"
    "H+AP8A/gB/AP4AfwD+AH8B/AA/gfwAP8P8AB//+AAf//AAD//gAAP/wAAA/wAAAAAAAAAAAAD///4A///+AP///gD///4A///+AP"
    "///AAAA/wAAAP8AAAH+AAAB/gAAAfwAAAP8AAAD+AAAB/gAAAfwAAAP8AAAD+AAAB/gAAAfwAAAP8AAAD+AAAB/gAAAf4AAAP8AA"
    "AD/AAAA/gAAAf4AAAH8AAAD/AAAA/gAAAAAAAAAAAAAAH/AAAP/+AAH//wAD//+AB///wAf4P8AH8B/AB/AfwAfwH8AH8B/AB/g/"
    "gAP+/4AB//8AAH/8AAD//AAB//8AA///gAfwP8AH8B/AD+AP4A/gD+AP4A/gD+Af4A/wH8AH+D/AB///wAP//4AB//8AAP/+AAAf"
    "8AAAAAAAAAAAAAAf4AAAf/gAAf/+AAP//wAD//8AB/h/gAfwP4AP8B/AD+AfwA/gH8AP4B/gD+Af4A/wH+AH8D/gB/h/4Af//+AD"
    "///gAf//4AB/7+AAHw/AAAAfwAAAH8AAAD+AAAB/gAOA/wAD//8AA//+AAP//AAD//AAAH+AAAAAAAAAAAAAAB/wAAAf8AAAP/AA"
    "AD/4AAA/+AAAf/gAAH/8AAB//AAA/vwAAP7+AAD8/gAB/H4AAfx/AAH4fwAD+D8AA/g/gAP4P4AH8B+AB/AfwAf//8AP///AD///"
    "4A///+Af///gH///8B/AB/A/gAP4P4AD+D+AA/h/AAP8AAAAAAAAAAAH//+AB///gAf//4AH//+AB///gAf//4AH8AAAB/AAAAfw"
    "AAAH8AAAB/AAAAfwAAAH//wAB///AAf//4AH///AB///wAfwf+AH8B/gB/AP4AfwD+AH8A/gB/AP4AfwH+AH8D/AB///wAf//4AH"
    "//+AB//+AAf/+AAAAAAAAAAAAAf/4AAH//wAB///AAf//4AH//+AB/B/wAfwP8AH8B/AB/AfwAfwH8AH8B+AB/B/gAf//wAH//4A"
    "B//+AAf//4AH///AB/A/wAfwD8AH8A/gB/AP4AfwD+AH8A/gB/Af4AfwP8AH///AB///wAf//4AH//4AB//4AAAAAAAAAAAAA///"
    "gAP//4AD//+AA///gAP//4AD//+AA/gAAAP4AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAP4"
    "AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAP4AAAD+AAAAAAAAAAAAAAA//+AAP//gAD//4AA//+AAP//gAD8H4AA"
    "/B+AAPwfgAD8H4AA/B+AAPwfgAD8H4AA/B+AAPwfgAD8H4AA/B+AAPwfgAD4H4AB+B+AA/gfgA////AP///wD///8A////AP///w"
    "D4AA8A+AAPAPgADwD4AA8A+AAPAAAAAAAAAAAAP//wAD//+AA///gAP//4AD//+AA///gAP4AAAD+AAAA/gAAAP4AAAD+AAAA/gA"
    "AAP//wAD//8AA///AAP//wAD//8AA/gAAAP4AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAP//4AD//+AA///gAP//4AD//+AA///gAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+A8B8HwPA+A+DwfAHw8PgA8PDwAPjx8AB88+AAPvfAAD//wAA//8AAf//gAP//8AD"
    "7/fAB8fj4AfDw+APg8HwHwPA+B8DwPg+A8B8fgPAfgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH4AAA//4AAP//gAD//8A"
    "A///gAPh/4ACAD+AAAA/gAAAP4AAAD+AAAA/gAAAfwAAP/4AAD/8AAA//AAAP/4AAD//AAAA/4AAAD+AAAAfwAAAH8AAAB/AAAAf"
    "wAQAP8AHgP+AB///gAf//wAH//4AB//8AAB/wAAAAAAAAAAAAB/AD/AfwB/wH8Af8B/AP/AfwD/wH8B/8B/Af/AfwP/wH8D/8B/B"
    "//Afwf/wH8P38B/D9/Afx+fwH8fn8B/Px/Afz8fwH8+H8B/fh/Af3wfwH/8H8B/+B/Af/gfwH/wH8B/8B/Af+AfwH/gH8B/wB/Af"
    "8AfwH+AH8AAAAAAAAAAAP4AP8D+AH+A/gD/AP4B/gD+A/wA/gP8AP4H+AD+D/AA/h/gAP4/wAD+f4AA/v8AAP//AAD//wAA//+AA"
    "P//wAD//8AA///gAP/P4AD/j/AA/wf4AP4D+AD+A/wA/gH+AP4A/gD+AP8A/gB/gP4AP4D+AD/A/gAf4AAAAAAAAAAAAP//4AD//"
    "+AA///gAP//4AD//+AA///gAP4P4AD+D+AA/g/gAP4P4AD+D+AA/g/gAP4P4AD+D+AA/g/gAfwP4AH8D+AB/A/gAfwP4AH8D+AB/"
    "A/gA/wP4Af4D+AP+A/gf/gP4H/wD+B/4A/gf8AP4H8AD+BwAA/gAAAAAAAAAAH/AA/5/4AP+f+AH/n/gB/5/8A/+f/AP/n/wD/5/"
    "+B/+f/gf/n/8H/5/fD7+f3w+/n9+fv5/Pnz+fz58/n8f/P5/H/j+fx/4/n8P8P5/D/D+fw/w/n8H4P5/B+D+fwPg/n8AAP5/AAD+"
    "fwAA/n8AAP5/AAD+fwAA/gAAAAAAAAAAH8AH8B/AB/AfwAfwH8AH8B/AB/AfwAfwH8AH8B/AB/AfwAfwH8AH8B/AB/Af4A/wH///"
    "8B////Af///wH///8B////Af4A/wH8AH8B/AB/AfwAfwH8AH8B/AB/AfwAfwH8AH8B/AB/AfwAfwH8AH8B/AB/AfwAfwAAAAAAAA"
    "AAAAB8AAAH/4AAD//gAD//8AB///gAf//8AP8B/gD+AP4B/gD/AfwAfwH8AH8D+AA/g/gAP4P4AD+D+AA/g/gAP4P4AD+D+AA/g/"
    "gAP4H8AH8B/AB/AfwAfwH+AP4A/wH+AH/v/AB///wAP//4AB//8AAH/8AAAP4AAAAAAAAAAAAB////Af///wH///8B////Af///w"
    "H///8B/AB/AfwAfwH8AH8B/AB/AfwAfwH8AH8B/AB/AfwAfwH8AH8B/AB/AfwAfwH8AH8B/AB/AfwAfwH8AH8B/AB/AfwAfwH8AH"
    "8B/AB/AfwAfwH8AH8B/AB/AfwAfwH8AH8AAAAAAAAAAAB//wAAf//gAH//8AB///gAf//8AH///AB/A/wAfwH+AH8A/gB/AP4Afw"
    "D+AH8A/gB/Af4AfwP8AH///AB///wAf//4AH//8AB//8AAf/4AAH8AAAB/AAAAfwAAAH8AAAB/AAAAfwAAAH8AAAB/AAAAfwAAAH"
    "8AAAAAAAAAAAAAAAAPgAAAf/gAAf/+AAP//gAH//4AD//+AB/wDgAf4AIAP8AAAD+AAAA/gAAAfwAAAH8AAAB/AAAAfwAAAH8AAA"
    "B/AAAAfwAAAH8AAAA/gAAAP4AAAD/AAAAfwAIAH/AOAA/8fgAP//4AB//+AAH//gAA//gAAB/AAAAAAAAAAAAB////Af///wH///"
    "8B////Af///wH///8AAP4AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAP"
    "4AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAAAAAAAAAAH8AD+A/gB/AP4AfwD/AP8AfwD+AH+A/gA/gfwAP4H8AB"
    "/D/AAfw/gAD+f4AA/n8AAP9/AAB//gAAf/4AAD/+AAA//AAAH/wAAB/4AAAf+AAAD/AAAA/wAAAP8AAAD+AAAf/gAAH/wAAB/8AA"
    "Af+AAAH/AAAB+AAAAAAAAAAAAAAAAAAAAAfgAAAH4AAAB+AAAH/+AAP//8AP///wH///+B////g/x+P8P4fh/H8H4P5/B+D+fwfg"
    "/n8H4P5/B+D+fwfg/n8H4P4/h+H8P/fv/B////gP///wB///4AH//4AAH/gAAAfgAAAH4AAAB+AAAAAAAAAAAAAAAAAAAAAAAB/A"
    "B/AfwA/wD+AP4A/wH8AH8B/AA/g/gAP4f4AB/H8AAf7+AAD//gAAf/wAAH/4AAA/+AAAP/AAAB/wAAAf8AAAP/gAAD/4AAB//AAA"
    "//wAAP7+AAH8/wAB/H8AA/g/gAf4P4AH8B/AD+Af4A/gD+AfwAfwP8AH8AAAAAAAAAAAD8AfAA/APwAPwD8AD8A/AA/APwAPwD8A"
    "D8A/AA/APwAPwD8AD8A/AA/APwAPwD8AD8A/AA/APwAPwD8AD8A/AA/APwAPwD8AD8A/AA/APwAP///wD///8A////AP///wD///"
    "8AAAAPAAAADwAAAA8AAAAPAAAADwAAAAAAAAAAAP4AfwD+AH8A/gB/AP4AfwD+AH8A/gB/AP4AfwD+AH8A/gB/AP4AfwD/AH8A/w"
    "B/AH+AfwB///8Af///AD///wA///8AH///AAP//wAAAH8AAAB/AAAAfwAAAH8AAAB/AAAAfwAAAH8AAAB/AAAAfwAAAH8AAAB/AA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAHwH4D5+B+A+fgfgPn4H4D5+B+A+fgfgPn4H4D5+B+A+fgfgPn4H4D5+B+A+fgfgPn4H4D5+B+A+"
    "fgfgPn4H4D5+B+A+fgfgPn4H4H5////+f////n////5////+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB8D4HwfA+B"
    "8HwPgfB8D4HwfA+B8HwPgfB8D4HwfA+B8HwPgfB8D4HwfA+B8HwPgfB8D4HwfA+B8HwPgfB8D4HwfA+B8H////5////+f////j//"
    "//4AAAAeAAAAHgAAAB4AAAAeAAAAAAAAAAAAAAAAAAAAAAAAAAAAHwAAAf/gAAf/+AAH//wAB//+AAf//wAHAP+ABgA/gAAAP8AA"
    "AB/AAAAfwAAAD+AAf//gAP//4AD//+AA///gAP//4AB//+AAAA/gAAAP4AAAH8AAAB/ABAA/wAcAf4AH4/8AB///AAf//gAH//gA"
    "Af/wAAA/gAAAAAAAAAAAAAAAAAAAAAAAAAB+AH4B/4B+B//gfg//8H4f//h+H4H4fj8A/H4/APx+PgB+fn4Afn/+AH5//gB+f/4A"
    "fn/+AH5//gB+fn4Afn4+AH5+PwD8fj8A/H4fgfh+H//4fg//8H4H/+B+A//AAAB+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//4AB/"
    "/+AA///gAf//4AH//+AD///gA/wP4AP4D+AD+A/gA/AP4APwD+AD+A/gA/gP4AH//+AA///gAH//4AA//+AAH//gAD//4AA/j+AA"
    "f4/gAH8P4AD/D+AA/g/gAf4P4AH8D+AD+A/gA/gP4AfwD+AH8A/gAAAAAAAAAAAAB8AAAB/wAAB//AAA//4AAP/+AAH+fwAB+D+A"
    "A/gfgAPwH4AD8B+AB/AfwAfwH8AH8B/AB/AfwAfwH8AH8B/AB/AfwAfwH8AH8B/AB/AfwAPwH4AD8B+AA/AfgAP4P4AB/H8AAf//"
    "AAD//gAAf/wAAD/4AAAHwAAAAAAAAAAAAAAf4AAB/+AAA//gAAP/4AAD/+AAA//gAAMP4AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAP"
    "4AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAP4AAAD+AAAA/gAAAP4AAD//+AA///gAP//4AD//+AA///gAAAAAAA"
    "AAAAAA+AAAD/+AAD//wAA//+AAP//wAD+/8AA8B/AAMAf4AAAD+AAAA/gAAAP4AAAD8AAAB/AAAAfwAAAP4AAAH8AAAD+AAAB/gA"
    "AA/wAAAf4AAAP8AAAH+AAAD/AAAB/gAAA///AAP//4AD//+AA///gAP//4AD//+AAAAAAAAAAAAAD8AAAP/8AAH//gAB//8AAf//"
    "gAHx/4ABAD+AAAA/gAAAP4AAAD+AAAA/gAAAfwAAH/4AAB/8AAAf/AAAH/4AAB//AAAAf4AAAD+AAAAfwAAAH8AAAB/AAAAfwAIA"
    "P8ADwP+AA///gAP//wAD//4AA//8AAA/wAAAAAAAAAAAAAAB/gAAA/4AAAP+AAAH/gAAD/4AAA/+AAAf/gAAH/4AAD7+AAB8/gAA"
    "fP4AAPj+AAD4/gAB8P4AA+D+AAPg/gAHwP4AB8D+AA+A/gAP///gD///4A///+AP///gD///4Af//8AAAP4AAAD+AAAA/gAAAP4A"
    "AAD+AAAAAAAAAAAAAf//AAH//wAB//8AAf//AAH//wAB//8AAfgAAAH4AAAB+AAAAfgAAAH/8AAB//wAAf/+AAH//wAB//+AAcB/"
    "gAAAP4AAAB/AAAAfwAAAH8AAAB/AAAAfwAIAP8ADgD+AA/D/gAP//wAD//8AA//+AAH/+AAAH+AAAAAAAAAAAAAAA/wAAB//AAA/"
    "/wAAf/8AAP//AAD+AwAB/AAAAfgAAAPwAAAD8AAAA/HgAAP3/AAH//4AB///AAf//4AH/H+AB/gfgAf4H8AH8B/AA/AfwAPwH8AD"
    "8B/AA/gfgAH4H4AB/D+AAP//AAD//wAAf/4AAB/4AAAH4AAAAAAAAAAAAAP//8AD///AA///wAP//8AD///AA///gAAAP4AAAD+A"
    "AAA/AAAAfwAAAH4AAAD+AAAA/gAAAPwAAAH8AAAB+AAAA/gAAAP4AAAD8AAAB/AAAAfgAAAP4AAAD+AAAA/AAAAfwAAAH4AAAD+A"
    "AAA/gAAAPwAAAH8AAAAAAAAAAAAAAAfAAAB//AAA//4AAf//AAP//4AD/D+AA/g/gAPwH4AD8B+AA/AfgAP4P4AB/H8AAP/+AAB/"
    "/AAAP/gAAP/+AAH//wAD+D+AA/AfgAfwH8AH8B/AB/AfwAfwH8AH8B/AA/g/gAP//4AB//8AAP/+AAB//AAAD+AAAAAAAAAAAAAA"
    "D8AAAD/4AAD//AAB//4AAf/+AAP4fwAD8D8AB/A/gAfwH4AH8B+AB/AfgAfwP8AH8D/AA/A/wAP8f8AD///AAf//wAD//8AAf9+A"
    "AA8fgAAAH4AAAB+AAAA/AAAAfwABwP4AAf/+AAH//AAB//gAAf/gAAA/gAAAAAAA"
)

S = 32


def _decode(chars, n, b64):
    raw = np.frombuffer(base64.b64decode("".join(b64)), np.uint8)
    asp = raw[:n * 4].view(np.float32).copy()
    bits = np.unpackbits(raw[n * 4:]).reshape(n, S * S)
    return list(chars), asp, bits.reshape(n, S, S).astype(np.float32)


def _load_templates():
    ch, asp, tpl = _decode(_TPL_CHARS, _TPL_N, _TPL_B64)
    if _REAL_N:
        ch2, asp2, tpl2 = _decode(_REAL_CHARS, _REAL_N, _REAL_B64)
        ch, asp, tpl = ch + ch2, np.concatenate([asp, asp2]), np.concatenate([tpl, tpl2])
    tpl = np.stack([cv2.GaussianBlur(t, (0, 0), 1.2) for t in tpl])
    tpl /= np.linalg.norm(tpl.reshape(len(ch), -1), axis=1)[:, None, None]
    return ch, asp, tpl


# шаблоны букв, вырезанные с реальных кадров камеры робота (заполняются ниже)
_REAL_CHARS, _REAL_N, _REAL_B64 = "", 0, ()
_REAL_CHARS, _REAL_N = "ААББВВ", 6
_REAL_B64 = (
    "VVVVP+Q4Tj8fhWs/G8prP4ZeaD8++GA/AAAAAAAOAAAAPwAAAH8AAAD/gAAA/4AAAP/AAAD/4AAA/+AAAPPwAAHz8AAB8/gAAfH4"
    "AAHw/AAD4PwAA+B+AAPgfwAD4H8AA+f/gAf8/4AH+P/AB///4Af/D+AP+AfwD8AH8A+AA+APgAAAD4AAAB+AAAAfAAAAHgAAAAAA"
    "AAAAAAAAAAIAAAAfAAAAf4AAAH+AAAB/wAAAf8AAAH/gAAD/4AAA//AAAPHwAADx+AAB8PwAAfD8AAHw/gAB8P8AA+D/AAPw/4AD"
    "//+AA///wAP//8AD///gA/8H4Af4B/AHwAPwB8ADgAfAAAAHwAAAD4AAAA8AAAAIAAAAAAAAAAAAAAAAAD/gP///4D///+A////g"
    "P4//8D4f/+A8IAAAOAAAADAAAAAwAAAAMAAAADAAAAAwH/+AOAD/4DwB//A8Af/4PB//+DwgD/w4QAT8OEAEfDBABnwwYAZ8OGAG"
    "/DwgDvg/P//4P///8D///+A////AP///AB//8AAAAAAAAAAAAAAB//A////wP///8D////A////wP///4D/gAAA/wAAAP8AAAD/A"
    "AAA/wAAAP8AAAD///4A//j/gP/4f8D/+H/g////8H+AP/B/gB/wf4Af8H+AD/B/gA/wf4Af8H+AP/B////gf///4H///8B///+Af"
    "//8AH//wAAAAAAAAAAAAP//4AD///wA///+AP///wD///+A////gP8A/8D/AH/A+wA/wPsAP4D/AD+A/wB/gP///wD///wA/5/8A"
    "P8f/wD///+A////wP4AP+D7AB/g+wAf4PsAH+D/AB/g/wA/4P///+D////A////gP///wD///4A///wAAAAAAAAAAAAf//4AH///"
    "gB///8Af///gH///8B/gP/AfwB/wH8AP8B/AD/AfwA/wH8AP4B/gH+Af///AH///gB///4Af///gH///8B/AD/gfwAf4H8AH+B/A"
    "B/gfwAf4H8AH+B/AB/gf///4H///+B///+Af///AH///gA//8AAAAAAA"
)

_CH, _ASP, _TPL = _load_templates()
_TPL_N_ALL = len(_CH)


def _norm_glyph(binary):
    """Буква (белая на чёрном) -> 32x32 с сохранением пропорций; + отношение w/h."""
    ys, xs = np.nonzero(binary)
    crop = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    k = (S - 2) / max(h, w)
    nh, nw = max(1, round(h * k)), max(1, round(w * k))
    r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((S, S), np.uint8)
    y0, x0 = (S - nh) // 2, (S - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return (out > 127).astype(np.float32), w / h


ROT_ANGLES = (-15, -10, -5, 0, 5, 10, 15)   # табличка может висеть с наклоном
STRETCH = (1.0, 1.3, 1.6)                   # если смотрим сбоку, буква сжата по ширине


def _score_glyph(binary):
    g, asp = _norm_glyph(binary)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    n = np.linalg.norm(g)
    if n == 0:
        return np.zeros(_TPL_N_ALL)
    corr = (_TPL * (g / n)).sum(axis=(1, 2))
    return corr * np.exp(-1.5 * np.abs(np.log(asp / _ASP)))   # штраф за другие пропорции


def classify_char(binary):
    """-> (символ, уверенность 0..1). Перебирает небольшие повороты."""
    corr = np.zeros(_TPL_N_ALL)
    for k in STRETCH:
        b = binary if k == 1.0 else cv2.resize(
            binary, (max(1, round(binary.shape[1] * k)), binary.shape[0]),
            interpolation=cv2.INTER_NEAREST)
        h, w = b.shape
        p = int(0.3 * max(h, w)) + 2
        padded = cv2.copyMakeBorder(b, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
        c = (padded.shape[1] / 2, padded.shape[0] / 2)
        for a in ROT_ANGLES:
            if a == 0:
                r = padded
            else:
                M = cv2.getRotationMatrix2D(c, a, 1.0)
                r = cv2.warpAffine(padded, M, (padded.shape[1], padded.shape[0]),
                                   flags=cv2.INTER_NEAREST)
            if r.any():
                corr = np.maximum(corr, _score_glyph(r))
    best = {}
    for ch, s in zip(_CH, corr):
        if PLATE_ALPHABET and ch not in PLATE_ALPHABET:
            continue
        best[ch] = max(best.get(ch, 0.0), float(s))
    if not best:
        return None, 0.0
    ch = max(best, key=best.get)
    sc = best[ch]
    # Б и В отличаются правым верхом: у В там верхняя «петля», у Б пусто
    if ch in "БВ" and "Б" in best and "В" in best:
        ys, xs = np.nonzero(binary)
        c = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0
        hh, ww = c.shape
        ink = c[int(0.18 * hh):int(0.40 * hh), int(0.68 * ww):].mean()
        ch = "В" if ink > 0.45 else "Б"
        sc = max(best["Б"], best["В"])
    return ch, sc


def _overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


# ---- табличка ----------------------------------------------------------------
def read_plate(img, exclude=()):
    """Ищет тёмные буквы на белом поле (табличка в красной рамке).
    exclude — рамки (x1,y1,x2,y2), где буквы искать не надо (ArUco, манометр).
    -> {"text", "score", "box", "chars": [...]} или None"""
    H, W = img.shape[:2]
    N = H * W
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 2] > 140) & (hsv[..., 1] < 70)
    local = cv2.blur(gray, (41, 41)).astype(np.float32)
    dark = ((gray < local * 0.72) & (gray < 150) & (hsv[..., 1] < 110)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)

    cands = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if not (0.00025 * N <= area <= 0.03 * N and 0.03 * H <= h <= 0.35 * H):
            continue
        if not (0.25 <= w / h <= 1.6 and 0.15 <= area / (w * h) <= 0.85):
            continue
        box = (x, y, x + w, y + h)
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:   # обрезано краем кадра
            continue
        if any(_overlap(box, e) for e in exclude):
            continue
        # вокруг буквы должно быть белое поле таблички
        m = max(2, int(0.22 * h))
        X1, Y1, X2, Y2 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
        ring = np.ones((Y2 - Y1, X2 - X1), bool)
        ring[y - Y1:y - Y1 + h, x - X1:x - X1 + w] = False
        if white[Y1:Y2, X1:X2][ring].mean() < 0.6:
            continue
        glyph = (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        ch, sc = classify_char(glyph)
        if ch and sc >= PLATE_MIN_SCORE:
            cands.append({"ch": ch, "score": sc, "box": box, "glyph": glyph})
    if not cands:
        return None

    # склеиваем соседние буквы в одну строку (на случай кода из 2+ символов)
    cands.sort(key=lambda c: c["box"][0])
    groups = []
    for c in cands:
        x1, y1, x2, y2 = c["box"]
        h = y2 - y1
        for g in groups:
            gx1, gy1, gx2, gy2 = g["box"]
            gh = gy2 - gy1
            vert = min(y2, gy2) - max(y1, gy1)
            if 0.7 < h / gh < 1.4 and vert > 0.6 * min(h, gh) and 0 <= x1 - gx2 < 0.8 * gh:
                g["chars"].append(c)
                g["box"] = (gx1, min(gy1, y1), x2, max(gy2, y2))
                break
        else:
            groups.append({"chars": [c], "box": c["box"]})
    for g in groups:
        g["text"] = "".join(c["ch"] for c in g["chars"])
        g["score"] = float(np.mean([c["score"] for c in g["chars"]]))
    return max(groups, key=lambda g: g["score"])


# ---- манометр ----------------------------------------------------------------
def _color_masks(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (h >= 35) & (h <= 95) & (s > 45) & (v > 60)
    red = ((h <= 10) | (h >= 165)) & (s > 60) & (v > 70)
    return {"green": green, "red": red}


def read_gauge(img):
    """Полукруг: зелёный и красный секторы + тёмная стрелка из центра нижнего края.
    -> {"color": "зеленый"/"красный"/"посередине", "color_key", "angle", "boundary",
        "margin", "box", "pivot", "R"} или None"""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = _color_masks(hsv)
    colored = (masks["green"] | masks["red"]).astype(np.uint8)
    # стрелка может разрезать полукруг — склеиваем
    joined = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(joined)
    best = None
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 0.0008 * H * W or h < 8:
            continue
        if not (1.4 <= w / h <= 2.9):                    # полукруг: ширина ≈ 2 высоты
            continue
        comp = lab[y:y + h, x:x + w] == i
        if not (0.55 <= comp.sum() / (w * h) <= 0.95):   # полукруг заполняет ~78% рамки
            continue
        g = (masks["green"][y:y + h, x:x + w] & comp).sum()
        r = (masks["red"][y:y + h, x:x + w] & comp).sum()
        if min(g, r) < 0.08 * (g + r):                   # должны быть видны оба цвета
            continue
        if best is None or area > best[4]:
            best = (x, y, w, h, area, i)
    if best is None:
        return None

    x, y, w, h, _, i = best
    R = w / 2.0
    cx, cy = x + R, y + h                                # центр — середина нижнего края
    ys, xs = np.mgrid[y:y + h, x:x + w]
    dx, dy = xs - cx, cy - ys
    rr = np.hypot(dx, dy) / R
    ang = np.degrees(np.arctan2(dy, dx))                 # 0 — вправо, 90 — вверх, 180 — влево
    inside = (rr > 0.12) & (rr < 0.88) & (dy > 0)
    roi_v = hsv[y:y + h, x:x + w, 2].astype(np.float32)
    roi_s = hsv[y:y + h, x:x + w, 1].astype(np.float32)
    col_v = np.median(roi_v[(lab[y:y + h, x:x + w] == i) & inside])
    needle = inside & (roi_v < min(0.75 * col_v, 130)) & (roi_s < 90)
    if needle.sum() < 6:
        return None
    a = np.radians(ang[needle])
    wgt = rr[needle]                                     # дальние точки весомее (кончик)
    theta = math.degrees(math.atan2((np.sin(a) * wgt).sum(), (np.cos(a) * wgt).sum()))

    # граница цветов: между краями зелёного и красного секторов (стрелку не считаем)
    zone = inside & ~needle & (rr > 0.25)
    ga = ang[masks["green"][y:y + h, x:x + w] & zone]
    ra = ang[masks["red"][y:y + h, x:x + w] & zone]
    if len(ga) < 10 or len(ra) < 10:
        return None
    green_left = np.median(ga) > np.median(ra)
    if green_left:
        boundary = (np.percentile(ga, 3) + np.percentile(ra, 97)) / 2
    else:
        boundary = (np.percentile(ra, 3) + np.percentile(ga, 97)) / 2
    delta = theta - boundary                             # >0 — стрелка левее границы
    if abs(delta) <= GAUGE_MID_DEG:
        key = "mid"
    elif (delta > 0) == green_left:
        key = "green"
    else:
        key = "red"
    margin = min(1.0, abs(delta) / (2 * GAUGE_MID_DEG))  # насколько уверенно стрелка в секторе
    if key == "mid":
        margin = 1.0 - abs(delta) / GAUGE_MID_DEG / 2
    return {"color": COLOR_NAMES[key], "color_key": key, "angle": theta,
            "boundary": float(boundary), "margin": float(margin),
            "box": (x, y, x + w, y + h), "pivot": (cx, cy), "R": R}


# ---- рисование ---------------------------------------------------------------
_TR = dict(zip("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
               ("A B V G D E E ZH Z I Y K L M N O P R S T U F H C CH SH SCH ' Y ' E YU YA"
                ).split()))
_FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
_fonts = {}


def _ascii(s):
    return "".join(_TR.get(c.upper(), c) for c in s)


def _font(size):
    """Шрифт с кириллицей через Pillow; None — если Pillow/шрифта нет."""
    if size not in _fonts:
        _fonts[size] = None
        try:
            from PIL import ImageFont
            for p in _FONT_PATHS:
                try:
                    _fonts[size] = ImageFont.truetype(p, size)
                    break
                except OSError:
                    pass
        except ImportError:
            pass
    return _fonts[size]


def put_text(img, text, org, size=18, color=(255, 255, 255)):
    """Подпись по-русски (org — левый нижний угол, как у cv2.putText).
    Без Pillow пишет транслитом."""
    font = _font(size)
    x, y = int(org[0]), int(org[1])
    if font is None:
        cv2.putText(img, _ascii(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size / 30, color, 2 if size >= 18 else 1)
        return img
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text((x, y - size), text, font=font, fill=tuple(color[::-1]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def draw_results(img, plate=None, gauge=None, color=(0, 255, 255), show_conf=True):
    vis = img
    if plate:
        x1, y1, x2, y2 = plate["box"]
        cv2.rectangle(vis, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), color, 2)
        put_text(vis, f"табличка {plate['text']}" + (f" ({plate['score']:.2f})" if show_conf else ""),
                 (x1 - 4, max(20, y1 - 10)), 18, color)
    if gauge:
        x1, y1, x2, y2 = gauge["box"]
        cv2.rectangle(vis, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 2)
        cx, cy = gauge["pivot"]
        t = math.radians(gauge["angle"])
        cv2.line(vis, (int(cx), int(cy)),
                 (int(cx + gauge["R"] * math.cos(t)), int(cy - gauge["R"] * math.sin(t))),
                 (255, 0, 255), 2)
        put_text(vis, f"манометр: {gauge['color']}" + (f" ({gauge['margin']:.2f})" if show_conf else ""),
                 (x1 - 3, max(20, y1 - 10)), 18, color)
    return vis


def _aruco_boxes(img):
    try:
        from aruco_detect import detect_aruco
    except ImportError:
        return []
    _, ms = detect_aruco(img)
    out = []
    for m in ms:
        c = m["corners"]
        pad = 0.25 * m["side_px"]
        out.append((c[:, 0].min() - pad, c[:, 1].min() - pad,
                    c[:, 0].max() + pad, c[:, 1].max() + pad))
    return out


def analyze(img):
    gauge = read_gauge(img)
    excl = _aruco_boxes(img)
    if gauge:
        excl.append(gauge["box"])
    plate = read_plate(img, exclude=excl)
    return plate, gauge


def _fmt(plate, gauge):
    p = f"табличка {plate['text']} ({plate['score']:.2f})" if plate else "табличка —"
    g = (f"манометр {gauge['color']} (стрелка {gauge['angle']:.0f}°, "
         f"граница {gauge['boundary']:.0f}°)") if gauge else "манометр —"
    return f"{p} | {g}"


def _main_files(patterns):
    paths = []
    for p in patterns:
        paths += [x for x in glob.glob(p) if "_vision." not in x] or [p]
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"{p}: не открылся")
            continue
        plate, gauge = analyze(img)
        print(f"{p}: {_fmt(plate, gauge)}")
        cv2.imwrite(p.rsplit(".", 1)[0] + "_vision.jpg", draw_results(img.copy(), plate, gauge))


def _main_ros():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    rclpy.init()
    node = Node("vision_probe")

    def cb(msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        plate, gauge = analyze(img)
        print(_fmt(plate, gauge), flush=True)
        cv2.imwrite("last_vision.jpg", draw_results(img.copy(), plate, gauge))

    node.create_subscription(CompressedImage, "/front_camera/image_raw/compressed",
                             cb, qos_profile_sensor_data)
    print("Слушаю камеру, Ctrl+C — выход", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--ros":
        _main_ros()
    else:
        _main_files(sys.argv[1:])
