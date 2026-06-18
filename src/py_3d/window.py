"""Native py_3d pixel windowing."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
from time import perf_counter, sleep
from typing import Callable
import sys

from .buffer import PixelBuffer

KeyCallback = Callable[[str], None]
_WINDOW_CALLBACK = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
_NUMPY = None
_NUMPY_CHECKED = False


@dataclass(frozen=True)
class WindowEvent:
    kind: str
    key: str = ""
    pos: tuple[int, int] = (0, 0)
    rel: tuple[int, int] = (0, 0)
    button: int = 0
    y: int = 0
    size: tuple[int, int] = (0, 0)


@dataclass
class PixelWindow:
    """Display ``PixelBuffer`` frames in a native py_3d window."""

    width: int = 960
    height: int = 540
    title: str = "py_3d"
    fit_window: bool = True
    fullscreen: bool = False
    _events: list[WindowEvent] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("PixelWindow currently uses py_3d's native Win32 presenter.")
        self._win = _Win32Bindings()
        self._key_callback: KeyCallback | None = None
        self._scheduled: list[tuple[float, Callable[[], None]]] = []
        self._closed = False
        self._last_mouse: tuple[int, int] | None = None
        self._last_frame: PixelBuffer | None = None
        self._title = str(self.title)
        self._class_name = f"py_3d_Window_{id(self):x}"
        self._wndproc = self._win.WNDPROC(self._handle_message)
        self._register_class()
        self._hwnd = self._create_window()
        if not self._hwnd:
            raise ctypes.WinError()
        self._win.user32.ShowWindow(self._hwnd, self._win.SW_SHOW)
        self._win.user32.UpdateWindow(self._hwnd)

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    @property
    def size(self) -> tuple[int, int]:
        rect = wintypes.RECT()
        if self._win.user32.GetClientRect(self._hwnd, ctypes.byref(rect)):
            return (max(1, rect.right - rect.left), max(1, rect.bottom - rect.top))
        return (max(1, int(self.width)), max(1, int(self.height)))

    def set_title(self, title: str) -> None:
        self._title = str(title)
        self._win.user32.SetWindowTextW(self._hwnd, self._title)

    def bind_keys(self, callback: KeyCallback) -> None:
        self._key_callback = callback

    def poll_events(self) -> tuple[WindowEvent, ...]:
        message = wintypes.MSG()
        while self._win.user32.PeekMessageW(ctypes.byref(message), None, 0, 0, self._win.PM_REMOVE):
            self._win.user32.TranslateMessage(ctypes.byref(message))
            self._win.user32.DispatchMessageW(ctypes.byref(message))
        events = tuple(self._events)
        self._events.clear()
        return events

    def show(self, buffer: PixelBuffer) -> None:
        if self._closed:
            return
        self._last_frame = buffer
        target_width, target_height = self.size if self.fit_window else (buffer.width, buffer.height)
        payload = _bgra_bytes(buffer)
        info = self._win.BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(self._win.BITMAPINFOHEADER)
        info.bmiHeader.biWidth = buffer.width
        info.bmiHeader.biHeight = -buffer.height
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = self._win.BI_RGB
        hdc = self._win.user32.GetDC(self._hwnd)
        try:
            self._win.gdi32.SetStretchBltMode(hdc, self._win.COLORONCOLOR)
            self._win.gdi32.StretchDIBits(
                hdc,
                0,
                0,
                target_width,
                target_height,
                0,
                0,
                buffer.width,
                buffer.height,
                ctypes.c_char_p(payload),
                ctypes.byref(info),
                self._win.DIB_RGB_COLORS,
                self._win.SRCCOPY,
            )
        finally:
            self._win.user32.ReleaseDC(self._hwnd, hdc)

    def after(self, milliseconds: int, callback: Callable[[], None]) -> None:
        self._scheduled.append((perf_counter() + max(1, int(milliseconds)) / 1000.0, callback))

    def run(self) -> None:
        while not self._closed:
            for event in self.poll_events():
                if event.kind == "key_down" and self._key_callback is not None:
                    self._key_callback(event.key)
            now = perf_counter()
            ready = [item for item in self._scheduled if item[0] <= now]
            self._scheduled[:] = [item for item in self._scheduled if item[0] > now]
            for _when, callback in ready:
                callback()
            sleep(0.001)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._win.user32.DestroyWindow(self._hwnd)

    def _register_class(self) -> None:
        wc = self._win.WNDCLASSW()
        wc.lpfnWndProc = ctypes.cast(self._wndproc, ctypes.c_void_p)
        wc.hInstance = self._win.hinstance
        wc.lpszClassName = self._class_name
        wc.hCursor = self._win.user32.LoadCursorW(None, self._win.IDC_ARROW)
        wc.hbrBackground = self._win.COLOR_WINDOW + 1
        atom = self._win.user32.RegisterClassW(ctypes.byref(wc))
        if not atom:
            raise ctypes.WinError()

    def _create_window(self):
        if self.fullscreen:
            width = self._win.user32.GetSystemMetrics(self._win.SM_CXSCREEN)
            height = self._win.user32.GetSystemMetrics(self._win.SM_CYSCREEN)
            style = self._win.WS_POPUP | self._win.WS_VISIBLE
            x = y = 0
        else:
            width = max(1, int(self.width))
            height = max(1, int(self.height))
            style = self._win.WS_OVERLAPPEDWINDOW | self._win.WS_VISIBLE
            x = y = self._win.CW_USEDEFAULT
        return self._win.user32.CreateWindowExW(
            0,
            self._class_name,
            self._title,
            style,
            x,
            y,
            width,
            height,
            None,
            None,
            self._win.hinstance,
            None,
        )

    def _handle_message(self, hwnd, message, wparam, lparam):
        if message == self._win.WM_CLOSE:
            self.close()
            return 0
        if message == self._win.WM_DESTROY:
            self._closed = True
            self._events.append(WindowEvent("quit"))
            return 0
        if message == self._win.WM_PAINT:
            paint = self._win.PAINTSTRUCT()
            self._win.user32.BeginPaint(hwnd, ctypes.byref(paint))
            self._win.user32.EndPaint(hwnd, ctypes.byref(paint))
            if self._last_frame is not None:
                self.show(self._last_frame)
            return 0
        if message == self._win.WM_SIZE:
            width = _loword(lparam)
            height = _hiword(lparam)
            self._events.append(WindowEvent("resize", size=(max(1, width), max(1, height))))
            return 0
        if message in (self._win.WM_KEYDOWN, self._win.WM_SYSKEYDOWN):
            self._events.append(WindowEvent("key_down", key=_key_name(int(wparam))))
            return 0
        if message in (self._win.WM_KEYUP, self._win.WM_SYSKEYUP):
            self._events.append(WindowEvent("key_up", key=_key_name(int(wparam))))
            return 0
        if message == self._win.WM_MOUSEMOVE:
            pos = (_signed_loword(lparam), _signed_hiword(lparam))
            last = self._last_mouse or pos
            rel = (pos[0] - last[0], pos[1] - last[1])
            self._last_mouse = pos
            self._events.append(WindowEvent("motion", pos=pos, rel=rel))
            return 0
        if message in self._win.MOUSE_BUTTON_DOWNS:
            button = self._win.MOUSE_BUTTON_DOWNS[message]
            self._events.append(WindowEvent("button", pos=(_signed_loword(lparam), _signed_hiword(lparam)), button=button))
            return 0
        if message in self._win.MOUSE_BUTTON_UPS:
            button = self._win.MOUSE_BUTTON_UPS[message]
            self._events.append(WindowEvent("button_up", pos=(_signed_loword(lparam), _signed_hiword(lparam)), button=button))
            return 0
        if message == self._win.WM_MOUSEWHEEL:
            delta = _signed_hiword(wparam)
            self._events.append(WindowEvent("wheel", y=1 if delta > 0 else -1 if delta < 0 else 0))
            return 0
        return self._win.user32.DefWindowProcW(hwnd, message, wparam, lparam)


def _bgra_bytes(buffer: PixelBuffer) -> bytes:
    raw_rgb = getattr(buffer.pixels, "raw_rgb_bytes", None)
    if callable(raw_rgb):
        rgb = raw_rgb()
        pixel_count = buffer.width * buffer.height
        return _rgb_to_bgra_bytes(rgb, pixel_count)

    payload = bytearray(len(buffer.pixels) * 4)
    for index, pixel in enumerate(buffer.pixels):
        offset = index * 4
        payload[offset] = pixel.b
        payload[offset + 1] = pixel.g
        payload[offset + 2] = pixel.r
        payload[offset + 3] = 255
    return bytes(payload)


def _rgb_to_bgra_bytes(rgb: bytes, pixel_count: int) -> bytes:
    numpy = _numpy()
    if numpy is not None:
        source = numpy.frombuffer(rgb, dtype=numpy.uint8).reshape((pixel_count, 3))
        payload = numpy.empty((pixel_count, 4), dtype=numpy.uint8)
        payload[:, 0] = source[:, 2]
        payload[:, 1] = source[:, 1]
        payload[:, 2] = source[:, 0]
        payload[:, 3] = 255
        return payload.tobytes()

    payload = bytearray(pixel_count * 4)
    payload[0::4] = rgb[2::3]
    payload[1::4] = rgb[1::3]
    payload[2::4] = rgb[0::3]
    payload[3::4] = b"\xff" * pixel_count
    return bytes(payload)


def _numpy():
    global _NUMPY, _NUMPY_CHECKED
    if _NUMPY_CHECKED:
        return _NUMPY
    _NUMPY_CHECKED = True
    try:
        import numpy
    except Exception:
        _NUMPY = None
    else:
        _NUMPY = numpy
    return _NUMPY


def _loword(value: int) -> int:
    return int(value) & 0xFFFF


def _hiword(value: int) -> int:
    return (int(value) >> 16) & 0xFFFF


def _signed_loword(value: int) -> int:
    word = _loword(value)
    return word - 0x10000 if word & 0x8000 else word


def _signed_hiword(value: int) -> int:
    word = _hiword(value)
    return word - 0x10000 if word & 0x8000 else word


def _key_name(vk: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits = "0123456789"
    if 0x41 <= vk <= 0x5A:
        return letters[vk - 0x41].lower()
    if 0x30 <= vk <= 0x39:
        return digits[vk - 0x30]
    names = {
        0x08: "backspace",
        0x09: "tab",
        0x0D: "return",
        0x10: "shift",
        0x11: "ctrl",
        0x12: "alt",
        0x1B: "escape",
        0x20: "space",
        0x21: "pageup",
        0x22: "pagedown",
        0x25: "left",
        0x26: "up",
        0x27: "right",
        0x28: "down",
        0xA0: "lshift",
        0xA1: "rshift",
        0xA2: "lctrl",
        0xA3: "rctrl",
        0xDB: "leftbracket",
        0xDD: "rightbracket",
    }
    return names.get(vk, f"vk_{vk}")


class _Win32Bindings:
    WNDPROC = _WINDOW_CALLBACK(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

    class WNDCLASSW(ctypes.Structure):
        _fields_ = (
            ("style", wintypes.UINT),
            ("lpfnWndProc", ctypes.c_void_p),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HICON),
            ("hCursor", wintypes.HCURSOR),
            ("hbrBackground", wintypes.HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        )

    class PAINTSTRUCT(ctypes.Structure):
        _fields_ = (
            ("hdc", wintypes.HDC),
            ("fErase", wintypes.BOOL),
            ("rcPaint", wintypes.RECT),
            ("fRestore", wintypes.BOOL),
            ("fIncUpdate", wintypes.BOOL),
            ("rgbReserved", ctypes.c_byte * 32),
        )

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = (
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        )

    class RGBQUAD(ctypes.Structure):
        _fields_ = (
            ("rgbBlue", ctypes.c_byte),
            ("rgbGreen", ctypes.c_byte),
            ("rgbRed", ctypes.c_byte),
            ("rgbReserved", ctypes.c_byte),
        )

    BITMAPINFO = None

    def __init__(self) -> None:
        self.user32 = ctypes.windll.user32
        self.gdi32 = ctypes.windll.gdi32
        self.kernel32 = ctypes.windll.kernel32
        self.hinstance = self.kernel32.GetModuleHandleW(None)
        self._configure_types()

    def _configure_types(self) -> None:
        self.user32.RegisterClassW.argtypes = [ctypes.POINTER(self.WNDCLASSW)]
        self.user32.RegisterClassW.restype = getattr(wintypes, "ATOM", wintypes.WORD)
        self.user32.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            wintypes.LPVOID,
        ]
        self.user32.CreateWindowExW.restype = wintypes.HWND
        self.user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        self.user32.DefWindowProcW.restype = ctypes.c_ssize_t
        self.user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
        self.user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.gdi32.StretchDIBits.argtypes = [
            wintypes.HDC,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(self.BITMAPINFO),
            wintypes.UINT,
            wintypes.DWORD,
        ]

    CW_USEDEFAULT = -2147483648
    SW_SHOW = 5
    PM_REMOVE = 0x0001
    IDC_ARROW = 32512
    COLOR_WINDOW = 5
    WS_OVERLAPPEDWINDOW = 0x00CF0000
    WS_VISIBLE = 0x10000000
    WS_POPUP = 0x80000000
    WM_CLOSE = 0x0010
    WM_DESTROY = 0x0002
    WM_PAINT = 0x000F
    WM_SIZE = 0x0005
    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101
    WM_SYSKEYDOWN = 0x0104
    WM_SYSKEYUP = 0x0105
    WM_MOUSEMOVE = 0x0200
    WM_LBUTTONDOWN = 0x0201
    WM_LBUTTONUP = 0x0202
    WM_RBUTTONDOWN = 0x0204
    WM_RBUTTONUP = 0x0205
    WM_MBUTTONDOWN = 0x0207
    WM_MBUTTONUP = 0x0208
    WM_MOUSEWHEEL = 0x020A
    SM_CXSCREEN = 0
    SM_CYSCREEN = 1
    BI_RGB = 0
    DIB_RGB_COLORS = 0
    SRCCOPY = 0x00CC0020
    COLORONCOLOR = 3

    MOUSE_BUTTON_DOWNS = {
        WM_LBUTTONDOWN: 1,
        WM_MBUTTONDOWN: 2,
        WM_RBUTTONDOWN: 3,
    }
    MOUSE_BUTTON_UPS = {
        WM_LBUTTONUP: 1,
        WM_MBUTTONUP: 2,
        WM_RBUTTONUP: 3,
    }


class _BITMAPINFO(ctypes.Structure):
    _fields_ = (
        ("bmiHeader", _Win32Bindings.BITMAPINFOHEADER),
        ("bmiColors", _Win32Bindings.RGBQUAD * 1),
    )


_Win32Bindings.BITMAPINFO = _BITMAPINFO
