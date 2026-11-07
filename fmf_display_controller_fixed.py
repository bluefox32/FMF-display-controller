#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FMF-Elastomer Display Control System
蛍光磁性流体-エラストマーディスプレイ統合制御

実装概要:
- 電場駆動型（推奨）& 磁場駆動型（オプション）に対応
- RGB → 駆動電圧/電流の変換
- リアルタイム画像表示
- キャリブレーション & 色彩管理
"""

import ast

import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, Dict
from enum import Enum
import threading
from abc import ABC, abstractmethod

# Optional: Image processing
try:
    from PIL import Image
    import cv2
    HAS_VISION = True
except ImportError:
    HAS_VISION = False


# ============================================================================
# 定数定義
# ============================================================================

class DriveMode(Enum):
    """駆動方式"""
    ELECTRIC_FIELD = "e_field"      # 電場駆動（推奨）
    MAGNETIC_FIELD = "b_field"      # 磁場駆動
    HYBRID = "hybrid"               # ハイブリッド


class ColorMode(Enum):
    """色彩モード"""
    RGB = "rgb"                     # 標準RGB
    HSV = "hsv"                     # HSV色空間
    CIE_XY = "cie_xy"               # CIE色度座標


# ============================================================================
# データクラス定義
# ============================================================================

@dataclass
class HardwareConfig:
    """ハードウェア設定"""
    resolution: Tuple[int, int] = (1920, 1080)     # 解像度
    pixel_pitch_mm: float = 0.2                     # ピクセルピッチ
    
    # 電場駆動パラメータ
    electric_field_max_kv_mm: float = 1.0           # 最大電場強度 (kV/mm)
    electric_field_min_kv_mm: float = 0.0           # 最小電場強度
    voltage_max_V: float = 1000                     # 最大電圧
    voltage_min_V: float = 0                        # 最小電圧
    voltage_resolution_levels: int = 256            # 電圧分解能レベル
    
    # 磁場駆動パラメータ
    magnetic_field_max_mT: float = 100              # 最大磁場 (mT)
    magnetic_field_min_mT: float = 0                # 最小磁場
    current_max_A: float = 5.0                      # 最大駆動電流
    
    # 層構成
    num_layers: int = 3                             # RGB層数
    layer_thickness_um: float = 50                  # 層厚 (μm)
    
    # 光学パラメータ
    uv_wavelength_nm: float = 365                   # UV励起波長
    uv_intensity_mw_cm2: float = 10                 # UV強度
    refresh_rate_hz: int = 10                       # リフレッシュレート


@dataclass
class LayerConfig:
    """蛍光層の設定"""
    layer_id: int                                   # 層ID (0=Blue, 1=Green, 2=Red)
    emission_peak_nm: float                         # 発光ピーク波長
    emission_fwhm_nm: float = 30                    # 発光帯域幅
    quantum_yield: float = 0.75                     # 量子収率
    eff_concentration_vol_percent: float = 5.0     # EFF濃度 (vol%)
    
    def __post_init__(self):
        # デフォルト設定
        if self.layer_id == 0:  # Blue
            self.emission_peak_nm = self.emission_peak_nm or 490
            self.eff_concentration_vol_percent = self.eff_concentration_vol_percent or 2.0
        elif self.layer_id == 1:  # Green
            self.emission_peak_nm = self.emission_peak_nm or 530
            self.eff_concentration_vol_percent = self.eff_concentration_vol_percent or 5.0
        elif self.layer_id == 2:  # Red
            self.emission_peak_nm = self.emission_peak_nm or 625
            self.eff_concentration_vol_percent = self.eff_concentration_vol_percent or 10.0


# ============================================================================
# 光学シミュレーションエンジン
# ============================================================================

class OpticalEngine:
    """FMF-エラストマー層の光学特性を計算"""
    
    def __init__(self, layers: List[LayerConfig]):
        self.layers = layers
        self._build_spectral_basis()
    
    def _build_spectral_basis(self):
        """スペクトラル基底を構築（計算高速化用）"""
        self.wavelengths = np.linspace(380, 750, 100)  # 可視光範囲
        self.layer_spectra = []
        
        for layer in self.layers:
            spectrum = self._gaussian_spectrum(
                layer.emission_peak_nm,
                layer.emission_fwhm_nm,
                self.wavelengths
            )
            self.layer_spectra.append(spectrum)
    
    def _gaussian_spectrum(self, peak: float, fwhm: float, wavelengths: np.ndarray) -> np.ndarray:
        """ガウス型発光スペクトラムを生成"""
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        spectrum = np.exp(-((wavelengths - peak) ** 2) / (2 * sigma ** 2))
        return spectrum / np.max(spectrum)  # 正規化
    
    def calculate_output_spectrum(self, 
                                  pixel_values: np.ndarray,
                                  drive_voltages: np.ndarray) -> np.ndarray:
        """
        駆動電圧に対応した出力スペクトラムを計算
        
        Args:
            pixel_values: RGB値 (0-255) [3,]
            drive_voltages: 駆動電圧 (0-1000V) [3,]
        
        Returns:
            出力スペクトラム [num_wavelengths,]
        """
        output = np.zeros_like(self.wavelengths, dtype=np.float32)
        
        for i, (layer, v, rgb) in enumerate(zip(self.layers, drive_voltages, pixel_values)):
            # 電圧 → 粒子集約度
            aggregation = v / 1000.0  # 正規化 (0-1)
            
            # 粒子集約による光学密度変化
            # 高電圧 → 粒子集約 → 散乱増加 → 蛍光効率低下
            effective_qy = layer.quantum_yield * (1 - 0.5 * aggregation)
            
            # RGB値による輝度調整
            brightness = (rgb / 255.0)
            
            # 出力スペクトラム合成
            contribution = (self.layer_spectra[i] * 
                          brightness * 
                          effective_qy * 
                          (1 - aggregation * 0.3))  # 散乱減衰
            
            output += contribution
        
        return np.clip(output, 0, 1)
    
    def spectrum_to_rgb(self, spectrum: np.ndarray) -> Tuple[float, float, float]:
        """
        スペクトラム → RGB色座標に変換
        CIE標準観察者応答関数を使用
        """
        # 簡略版: 波長ピークから推定
        peak_idx = np.argmax(spectrum)
        peak_wavelength = self.wavelengths[peak_idx]
        
        # 波長 → RGB変換（簡略化）
        if peak_wavelength < 450:
            return (255, 0, 255)  # 紫
        elif peak_wavelength < 495:
            return (0, 0, 255)    # 青
        elif peak_wavelength < 570:
            return (0, 255, 0)    # 緑
        elif peak_wavelength < 590:
            return (255, 255, 0)  # 黄
        elif peak_wavelength < 620:
            return (255, 128, 0)  # オレンジ
        else:
            return (255, 0, 0)    # 赤
    
    def estimate_luminance(self, spectrum: np.ndarray) -> float:
        """スペクトラムから相対輝度を推定 (0-100 cd/m²相当)"""
        # 人間視覚の相対分光感度V(λ)を簡略化
        v_lambda = np.exp(-((self.wavelengths - 555) ** 2) / (2 * 50 ** 2))
        luminance = np.sum(spectrum * v_lambda)
        return luminance * 100  # スケーリング
    
    def estimate_contrast_ratio(self, spectrum_max: np.ndarray, 
                               spectrum_min: np.ndarray) -> float:
        """コントラスト比を推定"""
        l_max = self.estimate_luminance(spectrum_max) + 1e-5
        l_min = self.estimate_luminance(spectrum_min) + 1e-5
        return l_max / l_min


# ============================================================================
# カラースペース変換エンジン
# ============================================================================

class ColorSpaceEngine:
    """RGB ↔ 駆動パラメータ変換"""
    
    def __init__(self, hw_config: HardwareConfig, optical_engine: OpticalEngine):
        self.hw = hw_config
        self.optical = optical_engine
        self._build_lut()
    
    def _build_lut(self):
        """ルックアップテーブル（色度 → 駆動電圧）を構築"""
        # 3層 × 256段階の駆動電圧組み合わせをシミュレート
        n_levels = self.hw.voltage_resolution_levels
        self.lut = {}
        
        for b in range(32, 256, 32):  # 青層サンプル
            for g in range(32, 256, 32):  # 緑層サンプル
                for r in range(32, 256, 32):  # 赤層サンプル
                    # 各RGBに対応した最適駆動電圧を探索
                    rgb = np.array([b, g, r], dtype=np.uint8)
                    best_voltages = self._find_optimal_voltage(rgb)
                    # FIX1: key in caller order (R, G, B); values stay
                    # layer-ordered [B, G, R]
                    self.lut[(r, g, b)] = best_voltages
    
    def _find_optimal_voltage(self, rgb: np.ndarray) -> np.ndarray:
        """
        RGBに最も近い出力を与える駆動電圧を探索
        （簡略版: RGB値から直線変換）
        """
        # FIX2: OpticalEngine declares high voltage -> aggregation ->
        # quenching (dimmer). The drive is therefore mapped in the
        # quench domain: V = (1 - value/255) * Vmax, so the composed
        # output stays monotonic in the requested value. (The former
        # direct map folded the output back at the top of the range.)
        voltages = (1.0 - rgb / 255.0) * self.hw.voltage_max_V
        return voltages.astype(np.uint16)
    
    def rgb_to_voltages(self, rgb: Tuple[int, int, int]) -> np.ndarray:
        """RGB値（0-255）→ 駆動電圧（0-1000V）に変換"""
        r, g, b = rgb
        
        # LUTで検索（無ければ補間）
        if (r, g, b) in self.lut:
            return self.lut[(r, g, b)]
        
        # 線形補間による推定
        rgb_array = np.array([b, g, r], dtype=np.float32)
        # FIX2 (same as _find_optimal_voltage): quench-domain map
        voltages = (1.0 - rgb_array / 255.0) * self.hw.voltage_max_V

        return voltages.astype(np.uint16)
    
    def rgb_to_currents(self, rgb: Tuple[int, int, int]) -> np.ndarray:
        """RGB値 → 磁場駆動用電流（磁場駆動時）"""
        # FIX3: coils are a separate low-voltage domain. Per-channel
        # full-scale coil voltage = current_max_A * R_coil, so the
        # current spans 0..current_max_A over the value range instead
        # of saturating at the clip (formerly the 1000 V field-domain
        # voltage was divided by the coil resistance directly).
        r, g, b = rgb
        layer_values = np.array([b, g, r], dtype=np.float32)  # [B, G, R]
        coil_resistance = np.array([10.0, 15.0, 20.0])
        coil_v_max = self.hw.current_max_A * coil_resistance
        currents = (layer_values / 255.0) * coil_v_max / coil_resistance
        return np.clip(currents, 0, self.hw.current_max_A)
    
    def rgb_to_cie_xy(self, rgb: Tuple[int, int, int]) -> Tuple[float, float]:
        """RGB → CIE 1931 (x, y) 色度座標"""
        # 簡略版: RGB から直接推定
        r, g, b = [x / 255.0 for x in rgb]
        
        # RGB → XYZ への変換（sRGB仮定）
        r = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
        
        x = r * 0.4124 + g * 0.3576 + b * 0.1805
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = r * 0.0193 + g * 0.1192 + b * 0.9505
        
        denom = x + y + z + 1e-5
        return (x / denom, y / denom)


# ============================================================================
# ハードウェアドライバ（抽象基底）
# ============================================================================

class DisplayHardwareDriver(ABC):
    """ハードウェア通信の抽象基底クラス"""
    
    def __init__(self, config: HardwareConfig, drive_mode: DriveMode):
        self.config = config
        self.drive_mode = drive_mode
        self.is_connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """デバイスに接続"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """デバイスから切断"""
        pass
    
    @abstractmethod
    def write_frame(self, voltage_frame: np.ndarray) -> bool:
        """
        フレームデータをハードウェアに書き込み
        
        Args:
            voltage_frame: 電圧マップ [height, width, 3]
        """
        pass
    
    @abstractmethod
    def read_status(self) -> Dict:
        """ハードウェアステータスを読み込み"""
        pass


class MockDisplayDriver(DisplayHardwareDriver):
    """シミュレーション用ドライバ（実機がない場合）"""
    
    def __init__(self, config: HardwareConfig, drive_mode: DriveMode):
        super().__init__(config, drive_mode)
        self.frame_buffer = None
        self.frame_count = 0
    
    def connect(self) -> bool:
        print(f"[MockDriver] Connecting to {self.drive_mode.value}...")
        self.is_connected = True
        return True
    
    def disconnect(self) -> bool:
        print("[MockDriver] Disconnecting...")
        self.is_connected = False
        return True
    
    def write_frame(self, voltage_frame: np.ndarray) -> bool:
        if not self.is_connected:
            return False
        
        self.frame_buffer = voltage_frame.copy()
        self.frame_count += 1
        
        if self.frame_count % 30 == 0:  # 30フレームごとに進捗表示
            print(f"[MockDriver] Frame {self.frame_count}: "
                  f"Vmin={voltage_frame.min():.1f}V, "
                  f"Vmax={voltage_frame.max():.1f}V")
        
        return True
    
    def read_status(self) -> Dict:
        return {
            "connected": self.is_connected,
            "frame_count": self.frame_count,
            "drive_mode": self.drive_mode.value,
            "resolution": self.config.resolution
        }


class RealDisplayDriver(DisplayHardwareDriver):
    """実ハードウェア用ドライバ（USB/UART通信）"""
    
    def __init__(self, config: HardwareConfig, drive_mode: DriveMode, port: str = "/dev/ttyUSB0"):
        super().__init__(config, drive_mode)
        self.port = port
        self.serial_conn = None
        self.cmd_buffer = []
    
    def connect(self) -> bool:
        try:
            import serial
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=921600,
                timeout=1.0
            )
            self.is_connected = True
            print(f"[RealDriver] Connected to {self.port} at 921600 baud")
            return True
        except ImportError:
            print("[RealDriver] pyserial not available - using Mock mode")
            return False
        except Exception as e:
            print(f"[RealDriver] Connection failed: {e}")
            return False
    
    def disconnect(self) -> bool:
        if self.serial_conn:
            self.serial_conn.close()
            self.is_connected = False
        return True
    
    def write_frame(self, voltage_frame: np.ndarray) -> bool:
        """
        フレームデータをシリアルで送信
        プロトコル: [SYNC(2)] [LEN(4)] [DATA(N)] [CRC(2)]
        FIX4: LEN is 4 bytes (a 1920x1080x3 frame is 6.2 MB and did
        not fit the former 2-byte field); CRC now follows DATA as
        declared; voltages are quantised to the declared
        voltage_resolution_levels (256) instead of wrapping mod 256.
        """
        if not self.is_connected:
            return False

        # 電圧 (0..voltage_max_V) → 宣言分解能レベル (0..255) に量子化
        levels = self.config.voltage_resolution_levels - 1
        frame_u8 = np.rint(
            np.clip(voltage_frame, 0, self.config.voltage_max_V)
            / self.config.voltage_max_V * levels
        ).astype(np.uint8)
        frame_data = frame_u8.tobytes()

        # フレームヘッダ構築
        frame_len = len(frame_data)
        header = bytearray([0xAA, 0x55])  # Sync word
        header.extend(frame_len.to_bytes(4, 'little'))

        # CRC計算（簡略版）: DATA の後に付ける
        crc = sum(frame_data) & 0xFFFF
        trailer = crc.to_bytes(2, 'little')

        # 送信
        try:
            self.serial_conn.write(header + frame_data + trailer)
            return True
        except Exception as e:
            print(f"[RealDriver] Write error: {e}")
            return False
    
    def read_status(self) -> Dict:
        if not self.is_connected:
            return {"connected": False}
        
        try:
            # ステータスクエリ送信
            self.serial_conn.write(b'\xAA\x55\x00\x00')
            response = self.serial_conn.read(16)
            
            if len(response) >= 16:
                return {
                    "connected": True,
                    "temperature_C": int.from_bytes(response[0:2], 'little') * 0.1,
                    "voltage_rail_V": int.from_bytes(response[2:4], 'little') * 0.01,
                    "frame_rate_hz": int.from_bytes(response[4:6], 'little'),
                }
        except:
            pass
        
        return {"connected": self.is_connected}


# ============================================================================
# メインディスプレイコントローラ
# ============================================================================

class FMFDisplay:
    """FMFエラストマーディスプレイの統合制御クラス"""
    
    def __init__(self, 
                 hw_config: HardwareConfig,
                 drive_mode: DriveMode = DriveMode.ELECTRIC_FIELD,
                 use_real_hardware: bool = False):
        """
        Args:
            hw_config: ハードウェア設定
            drive_mode: 駆動方式（電場/磁場/ハイブリッド）
            use_real_hardware: 実ハードウェアを使用するか
        """
        self.config = hw_config
        self.drive_mode = drive_mode
        
        # 光学エンジン初期化
        self.layers = [
            LayerConfig(0, 490, eff_concentration_vol_percent=2.0),   # Blue
            LayerConfig(1, 530, eff_concentration_vol_percent=5.0),   # Green
            LayerConfig(2, 625, eff_concentration_vol_percent=10.0),  # Red
        ]
        self.optical_engine = OpticalEngine(self.layers)
        self.color_engine = ColorSpaceEngine(hw_config, self.optical_engine)
        
        # ハードウェアドライバ初期化
        if use_real_hardware:
            self.hw_driver = RealDisplayDriver(hw_config, drive_mode)
        else:
            self.hw_driver = MockDisplayDriver(hw_config, drive_mode)
        
        # 状態管理
        self.frame_buffer = np.zeros((*hw_config.resolution, 3), dtype=np.uint16)
        self.is_running = False
        self.frame_rate = hw_config.refresh_rate_hz
        self.frame_count = 0
        self.stats = {"frames": 0, "dropped": 0, "avg_latency_ms": 0}
        
        # キャリブレーション用
        self.calibration_data = None
    
    def connect(self) -> bool:
        """デバイスに接続"""
        success = self.hw_driver.connect()
        if success:
            print(f"[FMFDisplay] Connected ({self.drive_mode.value})")
        return success
    
    def disconnect(self) -> bool:
        """デバイスから切断"""
        self.stop()
        return self.hw_driver.disconnect()
    
    def display_solid_color(self, rgb: Tuple[int, int, int], duration_sec: float = 5.0):
        """
        単一色を表示
        
        Args:
            rgb: (R, G, B) 値 (0-255)
            duration_sec: 表示時間
        """
        voltages = self.color_engine.rgb_to_voltages(rgb)
        
        # フレーム全体に同じ電圧を適用
        frame = np.tile(voltages, (*self.config.resolution, 1))
        
        # 表示
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            self._write_frame(frame)
            time.sleep(1.0 / self.frame_rate)
    
    def display_gradient(self, duration_sec: float = 5.0):
        """
        グラデーション表示テスト
        左から右へ色が変わる
        """
        h, w, _ = self.frame_buffer.shape
        
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            for x in range(w):
                # 横方向グラデーション（青→緑→赤）
                ratio = x / w
                
                if ratio < 0.33:
                    rgb = (0, 0, int(255 * ratio / 0.33))
                elif ratio < 0.66:
                    rgb = (0, int(255 * (ratio - 0.33) / 0.33), 255)
                else:
                    rgb = (int(255 * (ratio - 0.66) / 0.34), 0, 255)
                
                voltages = self.color_engine.rgb_to_voltages(rgb)
                self.frame_buffer[:, x] = voltages
            
            self._write_frame(self.frame_buffer)
            time.sleep(1.0 / self.frame_rate)
    
    def display_checkerboard(self, block_size: int = 32, duration_sec: float = 5.0):
        """
        チェッカーボード表示テスト
        """
        h, w, _ = self.frame_buffer.shape
        
        colors = [
            (255, 0, 0),     # 赤
            (0, 255, 0),     # 緑
            (0, 0, 255),     # 青
            (255, 255, 0),   # 黄
            (255, 0, 255),   # マゼンタ
            (0, 255, 255),   # シアン
        ]
        
        start_time = time.time()
        frame_idx = 0
        
        while time.time() - start_time < duration_sec:
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    color_idx = ((y // block_size) + (x // block_size) + frame_idx) % len(colors)
                    rgb = colors[color_idx]
                    voltages = self.color_engine.rgb_to_voltages(rgb)
                    
                    y_end = min(y + block_size, h)
                    x_end = min(x + block_size, w)
                    self.frame_buffer[y:y_end, x:x_end] = voltages
            
            self._write_frame(self.frame_buffer)
            frame_idx += 1
            time.sleep(1.0 / self.frame_rate)
    
    def display_image(self, image_path: str, duration_sec: float = 10.0):
        """
        画像ファイルを表示
        
        Args:
            image_path: 画像ファイルパス
            duration_sec: 表示時間
        """
        if not HAS_VISION:
            print("[FMFDisplay] PIL/OpenCV not available")
            return
        
        # 画像読み込み
        img = Image.open(image_path)
        img = img.resize(self.config.resolution)
        img_array = np.array(img)
        
        if len(img_array.shape) == 2:  # グレースケール
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # RGB値 → 駆動電圧に変換
        h, w, c = img_array.shape
        frame = np.zeros((*self.config.resolution, 3), dtype=np.uint16)
        
        for y in range(h):
            for x in range(w):
                rgb = tuple(img_array[y, x, :3].astype(int))
                frame[y, x] = self.color_engine.rgb_to_voltages(rgb)
        
        # 表示
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            self._write_frame(frame)
            time.sleep(1.0 / self.frame_rate)
    
    def display_spectrum_sweep(self, duration_sec: float = 10.0):
        """
        スペクトラム走査テスト
        （各層の蛍光色を段階的に表示）
        """
        layers_colors = [
            (0, 0, 255),      # Blue
            (0, 255, 0),      # Green
            (255, 0, 0),      # Red
        ]
        
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            for color in layers_colors:
                self.display_solid_color(color, duration_sec=0.5)
    
    def _write_frame(self, frame: np.ndarray) -> bool:
        """フレームバッファをハードウェアに書き込み"""
        t0 = time.time()
        
        success = self.hw_driver.write_frame(frame)
        
        if success:
            self.frame_count += 1
            latency_ms = (time.time() - t0) * 1000
            self.stats["frames"] += 1
            
            # 移動平均で遅延を更新
            alpha = 0.1
            self.stats["avg_latency_ms"] = (
                alpha * latency_ms + 
                (1 - alpha) * self.stats.get("avg_latency_ms", 0)
            )
        else:
            self.stats["dropped"] += 1
        
        return success
    
    def start(self):
        """ディスプレイの駆動開始"""
        self.is_running = True
        print("[FMFDisplay] Display started")
    
    def stop(self):
        """ディスプレイの駆動停止"""
        self.is_running = False
        print("[FMFDisplay] Display stopped")
    
    def get_stats(self) -> Dict:
        """統計情報を取得"""
        stats = self.stats.copy()
        stats["hw_status"] = self.hw_driver.read_status()
        return stats
    
    def save_calibration(self, filepath: str):
        """キャリブレーションデータを保存"""
        calib = {
            "color_lut": {str(k): v.tolist() for k, v in self.color_engine.lut.items()},
            "layer_config": [asdict(layer) for layer in self.layers],
            "hw_config": asdict(self.config),
        }
        
        with open(filepath, 'w') as f:
            json.dump(calib, f, indent=2)
        
        print(f"[FMFDisplay] Calibration saved to {filepath}")
    
    def load_calibration(self, filepath: str):
        """キャリブレーションデータを読み込み"""
        with open(filepath, 'r') as f:
            calib = json.load(f)
        
        # 修正 (2026-11-07): キーは str(tuple) 形式のみ。eval は JSON を
        # 読むだけで任意コード実行になるため ast.literal_eval に置換
        self.color_engine.lut = {
            ast.literal_eval(k): np.array(v) for k, v in calib.get("color_lut", {}).items()
        }
        
        print(f"[FMFDisplay] Calibration loaded from {filepath}")


# ============================================================================
# メイン実装例
# ============================================================================

def main():
    """ディスプレイ制御のデモンストレーション"""
    
    print("=" * 70)
    print("FMF-Elastomer Display Control System")
    print("=" * 70)
    
    # ハードウェア設定
    config = HardwareConfig(
        resolution=(1920, 1080),
        pixel_pitch_mm=0.2,
        electric_field_max_kv_mm=1.0,
        voltage_max_V=1000,
        refresh_rate_hz=10,  # 低リフレッシュレート
    )
    
    # ディスプレイ初期化
    display = FMFDisplay(
        hw_config=config,
        drive_mode=DriveMode.ELECTRIC_FIELD,
        use_real_hardware=False  # シミュレーションモード
    )
    
    # 接続
    if not display.connect():
        print("[ERROR] Connection failed")
        return
    
    # デモンストレーション実行
    print("\n[DEMO] Solid Color Display")
    display.display_solid_color((255, 0, 0), duration_sec=2)  # 赤
    
    print("\n[DEMO] Color Gradient")
    display.display_gradient(duration_sec=3)
    
    print("\n[DEMO] Checkerboard Pattern")
    display.display_checkerboard(block_size=64, duration_sec=3)
    
    print("\n[DEMO] Spectrum Sweep")
    display.display_spectrum_sweep(duration_sec=3)
    
    # 統計情報表示
    print("\n" + "=" * 70)
    print("Display Statistics:")
    print("=" * 70)
    stats = display.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # キャリブレーション保存
    display.save_calibration("fmf_calibration.json")
    
    # 切断
    display.disconnect()
    print("\n[FMFDisplay] Demo completed")


if __name__ == "__main__":
    main()
