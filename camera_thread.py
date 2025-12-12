from PyQt6.QtCore import QThread, pyqtSignal
import pypylon.pylon as pylon
from pypylon import genicam
import numpy as np
import logging

class CameraThread(QThread):
    new_image_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)
    
    def __init__(self, image_process_thread):
        super().__init__()
        self.image_process_thread = image_process_thread
        self.running = False
        self.camera = None
        self.frame_rate = 30.0
        self.logger = logging.getLogger(__name__)
        
        # 连接信号
        self.new_image_signal.connect(self.image_process_thread.update_image)
    
    def run(self):
        self.running = True
        try:
            #print("Starting camera initialization")
            
            # 获取传输层工厂
            tl_factory = pylon.TlFactory.GetInstance()
            
            # 查找所有可用设备
            devices = tl_factory.EnumerateDevices()
            
            #print(f"Number of cameras found: {len(devices)}")
            if not devices:
                self.error_signal.emit("No cameras found.")
                return
            
            # 创建和连接相机
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            print(f"Using device: {self.camera.GetDeviceInfo().GetModelName()}")
            
            # 打开相机
            self.camera.Open()
            #print(f"Camera {self.camera.GetDeviceInfo().GetModelName()} opened")
            
            # 配置相机设置
            self._configure_camera()
            
            # 开始抓取
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print("Started grabbing images")
            
            while self.running and self.camera.IsGrabbing():
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                
                if grab_result.GrabSucceeded():
                    # 复制数组以避免数据竞争
                    img_array = grab_result.Array.copy()
                    self.new_image_signal.emit(img_array)
                
                grab_result.Release()
                
        except genicam.GenericException as e:
            error_msg = f"GenICam exception: {e}"
            self.logger.error(error_msg)
            self.error_signal.emit(error_msg)
        except Exception as e:
            error_msg = f"Camera error: {e}"
            self.logger.error(error_msg)
            self.error_signal.emit(error_msg)
        finally:
            self._cleanup()
    
    def _configure_camera(self):
        """配置相机参数"""
        try:
            # 设置为连续采集
            if genicam.IsAvailable(self.camera.TriggerMode):
                self.camera.TriggerMode.SetValue("Off")
            
            # 心跳超时（适用于GigE相机）
            if (self.camera.GetDeviceInfo().GetDeviceClass() == "BaslerGigE" and 
                genicam.IsAvailable(self.camera.GevHeartbeatTimeout)):
                self.camera.GevHeartbeatTimeout.SetValue(1000)
            
            # 帧率设置
            if genicam.IsAvailable(self.camera.AcquisitionFrameRateEnable):
                self.camera.AcquisitionFrameRateEnable.SetValue(True)
                
                if genicam.IsAvailable(self.camera.AcquisitionFrameRateAbs):
                    self.camera.AcquisitionFrameRateAbs.SetValue(self.frame_rate)
                elif genicam.IsAvailable(self.camera.AcquisitionFrameRate):
                    self.camera.AcquisitionFrameRate.SetValue(self.frame_rate)
            
            # 像素格式
            if genicam.IsAvailable(self.camera.PixelFormat):
                self.camera.PixelFormat.SetValue("Mono8")
            
            # ROI设置 (1600x1200)
            if (genicam.IsAvailable(self.camera.Width) and 
                genicam.IsAvailable(self.camera.Height)):
                self.camera.OffsetX.SetValue(0)
                self.camera.OffsetY.SetValue(0)
                self.camera.Width.SetValue(1600)
                self.camera.Height.SetValue(1200)
            
            #print("Camera parameters set successfully")
            
        except genicam.GenericException as e:
            error_msg = f"Failed to configure camera: {e}"
            self.logger.error(error_msg)
            self.error_signal.emit(error_msg)
            raise
    
    def _cleanup(self):
        """清理相机资源"""
        try:
            if self.camera:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                    print("Stopped grabbing images")
                
                if self.camera.IsOpen():
                    print(f"Closing camera {self.camera.GetDeviceInfo().GetModelName()}")
                    self.camera.Close()
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def stop(self):
        self.running = False
        self.wait(3000)  # 等待最多3秒以便安全关闭