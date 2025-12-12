# image_process_thread.py
import numpy as np
import cv2
import queue
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

class ImageProcessThread(QThread):
    """图像处理线程 - 用于预处理相机图像"""
    processed_image_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.mutex = QMutex()
        self.image_queue = queue.Queue(maxsize=5)
        
    def run(self):
        """主线程循环"""
        self.running = True
        
        while self.running:
            try:
                # 从队列获取图像
                image = self.image_queue.get(timeout=0.1)
                
                # 处理图像
                processed = self.process_image(image)
                
                # 发送处理后的图像
                self.processed_image_signal.emit(processed)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Image processing error: {e}")
    
    def process_image(self, image):
        """处理图像"""
        if image is None:
            return None
        
        # 如果是灰度图，转换为RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        return image
    
    def update_image(self, image):
        """更新图像"""
        if not self.running:
            return
        
        try:
            # 如果队列满了，清除旧图像
            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.image_queue.put(image.copy())
        except Exception as e:
            print(f"Failed to update image: {e}")
    
    def stop(self):
        """停止线程"""
        self.running = False
        self.wait(1000)