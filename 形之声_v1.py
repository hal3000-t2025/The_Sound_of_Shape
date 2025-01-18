# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny as edge
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import threading
import time
from datetime import datetime, timedelta
from skimage.morphology import skeletonize, thin, binary_dilation, binary_erosion
from scipy import ndimage
import os
import sys

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 如果是macOS系统，使用不同的字体
if sys.platform == 'darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 设置中文显示
if sys.platform.startswith('win'):
    # Windows系统
    from tkinter import font
    def create_font():
        return font.Font(family='Microsoft YaHei UI', size=9)
else:
    # macOS/Linux系统
    def create_font():
        return None

def resource_path(relative_path):
    """获取资源文件的绝对路径"""
    try:
        # PyInstaller创建临时文件夹,将路径存储在_MEIPASS中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class ProgressDialog:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Converting...")
        
        # 设置窗口大小和位置
        window_width = 400
        window_height = 150
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # 设置为模态窗口
        self.window.transient(parent)
        self.window.grab_set()
        
        # 禁用关闭按钮
        self.window.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # 创建进度显示组件
        self.progress_var = tk.DoubleVar()
        self.progress_label = tk.Label(self.window, text="Processing: 0%")
        self.progress_label.pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(
            self.window,
            variable=self.progress_var,
            maximum=100,
            length=300,
            mode='determinate'
        )
        self.progress_bar.pack(pady=10)
        
        self.time_label = tk.Label(self.window, text="Remaining: calculating...")
        self.time_label.pack(pady=10)

    def update(self, progress, remaining):
        self.progress_var.set(progress)
        self.progress_label.config(text=f"Processing: {progress:.1f}%")
        self.time_label.config(text=f"Remaining: {remaining}")

    def close(self):
        self.window.grab_release()
        self.window.destroy()

class LocalImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("形之声V1.0 - 视频边缘音频转换器")
        
        # 设置应用图标
        try:
            if sys.platform.startswith('win'):
                self.root.iconbitmap(resource_path('app_icon.ico'))
            elif sys.platform == 'darwin':  # macOS
                # macOS 使用 .icns 格式
                icon_path = resource_path('app_icon.icns')
                if os.path.exists(icon_path):
                    self.root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Failed to load icon: {e}")
        
        # 设置窗口大小和位置
        window_width = 800
        window_height = 600
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # 添加窗口关闭确认
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 设置默认字体
        self.default_font = create_font()
        if self.default_font:
            self.root.option_add('*Font', self.default_font)
        
        self.current_image = None
        self.edges = None
        self.video_path = None
        self.video_cap = None
        self.is_processing = False
        self.progress_dialog = None
        self.use_thinning = tk.BooleanVar(value=False)
        self.manga_threshold = tk.DoubleVar(value=0.2)
        self.line_thickness = tk.IntVar(value=1)
        self.smooth_factor = tk.DoubleVar(value=0.5)
        
        # 添加参数保存功能
        self.last_used_params = {
            'sigma': 1.0,
            'manga_threshold': 0.2,
            'line_thickness': 1,
            'smooth_factor': 0.5,
            'use_thinning': False
        }
        
        self.create_widgets()

    def create_widgets(self):
        # 主菜单栏
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开图片", command=self.open_image)
        file_menu.add_command(label="打开视频", command=self.open_video)
        file_menu.add_separator()
        file_menu.add_command(label="保存结果", command=self.save_result)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 参数菜单
        param_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="参数", menu=param_menu)
        param_menu.add_command(label="保存当前参数", command=self.save_parameters)
        param_menu.add_command(label="加载上次参数", command=self.load_parameters)
        param_menu.add_command(label="重置为默认", command=self.reset_parameters)

        # 添加帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)

        # 主控制区域
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        # 图片处理按钮
        image_frame = tk.LabelFrame(control_frame, text="图片处理")
        image_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Button(image_frame, text="打开图片", command=self.open_image).pack(side=tk.LEFT, padx=5)
        tk.Button(image_frame, text="保存图片", command=self.save_result).pack(side=tk.LEFT, padx=5)

        # 视频处理按钮
        video_frame = tk.LabelFrame(control_frame, text="视频处理")
        video_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Button(video_frame, text="打开视频", command=self.open_video).pack(side=tk.LEFT, padx=5)
        tk.Button(video_frame, text="转换为WAV", command=self.start_video_conversion).pack(side=tk.LEFT, padx=5)

        # 参数控制区域
        param_frame = tk.Frame(self.root)
        param_frame.pack(pady=5)
        
        # Sigma控制
        sigma_frame = tk.LabelFrame(param_frame, text="边缘检测")
        sigma_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(sigma_frame, text="模糊度:").pack(side=tk.LEFT)
        self.sigma_var = tk.DoubleVar(value=1.0)
        sigma_entry = tk.Entry(sigma_frame, textvariable=self.sigma_var, width=5)
        sigma_entry.pack(side=tk.LEFT, padx=5)
        
        # 漫画模式控制
        manga_frame = tk.LabelFrame(param_frame, text="漫画模式")
        manga_frame.pack(side=tk.LEFT, padx=10)
        
        # Enable 开关
        tk.Checkbutton(
            manga_frame, 
            text="启用", 
            variable=self.use_thinning,
            command=self.update_edge_detection
        ).pack(side=tk.LEFT, padx=5)
        
        # 阈值控制
        threshold_frame = tk.Frame(manga_frame)
        threshold_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(threshold_frame, text="阈值:").pack(side=tk.TOP)
        tk.Scale(
            threshold_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.manga_threshold,
            command=lambda x: self.delayed_update()
        ).pack(side=tk.TOP)
        
        # 线条粗细控制
        thickness_frame = tk.Frame(manga_frame)
        thickness_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(thickness_frame, text="线条粗细:").pack(side=tk.TOP)
        tk.Scale(
            thickness_frame,
            from_=1,
            to=5,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.line_thickness,
            command=lambda x: self.delayed_update()
        ).pack(side=tk.TOP)
        
        # 平滑度控制
        smooth_frame = tk.Frame(manga_frame)
        smooth_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(smooth_frame, text="平滑度:").pack(side=tk.TOP)
        tk.Scale(
            smooth_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.smooth_factor,
            command=lambda x: self.delayed_update()
        ).pack(side=tk.TOP)

        # Apply按钮
        tk.Button(param_frame, text="应用", command=self.update_edge_detection).pack(side=tk.LEFT, padx=10)

        # 进度显示框架
        self.progress_frame = tk.LabelFrame(self.root, text="转换进度")
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame, 
            variable=self.progress_var,
            maximum=100,
            length=300,
            mode='determinate'
        )
        
        # 进度文本和剩余时间
        self.progress_label = tk.Label(self.progress_frame, text="0%")
        self.time_label = tk.Label(self.progress_frame, text="Remaining: --:--")

        # 视频预览
        self.video_label = tk.Label(self.root)
        self.video_label.pack(pady=10)

        # 图像预览区域
        preview_frame = tk.Frame(self.root)
        preview_frame.pack(pady=10)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.ax1.set_title("原图")
        self.ax2.set_title("边缘检测")
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=preview_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

        # 状态栏
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def delayed_update(self):
        """使用延迟更新避免频繁刷新"""
        if hasattr(self, '_update_job'):
            self.root.after_cancel(self._update_job)
        self._update_job = self.root.after(200, self.update_edge_detection)

    def save_parameters(self):
        """保存当前参数设置"""
        self.last_used_params = {
            'sigma': self.sigma_var.get(),
            'manga_threshold': self.manga_threshold.get(),
            'line_thickness': self.line_thickness.get(),
            'smooth_factor': self.smooth_factor.get(),
            'use_thinning': self.use_thinning.get()
        }
        self.update_status("Parameters saved")
        
    def load_parameters(self):
        """加载上次的参数设置"""
        self.sigma_var.set(self.last_used_params['sigma'])
        self.manga_threshold.set(self.last_used_params['manga_threshold'])
        self.line_thickness.set(self.last_used_params['line_thickness'])
        self.smooth_factor.set(self.last_used_params['smooth_factor'])
        self.use_thinning.set(self.last_used_params['use_thinning'])
        self.update_edge_detection()
        self.update_status("Parameters loaded")
        
    def reset_parameters(self):
        """重置为默认参数"""
        self.sigma_var.set(1.0)
        self.manga_threshold.set(0.2)
        self.line_thickness.set(1)
        self.smooth_factor.set(0.5)
        self.use_thinning.set(False)
        self.update_edge_detection()
        self.update_status("Parameters reset to default")

    def update_status(self, message):
        """更新状态栏信息"""
        self.status_var.set(message)

    def open_video(self):
        """打开视频文件"""
        video_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        
        if video_path:
            # 关闭之前的视频
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None
            
            self.video_path = video_path
            self.video_cap = cv2.VideoCapture(video_path)
            
            if not self.video_cap.isOpened():
                messagebox.showerror("Error", "Failed to open video file")
                return
            
            ret, frame = self.video_cap.read()
            if ret:
                # 显示视频预览
                self.show_video_preview()
                
                # 将第一帧作为当前图片并进行边缘检测
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_image = frame_rgb
                self.update_edge_detection()
            else:
                messagebox.showerror("Error", "Failed to read video frame")

    def show_video_preview(self):
        if self.video_cap is None or not self.video_cap.isOpened():
            return

        ret, frame = self.video_cap.read()
        if ret:
            # 调整预览大小
            height, width = frame.shape[:2]
            max_size = 300
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

            # 转换颜色空间并显示
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk

    def start_video_conversion(self):
        if self.video_path is None:
            messagebox.showerror("Error", "Please open a video first!")
            return
            
        if self.is_processing:
            messagebox.showwarning("Warning", "Conversion already in progress!")
            return

        # 创建进度弹窗
        self.progress_dialog = ProgressDialog(self.root)
        
        # 在新线程中启动转换
        self.is_processing = True
        thread = threading.Thread(target=self.convert_video_to_wav)
        thread.start()

    def sort_edge_points(self, edge_points):
        """对边缘点进行排序，确保相邻点之间的距离最小"""
        if len(edge_points) <= 2:
            return edge_points
        
        # 找到最左边的点作为起始点
        start_idx = np.argmin(edge_points[:, 0])
        sorted_points = [edge_points[start_idx]]
        remaining_points = np.delete(edge_points, start_idx, axis=0)
        
        # 依次找到距离最近的点
        while len(remaining_points) > 0:
            last_point = sorted_points[-1]
            # 计算所有剩余点到当前点的距离
            distances = np.sum((remaining_points - last_point) ** 2, axis=1)
            next_idx = np.argmin(distances)
            
            # 添加最近的点
            sorted_points.append(remaining_points[next_idx])
            remaining_points = np.delete(remaining_points, next_idx, axis=0)
        
        return np.array(sorted_points)

    def convert_video_to_wav(self):
        try:
            save_path = filedialog.asksaveasfilename(
                title="Save WAV file",
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
            )
            
            if not save_path:
                self.finish_conversion()
                return

            # 重新打开视频文件
            if self.video_cap is not None:
                self.video_cap.release()
            self.video_cap = cv2.VideoCapture(self.video_path)
            
            if not self.video_cap.isOpened():
                raise Exception("Failed to reopen video file")

            # 获取视频原始参数
            current_sigma = self.sigma_var.get()
            original_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / original_fps

            if total_frames == 0 or original_fps == 0:
                raise Exception("Invalid video parameters")

            print(f"Original video: {total_frames} frames, {original_fps} fps, duration: {video_duration:.2f}s")

            # 设置压缩后的尺寸
            original_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if original_height >= original_width:
                height = 256
                width = round(original_width / original_height * height)
            else:
                width = 256
                height = round(original_height / original_width * width)

            # 更新进度显示为压缩阶段
            self.root.after(0, self.update_progress, 0, "Compressing video...")
            
            # 压缩视频
            compressed_frames = []
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            for i in range(total_frames):
                if not self.is_processing:
                    break
                    
                ret, frame = self.video_cap.read()
                if ret:
                    # 压缩帧
                    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    compressed_frames.append(resized_frame)
                    
                    # 更新压缩进度（0-20%）
                    progress = (i + 1) / total_frames * 20
                    self.root.after(0, self.update_progress, progress, "Compressing video...")

            if not self.is_processing:
                self.finish_conversion()
                return

            # 处理压缩后的帧
            edge_arrays = []
            start_time = datetime.now()
            
            for i, frame in enumerate(compressed_frames):
                if not self.is_processing:
                    break
                    
                # 转换为RGB并进行边缘检测
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = rgb2gray(frame_rgb)
                edges = edge(gray, sigma=current_sigma)
                
                # 应用细化处理（如果启用）
                edges = self.process_edges(edges)
                
                # 获取边缘点坐标
                y, x = np.where(edges == True)
                edge_points = np.column_stack((x, height - y))
                
                # 使用改进的排序方法
                if len(edge_points) > 0:
                    edge_points = self.sort_edge_points(edge_points)
                
                edge_arrays.append(edge_points)
                
                # 更新进度（20-70%）
                progress = 20 + (i + 1) / len(compressed_frames) * 50
                elapsed = datetime.now() - start_time
                estimated_total = elapsed / ((progress - 20) / 50)
                remaining = estimated_total - elapsed
                remaining_str = str(timedelta(seconds=int(remaining.total_seconds())))
                
                if i % 5 == 0:  # 减少进度更新频率
                    self.root.after(0, self.update_progress, progress, remaining_str)

            # WAV文件生成部分
            if self.is_processing and edge_arrays:
                self.root.after(0, self.update_progress, 70, "Generating WAV...")
                
                # WAV文件参数
                sampling_rate = 88200
                sound_channel = 2
                bits_per_sample = 8
                
                # 计算每帧的采样点数，使总时长与视频相同
                total_samples = int(sampling_rate * video_duration)
                samp_per_frame = int(total_samples / len(edge_arrays))
                
                print(f"Audio parameters: {sampling_rate}Hz, {samp_per_frame} samples per frame")
                print(f"Total audio samples: {total_samples}, expected duration: {total_samples/sampling_rate:.2f}s")
                
                data_size = total_samples * 2  # 双声道
                file_size = data_size + 36
                byte_rate = sampling_rate * 2
                block_align = sound_channel * bits_per_sample // 8

                # 创建WAV头
                header = np.array([
                    82, 73, 70, 70,  # "RIFF"
                    file_size % 256, (file_size // 256) % 256, (file_size // 65536) % 256, file_size // 16777216,
                    87, 65, 86, 69,  # "WAVE"
                    102, 109, 116, 32,  # "fmt "
                    16, 0, 0, 0,  # fmt chunk size
                    1, 0,  # PCM format
                    sound_channel % 256, sound_channel // 256,
                    sampling_rate % 256, (sampling_rate // 256) % 256, (sampling_rate // 65536) % 256, sampling_rate // 16777216,
                    byte_rate % 256, (byte_rate // 256) % 256, (byte_rate // 65536) % 256, byte_rate // 16777216,
                    block_align % 256, block_align // 256,
                    bits_per_sample % 256, bits_per_sample // 256,
                    100, 97, 116, 97,  # "data"
                    data_size % 256, (data_size // 256) % 256, (data_size // 65536) % 256, data_size // 16777216
                ], dtype=np.uint8)

                # 创建音频数据
                audio_data = np.zeros(data_size, dtype=np.uint8)
                
                # 生成音频数据
                for i, edge_points in enumerate(edge_arrays):
                    if not self.is_processing:
                        break
                        
                    len_edge = len(edge_points)
                    if len_edge > 0:
                        start_idx = i * samp_per_frame
                        end_idx = start_idx + samp_per_frame
                        if end_idx > total_samples:
                            end_idx = total_samples
                            
                        for j in range(start_idx, end_idx):
                            idx = ((j - start_idx) * len_edge) // samp_per_frame
                            audio_data[j * 2] = edge_points[idx, 0]
                            audio_data[j * 2 + 1] = edge_points[idx, 1]
                    elif i > 0:  # 如果当前帧没有边缘点，使用上一帧的最后一个点
                        last_points = edge_arrays[i-1]
                        if len(last_points) > 0:
                            start_idx = i * samp_per_frame
                            end_idx = start_idx + samp_per_frame
                            if end_idx > total_samples:
                                end_idx = total_samples
                                
                            for j in range(start_idx, end_idx):
                                audio_data[j * 2] = last_points[-1, 0]
                                audio_data[j * 2 + 1] = last_points[-1, 1]

                    # 更新进度（70-100%）
                    progress = 70 + (i + 1) / len(edge_arrays) * 30
                    if i % 10 == 0:
                        elapsed = datetime.now() - start_time
                        estimated_total = elapsed / ((progress - 70) / 30)
                        remaining = estimated_total - elapsed
                        remaining_str = str(timedelta(seconds=int(remaining.total_seconds())))
                        self.root.after(0, self.update_progress, progress, remaining_str)

                if self.is_processing:
                    # 合并头部和数据并写入文件
                    wav_data = np.concatenate((header, audio_data))
                    with open(save_path, 'wb') as f:
                        f.write(wav_data.tobytes())
                    
                    self.root.after(0, self.update_progress, 100, "Complete!")
                    messagebox.showinfo("Success", "Video converted successfully!")
            
        except Exception as e:
            error_msg = f"Conversion failed: {str(e)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
        finally:
            self.finish_conversion()

    def update_progress(self, progress, remaining):
        """更新进度显示"""
        if self.progress_dialog:
            self.progress_dialog.update(progress, remaining)

    def finish_conversion(self):
        """完成转换后的清理工作"""
        self.is_processing = False
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None

    def open_image(self):
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not image_path:
            return

        try:
            print(f"Opening image: {image_path}")
            img = Image.open(image_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img)
            self.current_image = self.process_image(img_array)  # 保存处理后的图片
            
            # 进行初始边缘检测
            self.update_edge_detection()
            
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def process_image(self, img_array):
        # 添加图片尺寸限制
        max_size = 1024
        h, w = img_array.shape[:2]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img = Image.fromarray(img_array)
            img_array = np.array(img.resize(new_size))
        return img_array

    def save_result(self):
        save_path = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if save_path:
            try:
                self.fig.savefig(save_path)
                messagebox.showinfo("Success", "Result saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save result: {str(e)}")

    def process_edges(self, edges):
        """改进的边缘处理方法"""
        if self.use_thinning.get():
            try:
                # 获取参数
                threshold = self.manga_threshold.get()
                thickness = self.line_thickness.get()
                smoothness = self.smooth_factor.get()
                
                # 1. 应用阈值
                binary_edges = edges > threshold
                
                # 2. 平滑处理
                if smoothness > 0:
                    sigma = smoothness * 2
                    binary_edges = ndimage.gaussian_filter(binary_edges.astype(float), sigma)
                    binary_edges = binary_edges > 0.5
                
                # 3. 骨架提取
                thinned = skeletonize(binary_edges)
                
                # 4. 线条增粗（如果需要）
                if thickness > 1:
                    for _ in range(thickness - 1):
                        thinned = binary_dilation(thinned)
                
                # 5. 最终平滑处理
                if smoothness > 0:
                    thinned = ndimage.gaussian_filter(thinned.astype(float), smoothness)
                    thinned = thinned > 0.5
                
                self.update_status(
                    f"Manga mode: threshold={threshold:.2f}, "
                    f"thickness={thickness}, smoothness={smoothness:.2f}"
                )
                return thinned.astype(np.bool_)
                
            except Exception as e:
                self.update_status(f"Error in manga processing: {str(e)}")
                return edges
        return edges

    def update_edge_detection(self):
        if self.current_image is None:
            print("No image loaded")
            return

        try:
            print(f"Updating edge detection with sigma = {self.sigma_var.get()}")
            print(f"Manga mode: {self.use_thinning.get()}, threshold: {self.manga_threshold.get()}")
            
            # 转换为灰度图像
            gray_img = rgb2gray(self.current_image)
            
            # 使用当前的 sigma 值进行边缘检测
            edges = edge(gray_img, sigma=self.sigma_var.get())
            
            # 应用细化处理（如果启用）
            self.edges = self.process_edges(edges)
            
            # 清除之前的图像
            self.ax1.clear()
            self.ax2.clear()
            
            # 重新显示原图和新的边缘检测结果
            self.ax1.set_title("原图")
            self.ax2.set_title("边缘检测")
            self.ax1.imshow(self.current_image)
            self.ax2.imshow(self.edges, cmap='gray')
            
            # 关闭坐标轴显示
            self.ax1.axis('off')
            self.ax2.axis('off')
            
            # 更新画布
            self.canvas.draw()
            
            print("Edge detection updated successfully")
            
        except Exception as e:
            print(f"Error updating edge detection: {str(e)}")
            messagebox.showerror("Error", f"Failed to update edge detection: {str(e)}")

    def __del__(self):
        """析构函数，确保资源被正确释放"""
        if hasattr(self, 'video_cap') and self.video_cap is not None:
            self.video_cap.release()

    def on_closing(self):
        """程序关闭时的清理工作"""
        if self.video_cap:
            self.video_cap.release()
        if self.is_processing:
            if messagebox.askyesno("Quit", "A process is running. Do you want to quit anyway?"):
                self.is_processing = False
                self.root.quit()
        else:
            self.root.quit()

    def show_help(self):
        """显示帮助信息"""
        help_text = """
形之声 (The Sound of Shape) - 视频边缘音频转换器

基本功能：
1. 图片处理
   - 打开图片：支持 JPG、PNG、BMP 等常见格式
   - 边缘检测：自动提取图片中的边缘
   - 保存结果：将处理后的边缘图保存为图片

2. 视频处理
   - 打开视频：支持 MP4、AVI、MOV 等常见格式
   - 转换为WAV：将视频边缘信息转换为音频

参数设置：
1. 边缘检测
   - 模糊度：控制边缘检测的精细程度（0.1-5.0）
   - 数值越大，边缘越模糊

2. 漫画模式
   - 启用：开启漫画线条优化
   - 阈值：控制边缘检测的敏感度（0-1）
   - 线条粗细：控制输出线条的粗细（1-5）
   - 平滑度：控制线条的平滑程度（0-1）

使用建议：
1. 普通图片/视频：
   - 模糊度：1.0
   - 漫画模式：关闭

2. 漫画/线稿：
   - 模糊度：0.5-1.0
   - 漫画模式：开启
   - 阈值：0.2-0.3
   - 线条粗细：1-2
   - 平滑度：0.7-0.9

注意事项：
1. 视频转换时间与视频长度成正比
2. 建议先用图片测试参数效果
3. 可以随时取消转换过程
4. 参数可以保存供下次使用
"""
        help_window = tk.Toplevel(self.root)
        help_window.title("使用说明")
        
        # 设置窗口大小和位置
        window_width = 500
        window_height = 600
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        help_window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # 创建文本框和滚动条
        text_frame = tk.Frame(help_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=text_widget.yview)
        
        # 插入帮助文本
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)  # 设置为只读

    def show_about(self):
        """显示关于信息"""
        about_text = """
形之声 (The Sound of Shape) v1.0

一个将视频边缘转换为音频的工具

特点：
- 支持图片和视频处理
- 漫画线条优化
- 实时预览效果
- 参数自定义

作者：[hal3000]
日期：2024
"""
        messagebox.showinfo("关于", about_text)

# 主函数入口
if __name__ == "__main__":
    root = tk.Tk()
    app = LocalImageProcessorApp(root)
    root.mainloop()
