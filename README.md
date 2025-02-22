# 形之声 (The Sound of Shape)

## 项目简介
**形之声** 是一个将视频边缘信息转换为音频的工具，同时支持图片处理和视频处理。该工具旨在通过边缘检测技术，将视觉信息转化为听觉信息，适合艺术创作、数据可视化以及实验性项目。

---

## 功能特性
### 1. 图片处理
- **打开图片**：支持 JPG、PNG、BMP 等常见格式。
- **边缘检测**：自动提取图片中的边缘。
- **保存结果**：将处理后的边缘图保存为图片文件。

### 2. 视频处理
- **打开视频**：支持 MP4、AVI、MOV 等格式。
- **转换为 WAV**：将视频的边缘信息转换为音频文件。

### 3. 参数自定义
- **边缘检测参数**：
  - 模糊度（Sigma）：控制边缘检测的精细程度。
- **漫画模式参数**：
  - 阈值：控制边缘检测的敏感度。
  - 线条粗细：调整输出线条的厚度。
  - 平滑度：优化线条的平滑程度。

---

## 使用方法
### 环境依赖
1. 安装 Python（推荐版本 >= 3.8）。
2. 安装必要的依赖库：
~~~
 pip install numpy matplotlib scikit-image opencv-python-headless pillow
~~~
### 启动程序
运行以下命令启动应用程序：
~~~
python 形之声_v1.py
~~~

### 操作步骤
1. 打开应用后，选择“文件”菜单以加载图片或视频。
2. 调整参数（如模糊度、阈值等）以优化边缘检测效果。
3. 点击“转换为 WAV”以生成音频文件。
4. 保存生成的结果到本地。

---

## 参数设置建议
| 场景         | 模糊度 (Sigma) | 阈值 (Threshold) | 线条粗细 | 平滑度 |
|--------------|----------------|------------------|----------|--------|
| 普通图片/视频 | 1.0            | -                | -        | -      |
| 漫画/线稿     | 0.5-1.0        | 0.2-0.3          | 1-2      | 0.7-0.9 |

---

## 注意事项
1. 视频转换时间与视频长度成正比，请耐心等待。
2. 建议先使用图片测试参数效果，再进行视频处理。
3. 参数设置可以保存并供下次使用。

---

## 关于作者
- **版本**：v1.0  
- **作者**：[hal3000]  
- **发布日期**：2024  
- **特点**：
  - 支持实时预览效果。
  - 漫画线条优化功能。
  - 自定义参数，适配多种场景。

---

## 开源协议
本项目基于 MIT 协议开源，欢迎自由使用和修改。若有任何问题或建议，请提交 Issue 或联系作者。

