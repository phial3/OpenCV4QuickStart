use anyhow::{Context, Result, Error};
use opencv::{
    core::{Mat, Scalar},
    highgui,
    imgcodecs,
    prelude::*,
};
use std::sync::{Arc, Mutex};

struct ImageProcessor {
    original_image: Mat,
    processed_image: Mat,
    brightness: i32,
}

impl ImageProcessor {
    fn new(image_path: &str) -> Result<Self> {
        let img = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR).context("无法读取图像")?;

        if img.empty() {
            anyhow::bail!("图像为空，请确认文件路径是否正确");
        }

        Ok(Self {
            original_image: img,
            processed_image: Mat::default(),
            brightness: 100, // 默认亮度
        })
    }

    fn update_brightness(&mut self) {
        let scale = self.brightness as f32 / 100.0;
        opencv::core::multiply(
            &self.original_image,
            &Scalar::all(scale as f64),
            &mut self.processed_image,
            1.0,
            -1
        ).expect("调整亮度失败");
    }

    fn display_image(&self, win_name: &str) {
        highgui::imshow(win_name, &self.processed_image).expect("显示图像失败");
    }
}

pub(crate) fn run() -> Result<()> {
    const BASE_PATH: &str = "../data/chapter3/lena.png";

    let mut processor = ImageProcessor::new(BASE_PATH)?;

    // 创建窗口
    let window_name = "滑动条改变图像亮度";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE).context("创建窗口失败")?;
    // 显示原始图像
    highgui::imshow(window_name, &processor.original_image.clone()).expect("显示图像失败");

    // 回调函数
    let processor_arc = Arc::new(Mutex::new(processor));

    let on_trackbar: Box<dyn FnMut(i32) + Send + Sync> = {
        let processor_clone = Arc::clone(&processor_arc);
        Box::new(move |value: i32| {
            let mut guard = processor_clone.lock().unwrap();
            guard.brightness = value;
            guard.update_brightness();
            guard.display_image(window_name);
        })
    };

    // 创建滑动条
    highgui::create_trackbar(
        "亮度值百分比",
        window_name,
        Some(&mut processor_arc.lock().unwrap().brightness),
        600,
        Some(on_trackbar),
    ).context("创建滑动条失败")?;

    // 等待用户输入
    highgui::wait_key(0).context("等待按键失败")?;

    Ok(())
}