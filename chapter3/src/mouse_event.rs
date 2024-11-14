use anyhow::{Result, Error, Context};
use std::sync::{Arc, Mutex};
use opencv::{
    core::{Mat, Point, Scalar, Vec3b, Vector, MatTraitConst},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter3/";

#[derive(Debug)]
struct DrawState {
    img: Mat,
    img_point: Mat,
    pre_point: Point,
}

impl DrawState {
    fn new(image_path: &str) -> opencv::Result<Self> {
        let img = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;
        let mut img_point = Mat::default();
        img.copy_to(&mut img_point)?;

        Ok(Self {
            img,
            img_point,
            pre_point: Point::new(0, 0),
        })
    }

    fn handle_mouse_event(
        &mut self,
        event: i32,
        x: i32,
        y: i32,
        flags: i32,
    ) -> opencv::Result<()> {
        // 处理右键点击
        if event == highgui::EVENT_RBUTTONDOWN {
            println!("点击鼠标左键才可以绘制轨迹");
        }

        // 处理左键点击
        if event == highgui::EVENT_LBUTTONDOWN {
            self.pre_point = Point::new(x, y);
            println!("轨迹起始坐标: ({}, {})", x, y);
        }

        // 处理鼠标移动
        if event == highgui::EVENT_MOUSEMOVE && (flags & highgui::EVENT_FLAG_LBUTTON) != 0 {
            // 检查坐标是否在图像范围内
            if x > 0 && y > 0 && x < self.img_point.cols() && y < self.img_point.rows() {
                let red = Vec3b::from([0, 0, 255]);

                // 使用at_2d_mut安全地修改像素
                if let Ok(pixel) = self.img_point.at_2d_mut::<Vec3b>(y, x) {
                    *pixel = red;
                }
                if let Ok(pixel) = self.img_point.at_2d_mut::<Vec3b>(y, x - 1) {
                    *pixel = red;
                }
                if let Ok(pixel) = self.img_point.at_2d_mut::<Vec3b>(y, x + 1) {
                    *pixel = red;
                }
                if let Ok(pixel) = self.img_point.at_2d_mut::<Vec3b>(y + 1, x) {
                    *pixel = red;
                }
                if let Ok(pixel) = self.img_point.at_2d_mut::<Vec3b>(y - 1, x) {
                    *pixel = red;
                }

                highgui::imshow("图像窗口 2", &self.img_point)?;

                // 通过绘制直线显示鼠标移动轨迹
                let pt = Point::new(x, y);
                imgproc::line(
                    &mut self.img,
                    self.pre_point,
                    pt,
                    Scalar::new(0.0, 0.0, 255.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    0,
                )?;
                self.pre_point = pt;
                highgui::imshow("图像窗口 1", &self.img)?;
            }
        }

        Ok(())
    }
}

pub(crate) fn run() -> Result<()> {
    let image_path = format!("{}lena.png", BASE_PATH);

    // 创建状态管理器
    let state = Arc::new(Mutex::new(
        DrawState::new(&image_path).expect("Failed to create DrawState")
    ));

    // 显示初始图像
    {
        let state_guard = state.lock()?;
        highgui::imshow("图像窗口 1", &state_guard.img)?;
        highgui::imshow("图像窗口 2", &state_guard.img_point)?;
    }

    // 创建鼠标回调
    let state_clone = Arc::clone(&state);
    let mouse_callback = move |event: i32, x: i32, y: i32, flags: i32| {
        if let Ok(mut state_guard) = state_clone.lock() {
            if let Err(e) = state_guard.handle_mouse_event(event, x, y, flags) {
                eprintln!("Error handling mouse event: {}", e);
            }
        }
    };

    // 设置鼠标回调
    highgui::set_mouse_callback("图像窗口 1", Some(Box::new(mouse_callback)))?;

    // 等待按键
    loop {
        let key = highgui::wait_key(10)?;
        if key > 0 {
            break;
        }
    }

    Ok(())
}