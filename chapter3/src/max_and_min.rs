use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Rect, Scalar, Size, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter3/";

pub(crate) fn run() -> Result<()> {
    // 创建两个单通道矩阵并填充数据
    let mut a: [f32; 12] = [1.0, 2.0, 3.3, 4.0, 5.0, 9.0, 5.0, 7.0, 8.2, 9.0, 10.0, 2.0];
    let mut b: [f32; 12] = [1.0, 2.2, 3.0, 1.0, 3.0, 10.0, 6.0, 7.0, 8.0, 9.3, 10.0, 1.0];

    let imga = unsafe { Mat::new_rows_cols_with_data_unsafe_def(3, 4, opencv::core::CV_32FC1, a.as_ptr() as *mut _)? };
    let imgb = unsafe { Mat::new_rows_cols_with_data_unsafe_def(3, 4, opencv::core::CV_32FC1, b.as_ptr() as *mut _)? };

    // 对两个单通道矩阵进行比较运算
    let mut my_max = Mat::default();
    let mut my_min = Mat::default();
    opencv::core::min(&imga, &imgb, &mut my_min)?;
    opencv::core::max(&imga, &imgb, &mut my_max)?;

    // 创建两个多通道矩阵
    let imgas =  unsafe { Mat::new_rows_cols_with_data_unsafe_def(2, 3, opencv::core::CV_32FC2, a.as_ptr() as *mut _)? };
    let imgbs =  unsafe { Mat::new_rows_cols_with_data_unsafe_def(2, 3, opencv::core::CV_32FC2, b.as_ptr() as *mut _)? };

    // 对两个多通道矩阵进行比较运算
    let mut my_maxs = Mat::default();
    let mut my_mins = Mat::default();
    opencv::core::min(&imgas, &imgbs, &mut my_mins)?;
    opencv::core::max(&imgas, &imgbs, &mut my_maxs)?;

    // 读取彩色图像并进行比较运算
    let img0 = imgcodecs::imread(&(BASE_PATH.to_string() + "lena.png"), imgcodecs::IMREAD_COLOR)?;
    let img1 = imgcodecs::imread(&(BASE_PATH.to_string() + "noobcv.jpg"), imgcodecs::IMREAD_COLOR)?;

    if img0.empty() || img1.empty() {
        println!("请确认图像文件名称是否正确");
        return Ok(());
    }

    let mut com_min = Mat::default();
    let mut com_max = Mat::default();
    opencv::core::min(&img0, &img1, &mut com_min)?;
    opencv::core::max(&img0, &img1, &mut com_max)?;

    // 显示比较结果
    highgui::imshow("comMin", &com_min)?;
    highgui::imshow("comMax", &com_max)?;

    // 与掩模进行比较运算
    let mut src1 = Mat::zeros(512, 512, opencv::core::CV_8UC3)?.to_mat()?;
    let rect = Rect::new(100, 100, 300, 300);
    let _ = src1.roi(rect)?.try_clone()?.set_to(&Scalar::all(255.0), &opencv::core::no_array())?;

    let mut comsrc1 = Mat::default();
    opencv::core::min(&img0, &src1, &mut comsrc1)?;
    highgui::imshow("comsrc1", &comsrc1)?;

    let mut src2 = Mat::new_size_with_default(
        Size::new(512, 512),
        opencv::core::CV_8UC3,
        Scalar::new(0.0, 0.0, 255.0, 0.0),
    )?;
    src2.set_to(&Scalar::new(0.0, 0.0, 255.0, 0.0), &opencv::core::no_array())?;

    let mut comsrc2 = Mat::default();
    opencv::core::min(&img0, &src2, &mut comsrc2)?;
    highgui::imshow("comsrc2", &comsrc2)?;

    // 对两张灰度图像进行比较运算
    let mut img0g = Mat::default();
    let mut img1g = Mat::default();
    let mut com_ming = Mat::default();
    let mut com_maxg = Mat::default();

    imgproc::cvt_color(&img0, &mut img0g, imgproc::COLOR_BGR2GRAY, 0)?;
    imgproc::cvt_color(&img1, &mut img1g, imgproc::COLOR_BGR2GRAY, 0)?;

    opencv::core::max(&img0g, &img1g, &mut com_maxg)?;
    opencv::core::min(&img0g, &img1g, &mut com_ming)?;

    highgui::imshow("comMinG", &com_ming)?;
    highgui::imshow("comMaxG", &com_maxg)?;

    // 等待按键并退出
    highgui::wait_key(0)?;

    Ok(())
}