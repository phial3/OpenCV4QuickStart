use anyhow::Result;
use opencv::{
    prelude::*,
    core::{Mat, Size},
    imgcodecs,
    imgproc,
    highgui,
};

const BASE_PATH: &str = "../data/chapter3/";

pub(crate) fn run() -> Result<()> {
    // 读取灰度图像
    let gray_img = imgcodecs::imread(&format!("{}lena.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if gray_img.empty() {
        println!("请确认图像文件名称是否正确");
        return Ok(());
    }

    let mut small_img = Mat::default();
    let mut big_img0 = Mat::default();
    let mut big_img1 = Mat::default();
    let mut big_img2 = Mat::default();

    imgproc::resize(&gray_img, &mut small_img, Size::new(15, 15), 0.0, 0.0, imgproc::INTER_AREA)?;  //先将图像缩小
    imgproc::resize(&small_img, &mut big_img0, Size::new(30, 30), 0.0, 0.0, imgproc::INTER_NEAREST)?;  //最近邻差值
    imgproc::resize(&small_img, &mut big_img1, Size::new(30, 30), 0.0, 0.0, imgproc::INTER_LINEAR)?;  //双线性差值
    imgproc::resize(&small_img, &mut big_img2, Size::new(30, 30), 0.0, 0.0, imgproc::INTER_CUBIC)?;  //双三次差值

    //图像尺寸太小，一定要设置可以调节窗口大小标志
    highgui::named_window("small_img", highgui::WINDOW_NORMAL)?;
    highgui::imshow("small_img", &small_img)?;

    highgui::named_window("big_img0", highgui::WINDOW_NORMAL)?;
    highgui::imshow("big_img0", &big_img0)?;

    highgui::named_window("big_img1", highgui::WINDOW_NORMAL)?;
    highgui::imshow("big_img1", &big_img1)?;

    highgui::named_window("big_img2", highgui::WINDOW_NORMAL)?;
    highgui::imshow("big_img2", &big_img2)?;

    highgui::wait_key(0)?;

    Ok(())
}