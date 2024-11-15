use anyhow::Result;
use opencv::{
    prelude::*,
    core::{Mat, Size, Point, BorderTypes},
    imgcodecs,
    imgproc,
    highgui,
};

const BASE_PATH: &str = "../data/chapter5/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&(BASE_PATH.to_owned() + "equalLena.png"), imgcodecs::IMREAD_ANYDEPTH)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    let mut result_high = Mat::default();
    let mut result_low = Mat::default();
    let mut result_gauss = Mat::default();

    //大阈值检测图像边缘
    imgproc::canny(&img, &mut result_high, 100.0, 200.0, 3, false)?;

    //小阈值检测图像边缘
    imgproc::canny(&img, &mut result_low, 20.0, 40.0, 3, false)?;

    //高斯模糊后检测图像边缘
    imgproc::gaussian_blur(&img, &mut result_gauss, Size::new(3, 3), 5.0, 5.0, 4)?;
    imgproc::canny(&result_gauss.clone(), &mut result_gauss, 100.0, 200.0, 3, false)?;

    //显示图像
    highgui::imshow("resultHigh", &result_high)?;
    highgui::imshow("resultLow", &result_low)?;
    highgui::imshow("resultGauss", &result_gauss)?;
    highgui::wait_key(0)?;

    Ok(())
}