use anyhow::{Result, Error, Context};
use opencv::{
    prelude::*,
    core::{Mat, Size},
    imgcodecs,
    imgproc,
    highgui,
};

const BASE_PATH: &str = "../data/chapter5/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let gray = imgcodecs::imread(&format!("{}{}", BASE_PATH, "equalLena_salt.png"), imgcodecs::IMREAD_ANYCOLOR)?;
    let img = imgcodecs::imread(&format!("{}{}", BASE_PATH, "lena_salt.png"), imgcodecs::IMREAD_ANYCOLOR)?;
    // 检查图像是否成功加载
    if gray.empty() || img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 创建用于存储结果的矩阵
    let mut img_result3 = Mat::default();
    let mut gray_result3 = Mat::default();
    let mut img_result9 = Mat::default();
    let mut gray_result9 = Mat::default();

    // 对含有椒盐噪声的彩色和灰度图像进行中值滤波，滤波模板为3×3
    imgproc::median_blur(&img, &mut img_result3, 3)?;
    imgproc::median_blur(&gray, &mut gray_result3, 3)?;

    // 加大滤波模板，图像滤波结果会变模糊
    imgproc::median_blur(&img, &mut img_result9, 9)?;
    imgproc::median_blur(&gray, &mut gray_result9, 9)?;

    // 显示滤波处理结果
    highgui::imshow("img", &img)?;
    highgui::imshow("gray", &gray)?;
    highgui::imshow("imgResult3", &img_result3)?;
    highgui::imshow("grayResult3", &gray_result3)?;
    highgui::imshow("imgResult9", &img_result9)?;
    highgui::imshow("grayResult9", &gray_result9)?;

    // 等待按键
    highgui::wait_key(0)?;

    Ok(())
}