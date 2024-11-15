use anyhow::{Result, Error, Context};
use opencv::{
    prelude::*,
    core::{Mat, Vec3b, Size},
    imgcodecs,
    imgproc,
    highgui,
};
use rand::Rng;

const BASE_PATH: &str = "../data/chapter5/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let mut lena = imgcodecs::imread(&format!("{}{}", BASE_PATH, "lena.png"), imgcodecs::IMREAD_COLOR)?;
    let mut equal_lena = imgcodecs::imread(&format!("{}{}", BASE_PATH, "equalLena.png"), imgcodecs::IMREAD_ANYDEPTH)?;
    if lena.empty() || equal_lena.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 显示原始图像
    highgui::imshow("lena原图", &lena)?;
    highgui::imshow("equalLena原图", &equal_lena)?;

    // 添加椒盐噪声
    salt_and_pepper(&mut lena, 10000)?;      // 彩色图像添加椒盐噪声
    salt_and_pepper(&mut equal_lena, 10000)?;  // 灰度图像添加椒盐噪声

    // 显示添加噪声后的图像
    highgui::imshow("lena添加噪声", &lena)?;
    highgui::imshow("equalLena添加噪声", &equal_lena)?;

    highgui::wait_key(0)?;

    Ok(())
}


// 盐噪声函数
fn salt_and_pepper(image: &mut Mat, n: i32) -> Result<()> {
    let mut rng = rand::thread_rng();

    for _ in 0..n / 2 {
        // 随机确定图像中位置
        let i = rng.gen_range(0..image.cols());
        let j = rng.gen_range(0..image.rows());
        let write_black = rng.gen_bool(0.5);  // 50%概率

        let index = j * image.cols() + i;

        if !write_black {
            // 添加白色噪声
            if image.typ() == opencv::core::CV_8UC1 {
                // 处理灰度图像
                let pixel = image.at_mut::<u8>(index)?;
                *pixel = 255;
            } else if image.typ() == opencv::core::CV_8UC3 {
                // 处理彩色图像
                let pixel = image.at_mut::<Vec3b>(index)?;
                (*pixel)[0] = 255;  // B
                (*pixel)[1] = 255;  // G
                (*pixel)[2] = 255;  // R
            }
        } else {
            // 添加黑色噪声
            if image.typ() == opencv::core::CV_8UC1 {
                let pixel = image.at_mut::<u8>(index)?;
                *pixel = 0;
            } else if image.typ() == opencv::core::CV_8UC3 {
                let pixel = image.at_mut::<Vec3b>(index)?;
                (*pixel)[0] = 0;  // B
                (*pixel)[1] = 0;  // G
                (*pixel)[2] = 0;  // R
            }
        }
    }
    Ok(())
}