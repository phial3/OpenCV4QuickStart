use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter3/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let mut img = imgcodecs::imread(&(BASE_PATH.to_string() + "lena.png"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        println!("请确认图像文件名称是否正确");
        return Ok(());
    }

    let mut img_x = Mat::default();
    let mut img_y = Mat::default();
    let mut img_xy = Mat::default();

    opencv::core::flip(&img, &mut img_x, 0)?;  //沿x轴对称
    opencv::core::flip(&img, &mut img_y, 1)?;  //沿y轴对称
    opencv::core::flip(&img, &mut img_xy, -1)?; //先x轴对称，再y轴对称

    highgui::imshow("img", &img)?;
    highgui::imshow("img_x", &img_x)?;
    highgui:: imshow("img_y", &img_y)?;
    highgui::imshow("img_xy", &img_xy)?;

    highgui::wait_key(0)?;

    Ok(())
}