use anyhow::{Result, Error, Context};
use opencv::{
    prelude::*,
    core::{Mat, Size, Point, RNG, Vec3b, Vector},
    imgcodecs,
    imgproc,
    highgui,
};

const BASE_PATH: &str = "../data/chapter6/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&format!("{}{}", BASE_PATH, "rice.png"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    let mut rice = Mat::default();
    let mut rice_bw = Mat::default();

    // 将图像转成灰度图像
    imgproc::cvt_color(&img, &mut rice, imgproc::COLOR_BGR2GRAY, 0)?;

    // 二值化图像
    imgproc::threshold(&rice, &mut rice_bw, 50.0, 255.0, imgproc::THRESH_BINARY)?;

    // 生成随机颜色，用于区分不同连通域
    let mut rng = RNG::new(10086)?;
    let mut out = Mat::default();
    // 统计连通域的个数
    let number = imgproc::connected_components(&rice_bw, &mut out, 8, opencv::core::CV_16U)?;

    // 以不同颜色标记出不同的连通域
    let mut colors = Vec::with_capacity(number as usize);
    for _ in 0..number {
        // 使用均匀分布的随机数确定颜色
        let vec3 = Vec3b::from_array([
            rng.uniform(0, 256)? as u8,
            rng.uniform(0, 256)? as u8,
            rng.uniform(0, 256)? as u8,
        ]);
        colors.push(vec3);
    }

    // 创建结果图像
    let mut result = Mat::zeros_size(rice.size()?, img.typ())?.to_mat()?;
    let w = result.cols();
    let h = result.rows();

    for row in 0..h {
        for col in 0..w {
            let label = out.at_2d::<u16>(row, col)?;
            if *label == 0 { // 背景的黑色不改变
                continue;
            }
            let color = colors[*label as usize];
            result.at_mut::<Vec3b>(row * w + col)?.copy_from(&color).expect("copy_from failed");
        }
    }

    // 显示结果
    highgui::imshow("原图", &img)?;
    highgui::imshow("标记后的图像", &result)?;
    highgui::wait_key(0)?;

    Ok(())
}
