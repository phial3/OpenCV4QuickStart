use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Vector, Scalar, Point, Rect},
    prelude::*,
    imgcodecs,
    imgproc,
    highgui,
};

const BASE_PATH: &str = "../data/chapter3/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&format!("{}lena.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        println!("请确认图像文件名称是否正确");
        return Ok(());
    }

    let mut gauss: Vector<Mat> = Vector::new();     // 高斯金字塔
    let mut laplace: Vector<Mat> = Vector::new();   // 拉普拉斯金字塔
    let level = 3; // 高斯金字塔下采样次数

    // 将原图作为高斯金字塔的第0层
    gauss.push(img);

    // 构建高斯金字塔
    for i in 0..level {
        let mut gauss_img = Mat::default();
        imgproc::pyr_down(
            &gauss.get(i)?,
            &mut gauss_img,
            opencv::core::Size::default(),
            opencv::core::BORDER_DEFAULT,
        )?;
        gauss.push(gauss_img);
    }

    // 构建拉普拉斯金字塔
    for i in (0..gauss.len()).rev() {
        if i == 0 {
            break;
        }

        let mut up_gauss = Mat::default();
        let mut lap_img: Mat;

        if i == gauss.len() - 1 {
            // 如果是高斯金字塔中的最上面一层图像
            let mut down = Mat::default();
            imgproc::pyr_down(
                &gauss.get(i)?,
                &mut down,
                opencv::core::Size::default(),
                opencv::core::BORDER_DEFAULT,
            )?;
            imgproc::pyr_up(
                &down,
                &mut up_gauss,
                opencv::core::Size::new(gauss.get(i)?.cols(), gauss.get(i)?.rows()),
                opencv::core::BORDER_DEFAULT,
            )?;
            lap_img = Mat::default();
            opencv::core::subtract(&gauss.get(i)?, &up_gauss, &mut lap_img, &Mat::default(), -1)?;
            laplace.push(lap_img);
        }

        imgproc::pyr_up(
            &gauss.get(i)?,
            &mut up_gauss,
            opencv::core::Size::new(gauss.get(i-1)?.cols(), gauss.get(i-1)?.rows()),
            opencv::core::BORDER_DEFAULT,
        )?;
        lap_img = Mat::default();
        opencv::core::subtract(&gauss.get(i-1)?, &up_gauss, &mut lap_img, &Mat::default(), -1)?;
        laplace.push(lap_img);
    }

    // 查看两个金字塔中的图像
    for i in 0..gauss.len() {
        let window_name_gauss = format!("G{}", i);
        let window_name_lap = format!("L{}", i);

        highgui::imshow(&window_name_gauss, &gauss.get(i)?)?;
        if i < laplace.len() {
            highgui::imshow(&window_name_lap, &laplace.get(i)?)?;
        }
    }

    highgui::wait_key(0)?;

    Ok(())
}