use anyhow::{Context, Error, Result};
use opencv::{
    core::{Mat, Size, Point, Vector, BorderTypes},
    highgui,
    imgcodecs,
    imgproc,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter5/";

pub(crate) fn run() -> Result<()> {
    // 读取三张图像
    let equal_lena = imgcodecs::imread(&(BASE_PATH.to_owned() + "equalLena.png"), imgcodecs::IMREAD_ANYDEPTH)?;
    let equal_lena_gauss = imgcodecs::imread(&(BASE_PATH.to_owned() + "equalLena_gauss.png"), imgcodecs::IMREAD_ANYDEPTH)?;
    let equal_lena_salt = imgcodecs::imread(&(BASE_PATH.to_owned() + "equalLena_salt.png"), imgcodecs::IMREAD_ANYDEPTH)?;

    // 确保图像读取成功
    if equal_lena.empty() || equal_lena_gauss.empty() || equal_lena_salt.empty() {
        println!("请确认图像文件名称是否正确");
        return Ok(());
    }

    let mut result_3 = Mat::default();
    let mut result_9 = Mat::default();
    let mut result_3gauss = Mat::default();
    let mut result_9gauss = Mat::default();
    let mut result_3salt = Mat::default();
    let mut result_9salt = Mat::default();

    // 调用均值滤波函数 blur 进行滤波, border_type=4, BorderTypes::BORDER_DEFAULT
    imgproc::blur(&equal_lena, &mut result_3, Size::new(3, 3), Point::new(-1, -1), 4)
        .context("均值滤波函数 blur1 调用失败").unwrap();
    imgproc::blur(&equal_lena, &mut result_9, Size::new(9, 9), Point::new(-1, -1), 4)
        .context("均值滤波函数 blur2 调用失败").unwrap();

    imgproc::blur(&equal_lena_gauss, &mut result_3gauss, Size::new(3, 3), Point::new(-1, -1), 4)
        .context("均值滤波函数 blur3 调用失败").unwrap();
    imgproc::blur(&equal_lena_gauss, &mut result_9gauss, Size::new(9, 9), Point::new(-1, -1), 4)
        .context("均值滤波函数 blur4 调用失败").unwrap();

    imgproc::blur(&equal_lena_salt, &mut result_3salt, Size::new(3, 3), Point::new(-1, -1), 4)
        .context("均值滤波函数 blur5 调用失败").unwrap();
    imgproc::blur(&equal_lena_salt, &mut result_9salt, Size::new(9, 9), Point::new(-1, -1), 4)
        .context("均值滤波函数 blur6 调用失败").unwrap();

    // 显示不含噪声图像
    highgui::named_window("equalLena", highgui::WINDOW_NORMAL)?;
    highgui::imshow("equalLena", &equal_lena)?;

    highgui::named_window("result_3", highgui::WINDOW_NORMAL)?;
    highgui::imshow("result_3", &result_3)?;

    highgui::named_window("result_9", highgui::WINDOW_NORMAL)?;
    highgui::imshow("result_9", &result_9)?;

    // 显示含有高斯噪声图像
    highgui::named_window("equalLena_gauss", highgui::WINDOW_NORMAL)?;
    highgui::imshow("equalLena_gauss", &equal_lena_gauss)?;

    highgui::named_window("result_3gauss", highgui::WINDOW_NORMAL)?;
    highgui::imshow("result_3gauss", &result_3gauss)?;

    highgui::named_window("result_9gauss", highgui::WINDOW_NORMAL)?;
    highgui::imshow("result_9gauss", &result_9gauss)?;

    // 显示含有椒盐噪声图像
    highgui::named_window("equalLena_salt", highgui::WINDOW_NORMAL)?;
    highgui::imshow("equalLena_salt", &equal_lena_salt)?;

    highgui::named_window("result_3salt", highgui::WINDOW_NORMAL)?;
    highgui::imshow("result_3salt", &result_3salt)?;

    highgui::named_window("result_9salt", highgui::WINDOW_NORMAL)?;
    highgui::imshow("result_9salt", &result_9salt)?;

    // 等待按键事件
    highgui::wait_key(0)?;

    Ok(())
}