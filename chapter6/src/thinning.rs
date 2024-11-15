
use anyhow::{Result, Error, Context};
use opencv::{
    prelude::*,
    core::{Mat, Size, Rect, Point, RNG, Vec3b, Scalar, Vector},
    imgcodecs,
    imgproc,
    ximgproc,
    highgui,
};

const BASE_PATH: &str = "../data/chapter6/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&(BASE_PATH.to_owned() + "LearnCV_black.png"), imgcodecs::IMREAD_ANYCOLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 创建英文文字和图形
    let mut words = Mat::zeros(100, 200, opencv::core::CV_8UC1)?.to_mat()?;
    imgproc::put_text(
        &mut words,
        "Learn",
        Point::new(30, 30),
        imgproc::FONT_HERSHEY_SIMPLEX,
        1.0,
        Scalar::all(255.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;

    imgproc::put_text(
        &mut words,
        "OpenCV 4",
        Point::new(30, 60),
        imgproc::FONT_HERSHEY_SIMPLEX,
        1.0,
        Scalar::all(255.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;
    imgproc::circle(&mut words, Point::new(80, 75), 10, Scalar::all(255.0), -1, imgproc::LINE_8, 0)?;
    imgproc::circle(&mut words, Point::new(130, 75), 10, Scalar::all(255.0), 3, imgproc::LINE_8, 0)?;

    // 进行细化
    let mut thin1 = Mat::default();
    let mut thin2 = Mat::default();
    ximgproc::thinning(&img, &mut thin1, ximgproc::THINNING_ZHANGSUEN)?;
    ximgproc::thinning(&words, &mut thin2, ximgproc::THINNING_ZHANGSUEN)?;

    // 显示处理结果
    highgui::imshow("thin1", &thin1)?;
    highgui::imshow("img", &img)?;
    highgui::named_window("thin2", highgui::WINDOW_NORMAL)?;
    highgui::imshow("thin2", &thin2)?;
    highgui::named_window("words", highgui::WINDOW_NORMAL)?;
    highgui::imshow("words", &words)?;

    highgui::wait_key(0)?;

    Ok(())
}
