use anyhow::{Context, Result, Error};
use opencv::{
    core::{Mat, Vector, Scalar, Point2d, Point2f, Size},
    highgui,
    imgproc,
    imgcodecs,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter3/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&format!("{}noobcvqr.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        println!("请确认图像文件名称是否正确");
        return Ok(());
    }

    // 定义源点和目标点的数组
    let src_points = vec![
        Point2f::new(94.0, 374.0),
        Point2f::new(507.0, 380.0),
        Point2f::new(1.0, 623.0),
        Point2f::new(627.0, 627.0),
    ];

    let dst_points = vec![
        Point2f::new(0.0, 0.0),
        Point2f::new(627.0, 0.0),
        Point2f::new(0.0, 627.0),
        Point2f::new(627.0, 627.0),
    ];

    // 将 Vec<Point2f> 转换为 Mat 类型
    let src_points_mat = Mat::from_slice(src_points.as_slice())?.try_clone()?;
    let dst_points_mat = Mat::from_slice(dst_points.as_slice())?.try_clone()?;
    // 计算透视变换矩阵
    // solve_method: DecompTypes::DECOMP_LU
    let rotation = imgproc::get_perspective_transform(&src_points_mat, &dst_points_mat, 0)?;

    // 创建输出图像
    let mut img_warp = Mat::default();

    // 进行透视变换
    imgproc::warp_perspective(&img, &mut img_warp, &rotation, Size::new(img.cols(), img.rows()), 1, 0, Scalar::all(0.0))?;

    // 显示图像
    highgui::imshow("img", &img)?;
    highgui::imshow("img_warp", &img_warp)?;

    // 等待按键
    highgui::wait_key(0)?;

    Ok(())
}