use anyhow::{Context, Result, Error};
use opencv::{
    core::{Mat, Vector, Scalar, Point2d, Point2f, Point3f, Size},
    highgui,
    imgproc,
    imgcodecs,
    prelude::*,
};


const BASE_PATH: &str = "../data/chapter3/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&format!("{}lena.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    let mut rotation0 = Mat::default();
    let mut rotation1 = Mat::default();
    let mut img_warp0 = Mat::default();
    let mut img_warp1 = Mat::default();

    // 设置旋转角度、输出图像尺寸和旋转中心
    let angle = 30.0;
    let dst_size = Size::new(img.cols(), img.rows()); // 输出图像的尺寸为原图像的尺寸
    let center = Point2f::new(img.cols() as f32 / 2.0, img.rows() as f32 / 2.0);

    // 计算旋转矩阵
    rotation0 = imgproc::get_rotation_matrix_2d(center, angle, 1.0)?;

    // 进行仿射变换
    imgproc::warp_affine(&img, &mut img_warp0, &rotation0, dst_size, 1, 0, Scalar::all(0.0))?;

    // 显示第一个旋转后的图像
    highgui::imshow("img_warp0", &img_warp0)?;

    // 定义仿射变换的源点和目标点
    let src_points: Vec<Point2f> = vec![
        Point2f::new(0.0, 0.0),
        Point2f::new(0.0, (img.cols() - 1) as f32),
        Point2f::new((img.rows() - 1) as f32, (img.cols() - 1) as f32),
    ];
    let dst_points: Vec<Point2f> = vec![
        Point2f::new((img.rows() as f32) * 0.11, (img.cols() as f32) * 0.20),
        Point2f::new((img.rows() as f32) * 0.15, (img.cols() as f32) * 0.70),
        Point2f::new((img.rows() as f32) * 0.81, (img.cols() as f32) * 0.85),
    ];

    // 计算仿射变换矩阵
    // 将 Vec<Point2f> 转换为 Mat
    let src_points_mat = Mat::from_slice(src_points.as_slice())?.try_clone()?;
    let dst_points_mat = Mat::from_slice(dst_points.as_slice())?.try_clone()?;
    rotation1 = imgproc::get_affine_transform(&src_points_mat, &dst_points_mat)?;

    // 进行仿射变换
    imgproc::warp_affine(&img, &mut img_warp1, &rotation1, dst_size, 1, 0, Scalar::all(0.0))?;

    // 显示第二个仿射变换后的图像
    highgui::imshow("img_warp1", &img_warp1)?;

    // 等待按键
    highgui::wait_key(0)?;


    Ok(())
}