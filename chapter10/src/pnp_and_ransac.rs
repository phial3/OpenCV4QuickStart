use anyhow::{Result, Error, Context};
use opencv::{
    prelude::*,
    core::{Mat, Point2f, Point3f, Vec4f, Scalar, Size, Vector, TermCriteria},
    imgcodecs,
    imgproc,
    highgui,
    calib3d,
};

const BASE_PATH: &str = "../data/chapter10/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&(BASE_PATH.to_string() + "left01.jpg"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("读取图像错误，请确认图像文件是否正确");
    }

    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // 棋盘格的尺寸
    let board_size = Size::new(9, 6);
    let mut img_points = Vector::<Point2f>::new();
    calib3d::find_chessboard_corners(&gray, board_size, &mut img_points, calib3d::CALIB_CB_ADAPTIVE_THRESH)?;

    // 细化棋盘格的角点
    calib3d::find4_quad_corner_subpix(&gray, &mut img_points, Size::new(5, 5))?;

    // 生成每个棋盘格角点的三维空间坐标
    let square_size = Size::new(10, 10); // 每个方格的实际尺寸
    let mut point_sets = Vector::<Point3f>::new();
    for j in 0..board_size.height {
        for k in 0..board_size.width {
            point_sets.push(Point3f::new(
                (j * square_size.width) as f32,
                (k * square_size.height) as f32,
                0.0, // 假设标定板在世界坐标系的 z=0 平面
            ));
        }
    }

    // 输入已知的相机内参矩阵和畸变系数
    let camera_matrix = Mat::from_slice_2d(&[
        &[532.016297, 0.0, 332.172519],
        &[0.0, 531.565159, 233.388075],
        &[0.0, 0.0, 1.0],
    ])?;

    let dist_coeffs = Mat::from_slice(&[-0.285188, 0.080097, 0.001274, -0.002415, 0.106579])?.try_clone()?;

    // 使用 PnP 算法计算旋转向量和平移向量
    let mut rvec = Mat::default();
    let mut tvec = Mat::default();
    calib3d::solve_pnp(&point_sets, &img_points, &camera_matrix, &dist_coeffs, &mut rvec, &mut tvec, false, calib3d::SOLVEPNP_ITERATIVE)?;
    println!("世界坐标系变换到相机坐标系的旋转向量：{:?}", rvec);

    // 旋转向量转换成旋转矩阵
    let mut R = Mat::default();
    calib3d::rodrigues(&rvec, &mut R, &mut opencv::core::no_array())?;
    println!("旋转向量转换成旋转矩阵：{:?}", R);

    // 使用 PnP + RANSAC 算法计算旋转向量和平移向量
    let mut rvec_ransac = Mat::default();
    let mut tvec_ransac = Mat::default();
    calib3d::solve_pnp_ransac(&point_sets, &img_points,
                              &camera_matrix, &dist_coeffs,
                              &mut rvec_ransac, &mut tvec_ransac,
                              false, 100, 8.0, 0.99,
                              &mut opencv::core::no_array(), calib3d::SOLVEPNP_ITERATIVE)?;

    let mut R_ransac = Mat::default();
    calib3d::rodrigues(&rvec_ransac, &mut R_ransac, &mut opencv::core::no_array())?;
    println!("RANSAC 旋转向量转换成旋转矩阵：{:?}", R_ransac);

    highgui::wait_key(0)?;

    Ok(())
}